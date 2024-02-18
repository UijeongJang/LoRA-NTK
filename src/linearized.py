
##############################################################################################################################################
########## This code is modified from github repo of Malladi et al.(2023) https://github.com/princeton-nlp/LM-Kernel-FT/tree/main  ###########
##############################################################################################################################################

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler

from functorch import vmap, jvp, jacrev, make_functional_with_buffers

import transformers
from transformers.data.data_collator import DataCollator
from torch.utils.tensorboard import SummaryWriter
from transformers.file_utils import is_torch_tpu_available

from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.trainer import SequentialDistributedSampler
import torch.distributed as dist
from transformers.trainer_utils import PredictionOutput, EvalPrediction
from transformers.utils import logging
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
import gc
from transformers.trainer_utils import TrainOutput

from src.linearhead_trainer import varsize_tensor_all_gather, LinearHeadTrainer

import numpy as np
from tqdm import tqdm


logger = logging.get_logger(__name__)

class LogitModelWrapper(nn.Module):
    def __init__(self, model, binary_classification):
        super().__init__()
        self.model = model
        self.binary_classification = binary_classification

    def forward(self, input_ids, attention_mask, mask_pos):
        logits = self.model(input_ids, attention_mask, mask_pos=mask_pos)[0] # don't provide labels
        if self.binary_classification:
            assert logits.size(1) == 2, "--binary_classification should have 2 logits"
            logits = (logits[:,1] - logits[:,0]).unsqueeze(-1)   
        return logits
        
# Linearized LoRA update  
class LinearLoraupdate(nn.Module):
    def __init__(self,model,binary_classification,target_layers,target_size):
        super().__init__()
        self.model = model
        self.binary_classification = binary_classification
        self.model_wrapper = LogitModelWrapper(model, binary_classification)
        self.target_layers = target_layers
        self.target_size = target_size
        
        self.Lora_a = [torch.empty(model.model_args.lora_r, self.target_size[i][1]) for i in range(len(self.target_layers))]
        self.Lora_b = [torch.zeros(self.target_size[i][0], model.model_args.lora_r) for i in range(len(self.target_layers))]
        for i, param in enumerate(self.Lora_a):
           torch.nn.init.normal_(param, mean = 0, std= 1/((np.sqrt(model.model_args.lora_r)*self.target_size[i][1])))  # scaling the initailization by \frac{1}{sqrt{r}}
           #torch.nn.init.kaiming_normal_(param)
      
        self.Lora_A_list = nn.ParameterList([nn.Parameter(param) for param in self.Lora_a])
        self.Lora_B_list = nn.ParameterList([nn.Parameter(param) for param in self.Lora_b])     
            
        self.lora_r = model.model_args.lora_r
        self.scaling = model.model_args.lora_alpha
        self.num_labels = None
        self.gradient_dtype = None
        
    def forward(self, input_ids, attention_mask, mask_pos, gradient):  #gradient = list of gradients, each element corresponds to each Lora layer
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, mask_pos=mask_pos)[0] # don't provide labels
            if self.binary_classification:
                assert logits.size(1) == 2, "--binary_classification should have 2 logits"
                logits = (logits[:,1] - logits[:,0]).unsqueeze(-1)
        
        self.num_labels = gradient[0].size(1) 
        self.gradient_dtype = gradient[0].dtype

        # Compute \langle G(X_i), B*A \rangle 
        output = sum(torch.sum((self.Lora_B_list[i]@self.Lora_A_list[i]) * gradient[i], dim=(-2,-1)) for i in range(len(self.target_layers)) ) 
        
        return output
       
# Linearized LoRA update 
class Linearupdate(nn.Module):
    def __init__(self,model,binary_classification,target_layers,target_size):
        super().__init__()
        self.model = model
        self.binary_classification = binary_classification
        self.model_wrapper = LogitModelWrapper(model, binary_classification)
        self.target_layers = target_layers
        self.target_size = target_size
        
        self.delta = [torch.zeros(self.target_size[i][0], self.target_size[i][1]) for i in range(len(self.target_layers))]
        self.delta_list = nn.ParameterList([nn.Parameter(param) for param in self.delta])
       
        self.lora_r = model.model_args.lora_r
        self.scaling = model.model_args.lora_alpha
        self.num_labels = None
        self.gradient_dtype = None
        
    
    def forward(self, input_ids, attention_mask, mask_pos, gradient):  #gradient = list of gradients, each element corresponds to each Lora layer
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, mask_pos=mask_pos)[0] # don't provide labels
            if self.binary_classification:
                assert logits.size(1) == 2, "--binary_classification should have 2 logits"
                logits = (logits[:,1] - logits[:,0]).unsqueeze(-1)
        
        self.num_labels = gradient[0].size(1) #train_gradient = [Batch, label, 768 , 768]
        self.gradient_dtype = gradient[0].dtype

        #Minimizing nuclear norm is equivalent to minimizing the sum of nuclaer norm of nontrivial diagnoals. 
        
        output = sum(torch.sum(self.delta_list[i] * gradient[i], dim=(-2,-1)) for i in range(len(self.target_layers)) ) 
        
        return output
               


def param_to_buffer(module, module_name, predicate):
    """Turns all parameters of a module into buffers."""
    modules = module.named_modules(prefix=str(module_name))
    next(modules) # Skip itself

    params = []
    for name, param in module.named_parameters(recurse=False, prefix=str(module_name)):
        if predicate(name):
            params.append((name.split(".")[-1], param))

    for name, param in params:
        delattr(module, name) # Unregister parameter
        module.register_buffer(name, param)
    for name, module in modules:
        param_to_buffer(module, name, predicate)
        
def buffers_to_params(module):
    for name, buf in list(module.named_buffers()):
        module.register_parameter(name, torch.nn.Parameter(buf))
        del module._buffers[name]
        

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


class LinearizedLoraTrainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        *posargs,
        **kwargs
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, *posargs, **kwargs)

        self.grad_dim = None
        self.train_targets = None
        self.num_labels = None
        self.gradient_dtype = None
 
        self.binary_classification = self.args.binary_classification
        self.model_wrapper = LogitModelWrapper(self.model, self.args.binary_classification)
        
        self.target_layers=[]
        self.target_size=[]
        for name, param in self.model_wrapper.named_parameters():
            
            if self.args.train_last_layer:
                # There are 12 layers in RoBERTa-base model
                if  "11" in name and "attention" in name and "query" in name and "weight" in name :
                    self.target_layers.append(name)
                    self.target_size.append(param.size())
                if  "11" in name and "attention" in name and "value" in name and "weight" in name :
                    self.target_layers.append(name)
                    self.target_size.append(param.size())
            else:
                if  "attention" in name and "query" in name and "weight" in name :
                    self.target_layers.append(name)
                    self.target_size.append(param.size())
                if  "attention" in name and "value" in name and "weight" in name :
                    self.target_layers.append(name)
                    self.target_size.append(param.size())
                
        logger.info("Calculating gradient with respect to {}".format(self.target_layers))
        
        #lora
        if self.model.model_args.apply_lora == True:
            self.lora_model = LinearLoraupdate(self.model,self.args.binary_classification,self.target_layers,self.target_size).to(self.args.device)
        else:
            self.lora_model = Linearupdate(self.model,self.args.binary_classification,self.target_layers,self.target_size).to(self.args.device)
        
    def get_unshuffled_dataloader(self, dataset: Optional[Dataset] = None, sharded: bool = False, batch_size: Optional[int] = -1):
        if sharded and is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif sharded and self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        bs = self.args.per_device_eval_batch_size if batch_size == -1 else batch_size
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=bs,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def profile_memory(self):
        import gc
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    print(type(obj), obj.size())
            except:
                pass

    ## Method to compute gradient 1
    def compute_gradient_perlayer(self, inputs_outer, layer_name ):
            
        def convert_to_buffer(name):
            if layer_name in name:
                #logger.info("Including {}".format(name))
                return False
            else:
                return True
        
        model_tmp = copy.deepcopy(self.model_wrapper)
        param_to_buffer(model_tmp, "", convert_to_buffer)

        model_tmp.eval()
  
        for name , param in model_tmp.named_parameters():
            param.requires_grad_(True)

        model_fn, params, buffers = make_functional_with_buffers(model_tmp)

        jacobian_fn = jacrev(model_fn)

        def curried_jacobian_fn(input_ids, attention_mask, mask_pos):
            return jacobian_fn(params, buffers, input_ids, attention_mask, mask_pos)

        targets = []
        for k, v in inputs_outer.items():
            if isinstance(v, torch.Tensor):
                inputs_outer[k] = v.to(self.args.device)
                
        grads_outer = curried_jacobian_fn(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"))[0]
    
            
        label = inputs_outer.get("labels")
        if self.args.binary_classification:
            label = (label * 2 - 1).float(0)
            
        targets.append(label)
    
        return (grads_outer, torch.cat(targets, dim=0) if targets else torch.tensor([]))  
    
    ## Method to compute gradient 2
    def compute_gradient_everylayer(self, inputs_outer):
        
        grads_outer=[] 
        with torch.no_grad():
            for layer_name in self.target_layers:   
                    grads_outer_layer, targets = self.compute_gradient_perlayer(inputs_outer, layer_name)
                    grads_outer.append(grads_outer_layer)
                    
        if self.grad_dim is None:
            self.grad_dim = sum(np.prod(x.shape[2:]) for x in grads_outer)
                
        return (grads_outer, targets) #list of gradients for each layer and targets.
    
    def compute_gradient_sharded(self, inputs_outer):
        with torch.no_grad():
            grads_outer , targets = self.compute_gradient_everylayer(inputs_outer)
        return  grads_outer, targets 

    #get pre-trained logits and training targets , this corresponds to f(X_i), Y_i
    def compute_model_logits(self, inputs_outer):

        self.model_wrapper.eval()
        
        logits = []
        targets = []
        with torch.no_grad():
            for k, v in inputs_outer.items():
                if isinstance(v, torch.Tensor):
                    inputs_outer[k] = v.to(self.args.device)
            label = inputs_outer.get("labels")
            if self.args.binary_classification:
                label = (label * 2 - 1).float()  # convert from {0, 1} to {-1, 1}
            preds = self.model_wrapper(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"))
            logits.append(preds.detach())
            targets.append(label)

        logits = torch.cat(logits, dim=0)
        targets = torch.cat(targets, dim=0)

        return logits, targets 

    def train(self, model_path=None, dev_objective=None):
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer
        eval_dataset = self.train_dataset
        return TrainOutput(0, 0.0, {}), None
    
    # Fine-tune linearized model
    def finetune(self, train_dataset, eval_dataset):
        
        dataloader_outer = self.get_unshuffled_dataloader(train_dataset, sharded=True, batch_size=self.args.per_device_train_batch_size)
        dataloader_outer_eval = self.get_unshuffled_dataloader(eval_dataset, sharded=True, batch_size=self.args.per_device_eval_batch_size)
        optimizer = optim.SGD(self.lora_model.parameters(), lr=self.args.linear_lr) # Weight decay will be implented manually

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(self.lora_model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(self.lora_model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                self.lora_model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        
        epoch_count = 0 
        
        #Make sure to freeze other parameters
        if self.model.model_args.apply_lora:
            for name, param in self.lora_model.named_parameters():
                if "Lora" not in name:
                    param.requires_grad_(False)
                if self.args.linear_freeze_A and "Lora_A" in name:
                    param.requires_grad_(False)
        else:
            for name, param in self.lora_model.named_parameters():
                if "delta" not in name:
                    param.requires_grad_(False)
         
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad:
                print(f"{name} is being trained")
                
                
        writer = SummaryWriter(f"./finallasttrain/{self.model.model_args.apply_lora}-{self.args.linear_lr}-{self.model.data_args.task_name}-{self.args.seed}-{self.model.model_args.lora_r}")
        
        file_exists = False
        eval_file_exists = False
        saved_gradients_eval = []
        saved_gradients = []
        for epoch in range(self.args.linear_num_epoch):
            
            epoch_count += 1
            reg = 0
            total_loss = 0
            
            if self.args.eval_during_training:
                total_loss_eval = 0
                eval_preds=[]
                eval_targets_list=[]
                
            for i, inputs_outer in enumerate(tqdm(dataloader_outer, desc="Fine-tuning")):
                if file_exists:
                    gradient = saved_gradients[i]
                    for j in range(len(self.target_layers)):
                        gradient[j].requires_grad_(False)
                else:
                    gradient, _  = self.compute_gradient_sharded(inputs_outer)
                    saved_gradients.append(gradient)
                    for j in range(len(self.target_layers)):
                        gradient[j].requires_grad_(False)
                    
                
                if self.num_labels is None:
                    self.num_labels = gradient[0].size(1) 
                    self.gradient_dtype = gradient[0].dtype
                    
                train_logits , targets = self.compute_model_logits(inputs_outer)
                train_logits = train_logits.to(self.args.device)

                output = model(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"), gradient)

                loss = nn.CrossEntropyLoss(reduction = 'sum')(train_logits + output  , targets)   
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                ## Weight decay
                if not self.model.model_args.apply_lora:
                    for name, param in self.lora_model.named_parameters():
                        if "delta" in name and self.args.linear_wd>=0.0000001: #Skip this if there is no weight decay (weight decay = 0)
                            u,s,v = torch.svd(param)
                            s = torch.nn.Threshold(0, 0)(s-  self.args.linear_lr * self.args.linear_wd)  #Soft-thresholding operator 
                            param = u@s@v
                if self.model.model_args.apply_lora:
                    for name, param in self.lora_model.named_parameters():
                        if "Lora" in name and self.args.linear_wd>=0.00000001:   #Skip this if there is no weight decay (weight decay = 0)
                            param = param -  self.args.linear_wd * self.args.linear_lr * param
                
                if self.model.model_args.apply_lora:
                    for name, param in self.lora_model.named_parameters():
                        if "Lora" in name and self.args.linear_wd>=0.0000001:  #Skip this if there is no weight decay (weight decay = 0)
                            reg += (1/2)*(param ** 2).sum()  
                else:
                    for name, param in self.lora_model.named_parameters():
                        if "delta" in name and self.args.linear_wd>=0.0000001:   #Skip this if there is no weight decay (weight decay = 0)
                            reg += torch.norm(param, p = 'nuc')
                            
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                total_loss += loss.item()
                            
            file_exists = True
                          
            avg_loss = (total_loss/ len(dataloader_outer.dataset)) + (reg * self.args.linear_wd) 
            logger.info(f"epoch : {epoch+1} train_loss : {avg_loss}")
            writer.add_scalar(f"train_loss_{self.model.data_args.task_name}/epoch", avg_loss, epoch)
            
            # Do evaluation during training if needed.
            if self.args.eval_during_training: 
                with torch.no_grad():
                    
                    for i, inputs_outer in enumerate(tqdm(dataloader_outer_eval, desc="Evaluating")):
                               
                        if eval_file_exists:
                            gradient_eval = saved_gradients_eval[i]
                            for j in range(len(self.target_layers)):
                                gradient_eval[j].to(self.args.device)
                                gradient_eval[j].requires_grad_(False)
                                
                        else:
                            gradient_eval, _  = self.compute_gradient_sharded(inputs_outer)
                            for j in range(len(self.target_layers)):  
                                gradient_eval[j].requires_grad_(False)
                        
                        
                        output_eval = model(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"), gradient_eval)
                        eval_logits, eval_targets = self.compute_model_logits(inputs_outer)
                        eval_logits = eval_logits.to(self.args.device)
                            
                        loss_eval = nn.CrossEntropyLoss(reduction = 'sum')(output_eval + eval_logits , eval_targets) 
                        dist.all_reduce(loss_eval, op=dist.ReduceOp.SUM) 
                        total_loss_eval += loss_eval.item()
                        
                        eval_preds.append( eval_logits + output_eval )
                        eval_targets_list.append(eval_targets)
                        
                        saved_gradients_eval_cpu = []
                        for j in range(len(self.target_layers)):
                            gradient_eval_layer= gradient_eval[j].detach().cpu()
                            saved_gradients_eval_cpu.append(gradient_eval_layer)
                        saved_gradients_eval.append(saved_gradients_eval_cpu)
                        
                       
                eval_file_exists = True

                avg_loss_eval = (total_loss_eval / len(dataloader_outer_eval.dataset) )+ (reg * self.args.linear_wd)
                logger.info(f"epoch : {epoch+1} eval_loss : {avg_loss_eval}")
                writer.add_scalar(f"eval_loss_{self.model.data_args.task_name}/epoch", avg_loss_eval, epoch)
                
                eval_preds = torch.cat(eval_preds, dim=0)
                eval_targets = torch.cat(eval_targets_list, dim=0)

                if self.args.local_rank != -1:
                    logger.info("Starting to gather kernel across GPUs")
                    eval_preds = varsize_tensor_all_gather(eval_preds.to(self.args.device), torch.distributed.get_world_size())
                    eval_targets = varsize_tensor_all_gather(eval_targets.to(self.args.device), torch.distributed.get_world_size())
                    logger.info("Finished gathering kernel across GPUs")

                # Now calculate the accuarcies
                metrics = None
                eval_preds = eval_preds.cpu()
                eval_targets = eval_targets.cpu()
                
                if self.args.binary_classification: # Make sure to compute loss before this transformation!
                    eval_preds = torch.cat([-eval_preds, eval_preds], dim=-1) # convert back to two logits
                    eval_targets = ((eval_targets + 1) / 2).long() # convert back from {-1, 1} to {0, 1}

                if self.compute_metrics is not None:
                    metrics = self.compute_metrics(EvalPrediction(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy()))

                # Prefix all keys with eval_
                for key in list(metrics.keys()):
                    if not key.startswith("eval_"):
                        metrics[f"eval_{key}"] = metrics.pop(key)
                metrics["eval_loss"] = avg_loss

                metrics.update(self.metrics())
                metrics["grad_dim"] = self.grad_dim

                output = PredictionOutput(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy(), metrics=metrics)
                metrics = output.metrics
                objective = default_dev_objective(metrics)
                logger.info(f"epoch : {epoch+1}  objective : {objective}")
                writer.add_scalar(f"Eval_acc_{self.model.data_args.task_name}/epoch", objective, epoch)
                         
        writer.flush()
        writer.close()
        self.save_model(self.args.output_dir)
        return avg_loss.item() , epoch_count
    
    
    def metrics(self):
        return {}
        
    def evaluate_gpu(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        
        dataloader_outer = self.get_unshuffled_dataloader(eval_dataset, sharded=True, batch_size=self.args.per_device_eval_batch_size)
        
        metrics = {}
        total_loss=0
        reg = 0
        eval_targets=[]
        eval_preds=[]
            
        with torch.no_grad():
            for inputs_outer in tqdm(dataloader_outer, desc="Evaluating"):
                gradient, targets = self.compute_gradient_sharded(inputs_outer)
                
                if self.num_labels is None:
                    self.num_labels = gradient[0].size(1) 
                    self.gradient_dtype = gradient[0].dtype

                output = self.lora_model(inputs_outer.get("input_ids"), inputs_outer.get("attention_mask"), inputs_outer.get("mask_pos"), gradient)
                eval_logits, _ = self.compute_model_logits(inputs_outer)
                eval_targets_coords = nn.functional.one_hot(targets.flatten(), self.num_labels).to(self.gradient_dtype) #[B , L]
                
                loss = nn.CrossEntropyLoss(reduction='sum')(output + eval_logits , targets) 
                dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
                total_loss += loss.item()
                eval_targets.append(targets)
                eval_preds.append(output)
                         
        if self.model.model_args.apply_lora:
            for name, param in self.lora_model.named_parameters():
                if "Lora" in name:
                    reg += (1/2)*(param ** 2).sum()  
        else:
            for name, param in self.lora_model.named_parameters():
                if "delta" in name:
                    reg += torch.norm(param, p = 'nuc')
          
        avg_loss = (total_loss / len(dataloader_outer.dataset) ) + (reg * self.args.linear_wd)
                
        eval_preds = torch.cat(eval_preds, dim=0)
        eval_targets = torch.cat(eval_targets, dim=0)
        eval_error = torch.tensor(avg_loss, dtype=self.gradient_dtype).to(self.gradient_dtype)
        
        return eval_preds , eval_targets, eval_error
   
    # Gather results from GPU
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
         
        with torch.no_grad():
           eval_preds , eval_targets, eval_error = self.evaluate_gpu(eval_dataset)

        if self.args.local_rank != -1:
            logger.info("Starting to gather kernel across GPUs")
            eval_preds = varsize_tensor_all_gather(eval_preds.to(self.args.device), torch.distributed.get_world_size())
            eval_targets = varsize_tensor_all_gather(eval_targets.to(self.args.device), torch.distributed.get_world_size())

            logger.info("Finished gathering kernel across GPUs")
            logger.info(f"pred size {eval_preds.size()}")
            logger.info(f"target size {eval_targets.size()}")
            logger.info(f"eval error is {eval_error}")
        
        eval_preds = eval_preds.cpu()
        eval_targets = eval_targets.cpu()

        if self.args.binary_classification: # Make sure to compute loss before this transformation!
            eval_preds = torch.cat([-eval_preds, eval_preds], dim=-1) # convert back to two logits
            eval_targets = ((eval_targets + 1) / 2).long() # convert back from {-1, 1} to {0, 1}

        if self.compute_metrics is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy()))

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        metrics["eval_loss"] = eval_error.item()

        metrics.update(self.metrics())
        metrics["grad_dim"] = self.grad_dim

        output = PredictionOutput(predictions=eval_preds.numpy(), label_ids=eval_targets.numpy(), metrics=metrics)
        self.log(output.metrics)

        return output
