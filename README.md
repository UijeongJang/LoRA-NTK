# LoRA Training in the NTK regime has No Spurious Local Minima

This is the implementation for the paper [LoRA Training in the NTK regime has No Spurious Local Minima]

## Installation
Please check the requirements.txt. We have the same requirements from Malladi et al. (2023). 

```
pip install -r requirements.txt
```

## Prepare the data
Run the following commands to download and prepare the data:

```bash
( cd data; bash download_dataset.sh )

for K in 16; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```
For more information, please refer to (https://github.com/princeton-nlp/LM-Kernel-FT/tree/main) of Malladi et al. (2023). 

**NOTE**: During training, the linearized model will create `gradients.pth` file to save pre-computed gradients $\mathbf{G}(X_i)$ for each data. 

## Run the code
To run experiments, you can use `run_fewshot.sh`:

```bash
NUM_GPU=4 TAG=$Your_tag TASK=$task SEED=$seed K=16 MODEL=roberta-base bash run_fewshot.sh
```

The results will be saved in a folder `tensorboard_results/{bool: apply_lora}-{learning_rate}-{task}-{seed}-{lora_rank}`. 

You may use the following command to check the results.

```bash
 tensorboard --logdir tensorboard_result
```
You can also add extra arguments to modify the hyperparameters:

```bash
NUM_GPU=4 TAG=$lr TASK=$task SEED=$seed K=16 MODEL=roberta-base bash run_fewshot.sh --per_device_train_batch_size 32 --per_device_eval_batch_size 32   --linear_num_epoch 1000 --do_eval True  --do_predict True --linear_lr $lr  --linear_wd $wd  --lora_r $T   --apply_lora $lora  --eval_during_training True --train_last_layer True
```
