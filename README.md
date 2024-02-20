# LoRA Training in the NTK regime has No Spurious Local Minima

This is the code for the paper [LoRA Training in the NTK regime has No Spurious Local Minima](https://arxiv.org/abs/2402.11867). We simply added `linearized.py` to the [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT/tree/main) repository, which was originally used to compute gradient kernels. For more information, please visit this repository.

## Installation
All the requirements and the installation process are identical to those of the [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT/tree/main) repository. Note that our implementation requires functorch, which is available in previous versions of PyTorch. Please check the `requirements.txt`. The main packages are:

```
torch==1.12.1
transformers==4.4.2
functorch==0.2.1
```

## Prepare the data
Just like in the [LM-Kernel-FT](https://github.com/princeton-nlp/LM-Kernel-FT/tree/main) repository, run the following commands to download and prepare the data:

```bash
( cd data; bash download_dataset.sh )

for K in 16; do
    # Generate k-shot splits for seeds 13,21,42,87,100 with a maximum of 1k test examples in data/k-shot-1k-test,
    # where k is the number of training/validation examples per label
    python tools/generate_k_shot_data.py --mode k-shot-1k-test --k $K
done
```

## Run the code
To run experiments, use `run_fewshot.sh`:

```bash
NUM_GPU=1 TAG=$Your_tag TASK=QNLI SEED=13 K=16 MODEL=roberta-base bash run_fewshot.sh
```

You may use additional arguments in your training. For example,

```bash
NUM_GPU=4 TAG=$Your_tag TASK=QNLI SEED=13 K=16 MODEL=roberta-base bash run_fewshot.sh --per_device_train_batch_size 32 --per_device_eval_batch_size 32   --linear_num_epoch 1000 --do_eval False  --do_predict True --linear_lr 0.001  --linear_wd 0.005  --lora_r 8   --apply_lora True  --eval_during_training True --train_last_layer True
```
To perform full fine-tuning, set `--apply_lora=False`. 

We have also added the following additional arguments for training linearized model: 

```
optional arguments:
  --linear_freeze_A             (bool) Whether or not to fix matrix A during training
  --linear_lr                   (float) Learning rate
  --linear_num_epoch            (int) Number of epochs
  --linear_wd                   (float) Weight decay parameter $\lambda$
  --eval_during_training        (bool) Whether or not to evaluate on 1000 test data during every epoch. This requires approximately 50GB of CPU memory.
  --train_last_layer            (bool) Whether or not to train only the last transformer layer. 

```
