---
library_name: peft
tags:
- trl
- sft
- generated_from_trainer
base_model: google/gemma-7b-it
model-index:
- name: gemma7bcars128
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# gemma7bcars128

This model is a fine-tuned version of [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it) on the None dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 100
- num_epochs: 10

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.41.1
- Pytorch 2.3.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1