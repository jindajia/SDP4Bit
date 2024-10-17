# SDP4Bit
This repository is the official implement of paper **SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training**.

## Overview
SDP4Bit is a communication quantization strategy designed to reduce the overhead of large-scale distributed training in Sharded Data Parallelism (ShardedDP). By utilizing quantization on weight differences and two-level gradient smooth quantization, SDP4Bit reduces the communication of weights and gradients to nearly 4 bits without compromising accuracy. 

## Paper Results Reproduce
### Preparing for Data
In the data processing step, we followed the [data preprocessing instructions](https://github.com/NVIDIA/Megatron-LM?tab=readme-ov-file#data-preprocessing) in Megatron-LM official repository. We use the [**pile deduplicated dataset**](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) provided by huggingface as our training baseline. For the vocabulary and merges file, we used same as gpt2 model. 
**Download**
```
from datasets import load_dataset
train_data = load_dataset('EleutherAI/the_pile_deduplicated', split='train', num_proc=16)
train_data.to_json(os.path.join(save_path, dataset_output_name), lines=True)
hf_hub_download(repo_id="gpt2", filename="merges.txt", local_dir=save_path)
hf_hub_download(repo_id="gpt2", filename="vocab.json", local_dir=save_path)
```
**Data Process**
We used [preprocess script](https://github.com/NVIDIA/Megatron-LM/blob/main/tools/preprocess_data.py) in Megatron-LM repository and the dataset download in last step. 
```
python preprocess_data.py \
		--input  pile.jsonl \
		--split  train \
		--columns  text \
		--output-prefix  pile \
		--vocab-file  vocab.json \
		--merge-file  merges.txt \
		--dataset-impl  mmap \
		--tokenizer-type  GPT2BPETokenizer \
		--append-eod \
		--torch-backend  mpi
```
### Accucracy Test Results Reproduce
![enter image description here](https://github.com/jindajia/SDP4Bit/raw/main/Figures/accuracy_test_table.png)
We set all models to run for a total of 80,000 training iterations. The learning rate was configured according to GPT-2 settings. 
Note: For each experimental group, we used the same training configuration for the same model, with only the quantization configuration being changed to ensure a fair comparison. The model configuration and detailed sample training scripts are provided below.
**Model Card**
```
125M
MODEL_ARGS="
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
"

OPTIMIZER_ARGS="
    --lr 0.0006 \
    --lr-decay-iters 70000 \
    --lr-decay-style cosine \
    --min-lr 0.00006 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-08 \
    --weight-decay .1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --loss-scale 0 \
    --loss-scale-window 1000 \
    --hysteresis 2 \
    --min-loss-scale 1 \
    --bf16 \
    --use-distributed-optimizer \
"

TRAINING_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-iters 80000 \
"
```

```
GPT 350M Model
MODEL_ARGS="
	--num-layers 24 \
	--hidden-size 1024 \
	--num-attention-heads 16 \
	--seq-length 2048 \
	--max-position-embeddings 2048 \
"

TRAINING_ARGS="
	--tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--micro-batch-size 8 \
	--global-batch-size 256 \
	--train-iters 80000 \
"

OPTIMIZER_ARGS="
	--lr 0.0003 \
	--lr-decay-iters 70000 \
	--lr-decay-style cosine \
	--min-lr 0.00003 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--adam-eps 1e-08 \
	--weight-decay .1 \
	--lr-warmup-fraction 0.01 \
	--clip-grad 1.0 \
	--loss-scale 0 \
	--loss-scale-window 1000 \
	--hysteresis 2 \
	--min-loss-scale 1 \
	--bf16 \
	--use-distributed-optimizer \
"
```

```
GPT 1.3B Model
MODEL_ARGS="
	--num-layers 24 \
	--hidden-size 2048 \
	--num-attention-heads 16 \
	--seq-length 2048 \
	--max-position-embeddings 2048 \
"

TRAINING_ARGS="
	--tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--micro-batch-size 2 \
	--global-batch-size 256 \
	--train-iters 80000 \
"

OPTIMIZER_ARGS="
	--lr 0.0002 \
	--lr-decay-iters 70000 \
	--lr-decay-style cosine \
	--min-lr 0.00002 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--adam-eps 1e-08 \
	--weight-decay .1 \
	--lr-warmup-fraction 0.01 \
	--clip-grad 1.0 \
	--loss-scale 0 \
	--loss-scale-window 1000 \
	--hysteresis 2 \
	--min-loss-scale 1 \
	--bf16 \
	--use-distributed-optimizer \
"
```

```
GPT 6.7B Model
MODEL_ARGS="
	--num-layers 32 \
	--hidden-size 4096 \
	--num-attention-heads 32 \
	--seq-length 2048 \
	--max-position-embeddings 2048 \
"

OPTIMIZER_ARGS="
	--lr 0.00012 \
	--lr-decay-iters 70000 \
	--lr-decay-style cosine \
	--min-lr 0.000012 \
	--adam-beta1 0.9 \
	--adam-beta2 0.95 \
	--adam-eps 1e-08 \
	--weight-decay .1 \
	--lr-warmup-fraction 0.01 \
	--clip-grad 1.0 \
	--loss-scale 0 \
	--loss-scale-window 1000 \
	--hysteresis 2 \
	--min-loss-scale 1 \
	--bf16 \
	--use-distributed-optimizer \
"

TRAINING_ARGS="
	--tensor-model-parallel-size 1 \
	--pipeline-model-parallel-size 1 \
	--micro-batch-size 2 \
	--global-batch-size 256 \
	--train-iters 80000 \
"
```

**Sample Training Scripts**
| Model |Baseline|qWD|TLq|TLq-HS|SDP4Bit|
|--|--|--|--|--|--|--|
| 125M |[link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/accuracy/125M/baseline/train.sh)  |[link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/accuracy/125M/quantWeightDiff/train.sh)  |[link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/accuracy/125M/quantGradwithoutHT/train.sh)  |[link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/accuracy/125M/quantGrad/train.sh)  |[link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/accuracy/125M/quantWeightDiff_Grad/train.sh)|

### Speed Test Results Reproduce
![enter image description here](https://github.com/jindajia/SDP4Bit/raw/main/Figures/speed_test_table.png)
We provide the detailed speed test scripts on H800 as below. Please note that since H800 node contains 8 GPUs, and A100 node contains 4GPU, we adjust the tensor parallel size and pipeline parallel size accordingly. This adjustment ensures that tensor parallel size won't exceed the number of GPUs in a node. The parallel configuration and training scripts are provide below. 

| Model Size | TP  | PP  | Accumulation Step |
|------------|-----|-----|-------------------|
| 1.3B       | 1   | 1   | 1                 |
| 2.7B       | 1   | 1   | 1                 |
| 6.7B       | 4   | 1   | 1                 |
| 13B        | 4(A100)/8(H800) | 2(A100)/1(H800) | 1                 |
| 18B        | 4(A100)/8(H800) | 2(A100)/1(H800) | 1                 |
**Traing Scripts**
| Model Size | Baseline      | SDP4Bit      |
|------------|----------------------|----------------------|
| 1.3B       |    [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/1_3B_Baseline.sh)       |     [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/1_3B_QWG.sh)     |
| 2.7B       | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/2_7B_Baseline.sh)          | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/2_7B_QWG.sh)		 |
| 6.7B       | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/6_7B_Baseline.sh)          | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/6_7B_QWG.sh)		 |
| 13B        | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/13B_Baseline.sh)          | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/13B_QWG.sh)		 |
| 18B        | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/18B_Baseline.sh)          | [link](https://github.com/jindajia/Megatron-LM/blob/jinda/final_speed_test/sample_scripts/speed/Exp1/18B_QWG.sh)		 |

## Arguments Usage
Belows are all quantization arguments you may use on your case.

### Weight Quantization Arguments
- `--quantized-weights`
    - Weight Communication will be quantized when this is enable
    - Default: not enabled
- `--weight-quantization-bits 4`
    - Specifies the number of bits used for quantizing weights.
    - Default: 4
- `--wq-group-size 2048`
    - Defines the group size for weight quantization.
    - Default: 2048

### Gradient Quantization Arguments
- `--quantized-gradients`
    - Gradient Communication will be quantized when this is enable
    - Default: not enabled
- `--gq-group-size-inter 128`
    - Defines the group size for gradient quantization between nodes (inter-node).
    - Default: 128
- `--gradient-quantization-bits-inter 4`
    - Specifies the number of bits used for inter-node gradient quantization.
    - Default: 4
- `--gq-group-size-intra 128`
    - Defines the group size for gradient quantization within nodes (intra-node).
    - Default: 512
- `--gradient-quantization-bits-intra 8`
    - Specifies the number of bits used for intra-node gradient quantization.
    - Default: 8
- `--hadamard-transform`
    - Enable this to reduce Gradient Quantization error.
    - Default: not enabled
- `--gradient-alltoall-pipeline 8`
    - Chunk gradients to overlap intra and inter node communication.
    - Default: 1
### Additional Settings
- `--no-async-tensor-model-parallel-allreduce`
    - To overlap intra and inter node all-to-all, this should be enabled to avoid setting 
---
**Note:** The implementation of SDP4Bit is built upon the official [Nvidia Megatron-LM](https://github.com/NVIDIA/Megatron-LM) codebase, leveraging its optimized framework for large-scale language model training.

