# Padding Free LLM Training

This is an example project to test how to use **padding-free training** and **Multipack Sampler** from [openchat](https://github.com/imoneoi/openchat),
achieving a **3~10x speedup** compared to the conventional padded training. 

# Supported Models
- [x] Llama
- [x] Mistral
- [x] QWen
- [x] Yi

# Training

```bash
deepspeed -i localhost:0,1 train.py \
--jsonl ./data/random_selected_sharegpt_gpt4_en_zh.jsonl \
--model_name_or_path "Qwen/Qwen-7B" \
--padding_free true \
--sample_max_length 4096  \
--batch_max_length 32768 \
--deepspeed "./configs/ds_stage3.json" \
--logging_steps 10 \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--bf16 \
--output_dir padding_free_qwen_7b
```

- padding_free: Should we use padding-free + Multipack Sampler for training
- sample_max_length: Maximum truncation length for a single sample.
- batch_max_length: Maximum truncation length after concat multiple samples

Example data is random selected from [openchat/openchat_sharegpt4_dataset](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
