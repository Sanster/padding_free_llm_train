import torch

from padding_free_train.dataset import tokenize_conversation
from padding_free_train.modeling.unpadded_qwen2 import PaddedQWenForCausalLM

if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_path = "/mnt/algo/code/llm_padding_free_train/tmp"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer)
    im_end_token_id = tokenizer.encode("<|im_end|>")
    model = PaddedQWenForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "你是谁"},
    ]
    tokens, _ = tokenize_conversation(tokenizer, messages)
    print(f"prompt length: {len(tokens)}")
    tokens += tokenizer.encode("<|im_start|>assistant\n")
    tokens = torch.LongTensor(tokens).cuda()
    outputs = model.generate(
        tokens.unsqueeze(0),
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))
