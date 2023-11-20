from padding_free_train.dataset import tokenize_conversation
from transformers import AutoTokenizer
from rich import print


def print_tokens_and_labels(tokenizer, tokens, labels):
    print("\n")
    for i, (input_id, label) in enumerate(zip(tokens, labels)):
        token = tokenizer.decode(input_id).replace("\n", "\\n")
        print(
            f"[{i:03d}] decode: [green]{token:14}[/green] "
            f"input_id: {input_id:12}, label: {label}"
        )


def test_qwen():
    conversation = [
        {"role": "system", "content": "You are a student."},
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "I am a student."},
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B-Chat", trust_remote_code=True
    )
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 151643
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokens, labels = tokenize_conversation(tokenizer, conversation)
    assert len(tokens) == len(labels)

    decoded_gt = """<|im_start|>system
You are a student.<|im_end|>
<|im_start|>user
Who are you?<|im_end|>
<|im_start|>assistant
I am a student.<|im_end|>"""
    assert tokenizer.decode(tokens) == decoded_gt

    print_tokens_and_labels(tokenizer, tokens, labels)
