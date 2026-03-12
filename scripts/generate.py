from pathlib import Path
import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

def generate_text(model, tokenizer, device, prompt, max_new_tokens=100, temperature=0.8, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Punjabi prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens
    )
    print(output)

if __name__ == "__main__":
    main()