from pathlib import Path
import argparse
from transformers import AutoTokenizer, GPT2LMHeadModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)

    print("Model directory:", args.model_dir)
    print("Parameter count:", model.num_parameters())
    print("Vocab size:", len(tokenizer))
    print("Context length:", model.config.n_positions)
    print("Layers:", model.config.n_layer)
    print("Heads:", model.config.n_head)
    print("Embedding dim:", model.config.n_embd)

if __name__ == "__main__":
    main()