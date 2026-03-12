from pathlib import Path
import argparse
import json
import math
import statistics

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel


def is_gurmukhi_char(ch: str) -> bool:
    return "\u0A00" <= ch <= "\u0A7F"


def gurmukhi_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(1 for ch in chars if is_gurmukhi_char(ch)) / len(chars)


def repetition_rate_words(text: str, n: int = 3) -> float:
    words = text.split()
    if len(words) < n:
        return 0.0

    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0

    unique = len(set(ngrams))
    total = len(ngrams)
    return 1 - (unique / total)


def generate_text(
    model,
    tokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_eval_text(eval_jsonl_path: Path, sample_size: int = 500) -> str:
    dataset = load_dataset("json", data_files={"eval": str(eval_jsonl_path)})["eval"]
    eval_subset = dataset.select(range(min(sample_size, len(dataset))))
    return "\n\n".join([row["text"] for row in eval_subset if row["text"].strip()])


def sliding_window_perplexity(
    model,
    tokenizer,
    device: str,
    eval_text: str,
    stride: int = 64,
) -> tuple[float, float]:
    max_length = model.config.n_positions
    encodings = tokenizer(eval_text, return_tensors="pt", truncation=False)
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    eval_loss = torch.stack(nlls).mean().item()
    perplexity = torch.exp(torch.tensor(eval_loss)).item()
    return eval_loss, perplexity


def main():
    parser = argparse.ArgumentParser(description="Benchmark Punjabi GPT scratch model")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="data/punjabi_corpus_cleaned.jsonl",
        help="Path to cleaned evaluation JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save benchmark artifacts",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of eval rows to use for perplexity text",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Tokens to generate per prompt",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    eval_jsonl = Path(args.eval_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = GPT2LMHeadModel.from_pretrained(str(model_dir)).to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Parameter count: {model.num_parameters():,}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Context length: {model.config.n_positions}")

    benchmark_prompts = [
        "ਪੰਜਾਬੀ ਭਾਸ਼ਾ",
        "ਇੱਕ ਪਿੰਡ ਵਿੱਚ",
        "ਅੱਜ ਦੇ ਸਮੇਂ ਵਿੱਚ ਸਿੱਖਿਆ",
        "ਕਿਸਾਨ ਲਈ ਸਭ ਤੋਂ ਮਹੱਤਵਪੂਰਨ ਗੱਲ",
        "ਪੰਜਾਬ ਦੇ ਇਤਿਹਾਸ ਬਾਰੇ",
        "ਪਰਿਵਾਰ ਦੀ ਮਹੱਤਤਾ",
        "ਸਿਹਤਮੰਦ ਜੀਵਨ ਸ਼ੈਲੀ",
        "ਇੱਕ ਛੋਟੀ ਕਹਾਣੀ",
        "ਤਕਨਾਲੋਜੀ ਦਾ ਭਵਿੱਖ",
        "ਬੱਚਿਆਂ ਦੀ ਸਿੱਖਿਆ",
    ]

    generation_results = []
    print("\nRunning prompt benchmark...")
    for prompt in benchmark_prompts:
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )
        generation_results.append(
            {
                "prompt": prompt,
                "output": output,
                "gurmukhi_ratio": gurmukhi_ratio(output),
                "repetition_rate_3gram_words": repetition_rate_words(output, n=3),
                "char_length": len(output),
                "word_length": len(output.split()),
            }
        )
        print(f"Done: {prompt}")

    avg_gurmukhi_ratio = statistics.mean(x["gurmukhi_ratio"] for x in generation_results)
    avg_repetition = statistics.mean(x["repetition_rate_3gram_words"] for x in generation_results)
    avg_char_length = statistics.mean(x["char_length"] for x in generation_results)
    avg_word_length = statistics.mean(x["word_length"] for x in generation_results)

    print("\nBuilding eval text...")
    eval_text = build_eval_text(eval_jsonl, sample_size=args.sample_size)
    print(f"Eval text characters: {len(eval_text):,}")

    print("Computing perplexity...")
    eval_loss, perplexity = sliding_window_perplexity(
        model=model,
        tokenizer=tokenizer,
        device=device,
        eval_text=eval_text,
        stride=64,
    )

    benchmark_report = {
        "model_name": model_dir.name,
        "model_type": "decoder-only causal language model",
        "language": "Punjabi (Gurmukhi)",
        "parameter_count": int(model.num_parameters()),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "context_length": int(model.config.n_positions),
        "device": device,
        "intrinsic_metrics": {
            "eval_loss_estimate": eval_loss,
            "perplexity": perplexity,
        },
        "generation_metrics": {
            "avg_gurmukhi_ratio": avg_gurmukhi_ratio,
            "avg_repetition_rate_3gram_words": avg_repetition,
            "avg_generated_char_length": avg_char_length,
            "avg_generated_word_length": avg_word_length,
        },
        "prompt_results": generation_results,
    }

    benchmark_json = output_dir / "benchmark_report.json"
    benchmark_txt = output_dir / "benchmark_samples.txt"

    with open(benchmark_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)

    with open(benchmark_txt, "w", encoding="utf-8") as f:
        for item in generation_results:
            f.write("=" * 100 + "\n")
            f.write(f"PROMPT: {item['prompt']}\n\n")
            f.write(f"OUTPUT:\n{item['output']}\n\n")
            f.write(f"Gurmukhi ratio: {item['gurmukhi_ratio']:.4f}\n")
            f.write(f"Repetition rate (3-gram words): {item['repetition_rate_3gram_words']:.4f}\n")
            f.write(f"Char length: {item['char_length']}\n")
            f.write(f"Word length: {item['word_length']}\n")

    print("\nBenchmark complete.")
    print(f"Eval loss: {eval_loss:.6f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Average Gurmukhi ratio: {avg_gurmukhi_ratio:.4f}")
    print(f"Average repetition rate: {avg_repetition:.4f}")
    print(f"Saved JSON report to: {benchmark_json}")
    print(f"Saved text samples to: {benchmark_txt}")


if __name__ == "__main__":
    main()