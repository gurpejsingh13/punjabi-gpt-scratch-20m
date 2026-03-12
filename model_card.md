# Model Card: Punjabi-GPT-Scratch-20M

## Overview
Punjabi-GPT-Scratch-20M is a decoder-only causal language model trained from scratch on Punjabi (Gurmukhi) text.

## Model Details
- Model type: GPT-style decoder-only LM
- Language: Punjabi (Gurmukhi)
- Parameters: 20,389,632
- Vocabulary size: 16,000
- Context length: 128

## Training Objective
Causal Language Modeling (next-token prediction)

## Dataset
The model was trained on a cleaned Punjabi text corpus derived from AI4Bharat Sangraha Punjabi/Gurmukhi data.

## Training Summary
- Epochs: 2
- Final validation loss: 1.416886
- Approx perplexity: 7.02
- Device: Apple Silicon MPS

## Strengths
- Generates coherent Punjabi script
- Low repetition
- Strong Gurmukhi character ratio

## Limitations
- Short context length
- Can drift semantically
- Not instruction-tuned
- Not suitable for factual or high-stakes use

## Intended Use
Research, experimentation, educational use, and further continued pretraining or instruction tuning.

## Out of Scope
Medical, legal, financial, or factual decision support.