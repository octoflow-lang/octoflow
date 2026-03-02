# tokenizer (L2)
ai/tokenizer — BPE tokenizer for Qwen/LLaMA/Mistral.

## Functions
tokenizer_load(path: string) → map
  Load tokenizer from HuggingFace JSON
tokenizer_build_simple(vocab: map) → map
  Build simple tokenizer from vocab map
tokenizer_encode(tok: map, text: string) → array
  Encode text to token ID array
tokenizer_decode(tok: map, ids: array) → string
  Decode token IDs to text
tokenizer_load_from_gguf(model: map) → map
  Extract tokenizer from loaded GGUF model
