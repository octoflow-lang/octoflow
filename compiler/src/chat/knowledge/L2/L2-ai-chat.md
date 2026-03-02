# chat (L2)
llm/chat — Chat template builder, auto-detects model format.

## Functions
resolve_chat_tokens(tok: map, model: map) → map
  Resolve special chat tokens from model
detect_chat_format(model: map) → string
  Detect chat format (Qwen/LLaMA/Gemma/Phi)
build_chat_tokens(tok: map, messages: array, fmt: string) → array
  Build token array from chat messages
get_stop_tokens(fmt: string) → array
  Get stop token IDs for format
is_stop_token(id: int, stops: array) → int
  Check if token is a stop token
