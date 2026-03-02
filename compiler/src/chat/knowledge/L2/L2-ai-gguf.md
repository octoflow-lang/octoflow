# gguf (L2)
llm/gguf — GGUF v3 parser for model files.

## Functions
gguf_load(data: array) → map
  Load GGUF model from byte array
gguf_load_from_file(path: string) → map
  Load GGUF model from file path
gguf_meta(model: map, key: string) → any
  Get metadata value by key
gguf_has_tensor(model: map, name: string) → int
  Check if tensor exists
gguf_tensor_type(model: map, name: string) → int
  Get tensor quantization type
gguf_tensor_offset(model: map, name: string) → int
  Get tensor data offset in bytes
gguf_tensor_count(model: map) → int
  Total number of tensors
