# json (L2)
data/json — Pure .flow JSON parser/encoder (recursive descent, dot-notation nesting, array indexing)

## Functions
json_decode(s: string) → map
  Parse JSON string into map/array
json_get_array(s: string) → array
  Parse JSON string expecting array result
json_has_array(s: string) → float
  Test if JSON string contains top-level array (1.0/0.0)
json_encode(obj: map) → string
  Serialize map/array to JSON string
