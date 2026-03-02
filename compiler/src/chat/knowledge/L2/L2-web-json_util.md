# json_util (L2)
web/json_util — JSON utilities (pretty-print, merge, safe access with default, flatten to dot-notation)

## Functions
json_pretty(obj: map) → string
  Pretty-print map/array as indented JSON
json_merge(a: map, b: map) → map
  Deep-merge two maps (b overrides a)
json_get(obj: map, s: string, default: string) → string
  Safe dot-notation access with default value
json_flatten(obj: map) → map
  Flatten nested map to dot-notation keys
