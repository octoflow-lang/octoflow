# crypto — Cryptography & Encoding

Hash functions, encoding utilities, and random generation.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `hash` | 4 | Non-cryptographic hash functions |
| `encoding` | 4 | Base64 and hex encoding |
| `random` | 7 | Random numbers, UUIDs, tokens |

## hash

```
use crypto.hash
```

| Function | Description |
|----------|-------------|
| `djb2(s)` | DJB2 hash — fast, general-purpose |
| `fnv1a(s)` | FNV-1a hash — good distribution for hash tables |
| `checksum(arr)` | Simple numeric checksum (sum of elements) |
| `crc_simple(s)` | Simple CRC-like hash for data integrity |

```
let h = djb2("hello world")
print("Hash: {h}")
```

## encoding

```
use crypto.encoding
```

| Function | Description |
|----------|-------------|
| `to_base64(s)` | Encode string to base64 |
| `from_base64(s)` | Decode base64 to string |
| `to_hex(s)` | Encode string to hex |
| `from_hex(s)` | Decode hex to string |

```
let encoded = to_base64("Hello, OctoFlow!")
let decoded = from_base64(encoded)
```

## random

```
use crypto.random
```

| Function | Description |
|----------|-------------|
| `random_float(lo, hi)` | Random float in [lo, hi) |
| `random_int(lo, hi)` | Random integer in [lo, hi] |
| `random_hex(length)` | Random hex string of given length |
| `uuid_v4()` | Generate UUID v4 string |
| `random_token(length)` | Random alphanumeric token |
| `random_choice(arr)` | Randomly select element from array |
| `random_shuffle(arr)` | Shuffle array in-place |

```
let id = uuid_v4()
let token = random_token(32)
let roll = random_int(1, 6)
```

## Built-in Encoding

These are always available without `use`:

| Function | Description |
|----------|-------------|
| `base64_encode(s)` | Base64 encode |
| `base64_decode(s)` | Base64 decode |
| `hex_encode(s)` | Hex encode |
| `hex_decode(s)` | Hex decode |
