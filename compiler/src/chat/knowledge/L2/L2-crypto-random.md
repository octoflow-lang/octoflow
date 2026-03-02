# random (L2)
crypto/random — Random generation (float, int, hex, UUID v4, token, choice, shuffle)

## Functions
random_float() → float
  Random float in [0, 1)
random_int(lo: int, hi: int) → int
  Random integer in [lo, hi]
random_hex(n: int) → string
  Random hex string of n bytes
uuid_v4() → string
  Generate UUID v4 string
random_token(n: int) → string
  Random alphanumeric token of length n
random_choice(arr: array) → string
  Pick random element from array
random_shuffle(arr: array) → array
  Shuffle array in place
