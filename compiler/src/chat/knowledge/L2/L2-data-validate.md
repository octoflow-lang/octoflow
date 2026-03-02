# validate (L2)
data/validate — Data validation (is_numeric, is_email, is_url, in_range, clamp_value)

## Functions
is_numeric(s: string) → float
  Test if string is a valid number (1.0/0.0)
is_email(s: string) → float
  Test if string is a valid email (1.0/0.0)
is_url(s: string) → float
  Test if string is a valid URL (1.0/0.0)
in_range(x: float, lo: float, hi: float) → float
  Test if value is within [lo, hi] (1.0/0.0)
clamp_value(x: float, lo: float, hi: float) → float
  Clamp value to [lo, hi] range
