# datetime (L2)
time/datetime — Date/time utilities.

## Functions
unix_now() → float
  Current Unix timestamp in seconds
ms_now() → float
  Current time in milliseconds
unix_to_date(ts: float) → map
  Convert timestamp to {year, month, day, hour, min, sec}
is_leap_year(year: int) → int
  Check if year is a leap year
format_date(ts: float) → string
  Format timestamp as YYYY-MM-DD
format_datetime_full(ts: float) → string
  Format as YYYY-MM-DD HH:MM:SS
day_of_week(ts: float) → int
  Day of week (0=Sunday)
day_name(ts: float) → string
  Day name (Monday, Tuesday, ...)
days_between(a: float, b: float) → int
  Number of days between two timestamps
hours_between(a: float, b: float) → float
  Hours between two timestamps
