# timer (L2)
sys/timer — Timing and benchmarking utilities.

## Functions
stopwatch() → float
  Start stopwatch, return timestamp
elapsed(start: float) → float
  Milliseconds since start
elapsed_secs(start: float) → float
  Seconds since start
benchmark(label: string) → float
  Begin named benchmark
benchmark_end(label: string, start: float) → float
  End benchmark and print duration
sleep_busy(ms: float) → int
  Busy-wait sleep for ms milliseconds
format_duration(ms: float) → string
  Format ms as human-readable string
