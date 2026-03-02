# parse_args (L3)

## Working Example
```flow
use sys/args

let args = parse_args()

let verbose = arg_flag(args, "--verbose")
print("verbose mode: {verbose}")

let outfile = arg_value(args, "--output")
print("output file: {outfile}")

let jobs = arg_value(args, "--jobs")
print("parallel jobs: {jobs}")

let has_dry = arg_flag(args, "--dry-run")
if has_dry == 1.0
  print("dry-run enabled, no writes")
end

let missing = arg_value(args, "--missing")
let found = len(missing)
if found == 0.0
  print("--missing flag was not provided")
end
```

## Expected Output
```
verbose mode: 1.0
output file: results.csv
parallel jobs: 4
dry-run enabled, no writes
--missing flag was not provided
```

*Run with:* `octoflow run script.flow --verbose --output results.csv --jobs 4 --dry-run`

## Common Mistakes
- DON'T: `arg_value("--output")` → DO: `arg_value(args, "--output")` (pass parsed args)
- DON'T: `if verbose == true` → DO: `if verbose == 1.0`
- DON'T: `args["--output"]` → DO: `arg_value(args, "--output")`

## Edge Cases
- arg_flag returns 0.0 if not present, 1.0 if present
- arg_value returns empty string if flag exists but has no value
- Unknown flags are silently ignored
