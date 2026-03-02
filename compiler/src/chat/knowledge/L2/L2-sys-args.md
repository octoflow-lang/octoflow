# args (L2)
sys/args — CLI argument parsing from FLOW_ARGS.

## Functions
parse_args() → map
  Parse CLI args into flags, values, positional
arg_flag(args: map, name: string) → int
  Check if boolean flag is set
arg_value(args: map, name: string) → string
  Get named argument value or empty
arg_required(args: map, name: string) → string
  Get named argument, error if missing
