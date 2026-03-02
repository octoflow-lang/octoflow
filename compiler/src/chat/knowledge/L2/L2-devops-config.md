# config (L2)
devops/config — Configuration file parsing (INI, .env).

## Functions
parse_ini(text: string) → map
  Parse INI file into nested map of sections
parse_env_file(text: string) → map
  Parse .env KEY=VALUE pairs into map
config_get(cfg: map, key: string) → string
  Get config value by key
