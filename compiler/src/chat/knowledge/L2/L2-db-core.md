# core (L2)
db/core — In-memory columnar database with CSV import.

## Functions
db_create() → map
  Create empty database
db_table(db: map, name: string, cols: array) → map
  Create table with column names
db_insert(db: map, table: string, row: map) → map
  Insert row into table
db_count(db: map, table: string) → int
  Count rows in table
db_select_column(db: map, table: string, col: string) → array
  Get all values in column
db_select_row(db: map, table: string, idx: int) → map
  Get row by index
db_where(db: map, table: string, col: string, val: any) → array
  Filter rows where col == val
db_import_csv(db: map, table: string, path: string) → map
  Import CSV file into table
