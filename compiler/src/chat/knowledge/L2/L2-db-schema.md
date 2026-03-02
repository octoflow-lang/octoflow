# schema (L2)
db/schema — Database schema inspection and modification.

## Functions
db_columns(db: map, table: string) → array
  List column names
db_describe(db: map, table: string) → map
  Describe table structure (columns, row count)
db_rename_column(db: map, table: string, old: string, new: string) → map
  Rename a column
