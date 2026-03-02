# query (L2)
db/query — Database query operations.

## Functions
db_join(db: map, t1: string, t2: string, on: string) → array
  Inner join two tables on column
db_group_by(db: map, table: string, col: string, agg: string) → map
  Group by column with aggregation
db_order_by(db: map, table: string, col: string) → array
  Sort rows by column ascending
