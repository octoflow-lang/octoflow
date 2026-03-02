# db — In-Memory Database

Relational-style tables with SQL-like queries, all in memory.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `core` | 9 | Create tables, insert, select, filter |
| `query` | 3 | Join, group by, order by |
| `schema` | 3 | Column introspection and renaming |

## core

```
use db.core
```

| Function | Description |
|----------|-------------|
| `db_create()` | Create new in-memory database |
| `db_table(db, name, columns)` | Create table with named columns |
| `db_tables(db)` | List all table names |
| `db_insert(table, row)` | Insert row (map of column:value) |
| `db_count(table)` | Row count |
| `db_select_column(table, col, n)` | Get all values in column |
| `db_select_row(table, idx)` | Get row by index |
| `db_where(table, col, op, val)` | Filter rows by condition |
| `db_import_csv(table, path)` | Import CSV data into table |

```
let db = db_create()
let users = db_table(db, "users", ["name", "age", "city"])

let mut row = map()
row["name"] = "Alice"
row["age"] = 30
row["city"] = "Portland"
db_insert(users, row)

let young = db_where(users, "age", "<", 25)
```

## query

```
use db.query
```

| Function | Description |
|----------|-------------|
| `db_join(t1, t2, col, n1, n2)` | Inner join on shared column |
| `db_group_by(table, col, agg_col, n, op)` | Group by with aggregation (sum, count, mean, min, max) |
| `db_order_by(table, col, n, ascending)` | Sort table by column |

```
let totals = db_group_by(orders, "category", "amount", 100, "sum")
let sorted = db_order_by(users, "age", 50, 1.0)
```

## schema

```
use db.schema
```

| Function | Description |
|----------|-------------|
| `db_columns(table)` | Get column names as array |
| `db_describe(table)` | Table metadata (columns, row_count, col_count) |
| `db_rename_column(table, old_name, new_name)` | Rename column in-place |
