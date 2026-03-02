# pipeline (L2)
data/pipeline — Data transform pipelines (chain named transforms, batch process, parallel merge)

## Functions
chain_apply(data: array, fns: array) → array
  Apply sequence of named transforms to data
batch_process(data: array, fn: string, n: int) → array
  Process data in batches of size n
parallel_merge(arr: array, arr: array) → array
  Merge results from parallel transform branches
