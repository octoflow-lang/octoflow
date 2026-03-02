# collections — Data Structures

Stacks, queues, heaps, graphs, sets, counters, and other
collection types built on top of OctoFlow's maps and arrays.

## Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `collections` | 25 | Stack, queue, counter, set, pair, default map |
| `stack` | 7 | Dedicated stack implementation |
| `queue` | 7 | Dedicated queue implementation |
| `heap` | 7 | Min-heap / priority queue |
| `graph` | 8 | Graph with adjacency list, BFS |

## collections

```
use collections.collections
```

### Stack (LIFO)

| Function | Description |
|----------|-------------|
| `stack_new()` | Create new stack |
| `stack_push(s, val)` | Push value |
| `stack_pop(s)` | Pop and return top value |
| `stack_peek(s)` | View top without removing |
| `stack_is_empty(s)` | 1.0 if empty |

### Queue (FIFO)

| Function | Description |
|----------|-------------|
| `queue_new()` | Create new queue |
| `queue_enqueue(q, val)` | Add to back |
| `queue_dequeue(q)` | Remove from front |
| `queue_size(q)` | Queue size |
| `queue_is_empty(q)` | 1.0 if empty |

### Counter

| Function | Description |
|----------|-------------|
| `counter_new()` | Create new counter |
| `counter_add(c, key)` | Increment key count |
| `counter_get(c, key)` | Get count for key |
| `counter_from_array(arr)` | Build counter from array |

### Set

| Function | Description |
|----------|-------------|
| `set_new()` | Create new set |
| `set_add(s, val)` | Add value |
| `set_has(s, val)` | 1.0 if value present |
| `set_remove(s, val)` | Remove value |
| `set_size(s)` | Set size |

### DefaultMap and Pair

| Function | Description |
|----------|-------------|
| `dmap_new(default_val)` | Create map with default value |
| `dmap_get(m, key)` | Get value (returns default if missing) |
| `dmap_set(m, key, val)` | Set value |
| `pair(a, b)` | Create pair/tuple |
| `pair_first(p)` | First element |
| `pair_second(p)` | Second element |

```
let mut s = set_new()
set_add(s, "apple")
set_add(s, "banana")
print(set_has(s, "apple"))    // 1

let c = counter_from_array(["a", "b", "a", "a", "b"])
print(counter_get(c, "a"))    // 3
```

## heap

```
use collections.heap
```

| Function | Description |
|----------|-------------|
| `heap_create()` | Create empty min-heap |
| `heap_push(heap, val)` | Insert value |
| `heap_pop(heap)` | Remove and return minimum |
| `heap_peek(heap)` | View minimum without removing |
| `heap_size(heap)` | Number of elements |
| `heap_is_empty(heap)` | 1.0 if empty |
| `heapify(arr)` | Build min-heap from array in-place |

```
let mut h = heap_create()
heap_push(h, 5)
heap_push(h, 2)
heap_push(h, 8)
let smallest = heap_pop(h)    // 2
```

## graph

```
use collections.graph
```

| Function | Description |
|----------|-------------|
| `graph_create()` | Create empty directed graph |
| `graph_add_node(g, node)` | Add node |
| `graph_add_edge(g, from, to, weight)` | Add directed weighted edge |
| `graph_neighbors(g, node)` | Get neighbor names as array |
| `graph_weight(g, from, to)` | Get edge weight |
| `graph_nodes(g)` | All node names |
| `graph_bfs(g, start)` | Breadth-first search traversal order |
| `graph_has_edge(g, from, to)` | 1.0 if edge exists |

```
let mut g = graph_create()
graph_add_node(g, "A")
graph_add_node(g, "B")
graph_add_node(g, "C")
graph_add_edge(g, "A", "B", 1.0)
graph_add_edge(g, "B", "C", 2.0)

let order = graph_bfs(g, "A")    // ["A", "B", "C"]
```
