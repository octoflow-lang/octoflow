# heap (L2)
collections/heap — Binary min-heap.

## Functions
heap_create() → map
  Create empty min-heap
heap_push(h: map, val: float) → map
  Insert value into heap
heap_pop(h: map) → float
  Remove and return minimum
heap_peek(h: map) → float
  View minimum without removing
heap_size(h: map) → int
  Number of elements
heap_is_empty(h: map) → int
  Check if heap is empty
heapify(arr: array) → map
  Build heap from array
