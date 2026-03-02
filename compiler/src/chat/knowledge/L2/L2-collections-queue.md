# queue (L2)
collections/queue — FIFO queue with O(1) dequeue via map-based indexing.

## Functions
queue_create() → map
  Create empty queue
queue_enqueue(q: map, val: any) → map
  Add element to back
queue_dequeue(q: map) → any
  Remove element from front
queue_peek(q: map) → any
  View front element
queue_size(q: map) → int
  Number of elements
queue_is_empty(q: map) → int
  Check if queue is empty
queue_clear(q: map) → map
  Remove all elements
