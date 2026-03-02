# collections (L2)
collections/collections — Stack, queue, counter, set, default map, pair.

## Functions
stack_new() → map
  Create new stack
stack_push(s: map, val: any) → map
  Push value onto stack
stack_pop(s: map) → any
  Pop top value
stack_peek(s: map) → any
  View top without removing
queue_new() → map
  Create new queue
queue_enqueue(q: map, val: any) → map
  Add to back of queue
queue_dequeue(q: map) → any
  Remove from front
counter_new() → map
  Create frequency counter
counter_add(c: map, key: string) → map
  Increment key count
set_new() → map
  Create empty set
set_add(s: map, val: any) → map
  Add element to set
set_has(s: map, val: any) → int
  Check membership
set_union(a: map, b: map) → map
  Union of two sets
set_intersection(a: map, b: map) → map
  Intersection of two sets
set_difference(a: map, b: map) → map
  Difference of two sets
dmap_new(default: any) → map
  Create map with default value
pair(a: any, b: any) → map
  Create key-value pair
