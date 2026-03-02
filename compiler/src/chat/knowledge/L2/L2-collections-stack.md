# stack (L2)
collections/stack — LIFO stack backed by array.

## Functions
stack_create() → map
  Create empty stack
stack_push(s: map, val: any) → map
  Push value onto top
stack_pop(s: map) → any
  Pop value from top
stack_peek(s: map) → any
  View top without removing
stack_size(s: map) → int
  Number of elements
stack_is_empty(s: map) → int
  Check if stack is empty
stack_clear(s: map) → map
  Remove all elements
