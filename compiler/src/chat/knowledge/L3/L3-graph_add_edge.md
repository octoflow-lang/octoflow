# graph_add_edge (L3)

## Working Example
```flow
use collections/graph

let g = graph_create()

graph_add_node(g, "A")
graph_add_node(g, "B")
graph_add_node(g, "C")
graph_add_node(g, "D")

graph_add_edge(g, "A", "B")
graph_add_edge(g, "A", "C")
graph_add_edge(g, "B", "D")
graph_add_edge(g, "C", "D")

let has = graph_has_edge(g, "A", "B")
print("A->B exists: {has}")

let missing = graph_has_edge(g, "A", "D")
print("A->D exists: {missing}")

let visited = graph_bfs(g, "A")
let count = len(visited)
print("BFS from A visited {count} nodes")

for i in range(0, count)
    let node = visited[i]
    print("  visited: {node}")
end
```

## Expected Output
```
A->B exists: 1.0
A->D exists: 0.0
BFS from A visited 4 nodes
  visited: A
  visited: B
  visited: C
  visited: D
```

## Common Mistakes
- DON'T: `graph_add_edge(g, "X", "Y")` with unknown nodes → DO: call `graph_add_node` first
- DON'T: `g.add_edge("A", "B")` → DO: `graph_add_edge(g, "A", "B")`
- DON'T: `if has == true` → DO: `if has == 1.0` (no booleans)

## Edge Cases
- Adding a duplicate edge is a no-op
- BFS on a disconnected node returns a single-element list
- graph_has_edge returns 0.0 for nonexistent nodes without error
