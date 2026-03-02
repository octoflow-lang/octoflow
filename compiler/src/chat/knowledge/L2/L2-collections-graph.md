# graph (L2)
collections/graph — Adjacency list graph with weighted edges and BFS.

## Functions
graph_create() → map
  Create empty graph
graph_add_node(g: map, id: string) → map
  Add node to graph
graph_add_edge(g: map, from: string, to: string, w: float) → map
  Add weighted edge
graph_neighbors(g: map, id: string) → array
  Get adjacent nodes
graph_weight(g: map, from: string, to: string) → float
  Get edge weight
graph_nodes(g: map) → array
  List all node IDs
graph_bfs(g: map, start: string) → array
  Breadth-first traversal
graph_has_edge(g: map, from: string, to: string) → int
  Check if edge exists
