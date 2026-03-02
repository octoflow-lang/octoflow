# struct (L3)

## Working Example
```flow
struct Point(x, y)

struct Rect(origin, width, height)

fn distance(a, b)
  let dx = b.x - a.x
  let dy = b.y - a.y
  return sqrt(dx * dx + dy * dy)
end

fn area(r)
  return r.width * r.height
end

fn describe_point(p)
  let px = p.x
  let py = p.y
  print("Point({px}, {py})")
end

let p1 = Point(0.0, 0.0)
let p2 = Point(3.0, 4.0)

describe_point(p1)
describe_point(p2)

let dist = distance(p1, p2)
print("distance: {dist}")

let r = Rect(p1, 10.0, 5.0)
let a = area(r)
print("area: {a}")

let ox = r.origin.x
let oy = r.origin.y
print("origin: ({ox}, {oy})")
```

## Expected Output
```
Point(0.0, 0.0)
Point(3.0, 4.0)
distance: 5.0
area: 50.0
origin: (0.0, 0.0)
```

## Common Mistakes
- DON'T: `struct Point { x, y }` → DO: `struct Point(x, y)` (parentheses, no braces)
- DON'T: `Point.new(1.0, 2.0)` → DO: `Point(1.0, 2.0)`
- DON'T: `print("point: {p}")` for struct → DO: access fields individually

## Edge Cases
- Structs can contain other structs (nested field access: `r.origin.x`)
- Struct instances are passed by reference to functions
- Fields are accessed with dot notation: `p.x`
