# re_match (L3)

## Working Example
```flow
use string/regex

let text = "Order #12345 placed on 2026-02-28"

let order = re_match(text, "Order #(\\d+)")
print("Order ID: {order}")

let dates = re_find(text, "\\d{4}-\\d{2}-\\d{2}")
print("Dates found: {dates}")

let masked = re_replace(text, "\\d{5}", "XXXXX")
print("Masked: {masked}")

let is_order = re_test(text, "^Order")
print("Starts with Order: {is_order}")

let parts = re_split("a,b,,c,d", ",+")
print("Parts: {parts}")
```

## Expected Output
```
Order ID: ["Order #12345", "12345"]
Dates found: ["2026-02-28"]
Masked: Order #XXXXX placed on 2026-02-28
Starts with Order: 1.0
Parts: ["a", "b", "c", "d"]
```

## Common Mistakes
- DON'T: `text.match(pattern)` --> DO: `re_match(text, pattern)`
- DON'T: `re_match(pattern, text)` --> DO: `re_match(text, pattern)` (string first)
- DON'T: forget to escape backslashes --> DO: `"\\d+"` for digit pattern
- DON'T: `if is_order` --> DO: `if is_order == 1.0`

## Edge Cases
- re_match returns array of capture groups; index 0 is full match
- re_find returns all non-overlapping matches as an array
- re_test returns 1.0 or 0.0; does not return capture groups
