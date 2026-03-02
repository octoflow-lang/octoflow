# url_encode (L3)

## Working Example
```flow
use web/url

let raw = "hello world & friends=100%"
let encoded = url_encode(raw)
print("Encoded: {encoded}")

let path_segment = "my folder/sub dir"
let safe_path = url_encode(path_segment)
let full = "https://example.com/files/" + safe_path
print("URL: {full}")

let email = "user+tag@example.com"
let enc_email = url_encode(email)
print("Email param: {enc_email}")
```

## Expected Output
```
Encoded: hello%20world%20%26%20friends%3D100%25
URL: https://example.com/files/my%20folder%2Fsub%20dir
Email param: user%2Btag%40example.com
```

## Common Mistakes
- DON'T: manually replace chars --> DO: `url_encode(s)` handles all special chars
- DON'T: `s.encode()` --> DO: `url_encode(s)` (function not method)
- DON'T: double-encode already-encoded strings --> DO: encode raw strings only

## Edge Cases
- Encodes spaces as %20, ampersands as %26, slashes as %2F
- Safe characters (letters, digits, hyphen, underscore, period) are not encoded
- Use query_string for key=value pairs; url_encode for individual values
