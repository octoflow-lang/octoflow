# net — Networking

TCP, UDP, and socket-level networking primitives.

> Network modules are built into the compiler as builtins. No `use` import
> needed — just pass `--allow-net` when running your script.

## TCP

| Function | Description |
|----------|-------------|
| `tcp_connect(host, port)` | Connect to host, returns fd |
| `tcp_send(fd, data)` | Send string data, returns bytes sent |
| `tcp_recv(fd, max)` | Receive up to max bytes as string |
| `tcp_close(fd)` | Close connection |
| `tcp_listen(port)` | Listen on port, returns server fd |
| `tcp_accept(fd)` | Accept connection, returns client fd |

```
let fd = tcp_connect("example.com", 80)
tcp_send(fd, "GET / HTTP/1.0\r\n\r\n")
let response = tcp_recv(fd, 4096)
tcp_close(fd)
print(response)
```

## UDP

| Function | Description |
|----------|-------------|
| `udp_socket()` | Create UDP socket |
| `udp_send_to(fd, host, port, data)` | Send UDP packet |
| `udp_recv_from(fd, max)` | Receive UDP packet |

```
let sock = udp_socket()
udp_send_to(sock, "127.0.0.1", 9000, "hello")
let msg = udp_recv_from(sock, 1024)
```

## TCP Server Example

```
let server = tcp_listen(8080)
print("Listening on port 8080")

while 1
    let client = tcp_accept(server)
    let request = tcp_recv(client, 4096)
    tcp_send(client, "HTTP/1.0 200 OK\r\n\r\nHello!")
    tcp_close(client)
end
```

## See Also

- [web](web.md) — Higher-level HTTP client and server
- [Builtins Reference](../builtins.md) — Complete networking builtins
