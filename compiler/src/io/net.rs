//! Raw OS TCP/UDP sockets — zero external dependencies.
//!
//! Uses platform socket APIs directly:
//!   Windows: ws2_32.dll  (WinSock2)
//!   Linux/macOS: libc   (POSIX sockets)
//!
//! All socket handles are stored as i64 (fd / SOCKET).
//! Callers encode handles as f32 — precision is fine for fd values < 2^24.

// ── Platform layer ────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
mod os {
    // WinSock2 constants
    pub const AF_INET: i32     = 2;
    pub const SOCK_STREAM: i32 = 1;
    pub const SOCK_DGRAM: i32  = 2;
    pub const IPPROTO_TCP: i32 = 6;
    pub const IPPROTO_UDP: i32 = 17;
    pub const INVALID_SOCKET: usize = usize::MAX;
    pub const SOCKET_ERROR: i32 = -1;
    pub const SOMAXCONN: i32   = 0x7fffffff;
    pub const FIONBIO: u32     = 0x8004667E;
    #[allow(dead_code)]
    pub const WSAEWOULDBLOCK: i32 = 10035;
    pub const SOL_SOCKET: i32  = 0xFFFF;
    pub const SO_REUSEADDR: i32 = 0x0004;

    #[repr(C)]
    pub struct SockAddrIn {
        pub sin_family: u16,
        pub sin_port:   u16,    // big-endian
        pub sin_addr:   u32,    // big-endian IPv4
        pub sin_zero:   [u8; 8],
    }

    #[repr(C)]
    pub struct WsaData {
        pub w_version: u16,
        pub w_high_version: u16,
        pub sz_description: [u8; 257],
        pub sz_system_status: [u8; 129],
        pub i_max_sockets: u16,
        pub i_max_udp_dg: u16,
        pub lp_vendor_info: *mut u8,
    }

    #[link(name = "ws2_32")]
    extern "system" {
        pub fn WSAStartup(w_version_requested: u16, lp_wsa_data: *mut WsaData) -> i32;
        #[allow(dead_code)]
        pub fn WSACleanup() -> i32;
        pub fn socket(af: i32, sock_type: i32, protocol: i32) -> usize;
        pub fn connect(s: usize, name: *const SockAddrIn, name_len: i32) -> i32;
        pub fn bind(s: usize, name: *const SockAddrIn, name_len: i32) -> i32;
        pub fn listen(s: usize, backlog: i32) -> i32;
        pub fn accept(s: usize, addr: *mut SockAddrIn, addr_len: *mut i32) -> usize;
        pub fn send(s: usize, buf: *const u8, len: i32, flags: i32) -> i32;
        pub fn recv(s: usize, buf: *mut u8, len: i32, flags: i32) -> i32;
        pub fn sendto(s: usize, buf: *const u8, len: i32, flags: i32,
                      to: *const SockAddrIn, to_len: i32) -> i32;
        pub fn recvfrom(s: usize, buf: *mut u8, len: i32, flags: i32,
                        from: *mut SockAddrIn, from_len: *mut i32) -> i32;
        pub fn closesocket(s: usize) -> i32;
        pub fn ioctlsocket(s: usize, cmd: u32, argp: *mut u32) -> i32;
        pub fn setsockopt(s: usize, level: i32, optname: i32, optval: *const u8, optlen: i32) -> i32;
        #[allow(dead_code)]
        pub fn WSAGetLastError() -> i32;
        pub fn inet_addr(cp: *const u8) -> u32;
        pub fn htons(host_short: u16) -> u16;
        pub fn ntohl(net_long: u32) -> u32;
        pub fn htonl(host_long: u32) -> u32;
        pub fn gethostbyname(name: *const u8) -> *mut HostEnt;
    }

    #[repr(C)]
    pub struct HostEnt {
        pub h_name: *mut u8,
        pub h_aliases: *mut *mut u8,
        pub h_addrtype: i16,
        pub h_length: i16,
        pub h_addr_list: *mut *mut u8,
    }

    pub fn init_winsock() {
        let mut data: WsaData = unsafe { std::mem::zeroed() };
        unsafe { WSAStartup(0x0202, &mut data); }
    }

    pub fn resolve_host(host: &str) -> Option<u32> {
        let mut h = host.to_string();
        h.push('\0');
        unsafe {
            // Try direct IPv4 first
            let addr = inet_addr(h.as_ptr());
            if addr != 0xffffffff_u32 {
                return Some(addr);
            }
            // DNS lookup
            let he = gethostbyname(h.as_ptr());
            if he.is_null() { return None; }
            let list = (*he).h_addr_list;
            if list.is_null() || (*list).is_null() { return None; }
            let addr_ptr = *list as *const u32;
            Some(*addr_ptr)
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod os {
    pub const AF_INET: i32     = 2;
    pub const SOCK_STREAM: i32 = 1;
    pub const SOCK_DGRAM: i32  = 2;
    pub const IPPROTO_TCP: i32 = 6;
    pub const IPPROTO_UDP: i32 = 17;
    pub const INVALID_SOCKET: usize = usize::MAX;
    pub const SOCKET_ERROR: i32 = -1;
    pub const SOMAXCONN: i32   = 128;
    pub const F_GETFL: i32     = 3;
    pub const F_SETFL: i32     = 4;
    pub const O_NONBLOCK: i32  = 2048; // Linux
    pub const EWOULDBLOCK: i32 = 11;   // Linux EAGAIN
    pub const SOL_SOCKET: i32  = 1;
    pub const SO_REUSEADDR: i32 = 2;

    #[repr(C)]
    pub struct SockAddrIn {
        pub sin_family: u16,
        pub sin_port:   u16,
        pub sin_addr:   u32,
        pub sin_zero:   [u8; 8],
    }

    extern "C" {
        pub fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
        pub fn socket(domain: i32, sock_type: i32, protocol: i32) -> i32;
        pub fn connect(sockfd: i32, addr: *const SockAddrIn, addrlen: u32) -> i32;
        pub fn bind(sockfd: i32, addr: *const SockAddrIn, addrlen: u32) -> i32;
        pub fn listen(sockfd: i32, backlog: i32) -> i32;
        pub fn accept(sockfd: i32, addr: *mut SockAddrIn, addrlen: *mut u32) -> i32;
        pub fn send(sockfd: i32, buf: *const u8, len: usize, flags: i32) -> isize;
        pub fn recv(sockfd: i32, buf: *mut u8, len: usize, flags: i32) -> isize;
        pub fn sendto(sockfd: i32, buf: *const u8, len: usize, flags: i32,
                      dest_addr: *const SockAddrIn, addrlen: u32) -> isize;
        pub fn recvfrom(sockfd: i32, buf: *mut u8, len: usize, flags: i32,
                        src_addr: *mut SockAddrIn, addrlen: *mut u32) -> isize;
        pub fn close(fd: i32) -> i32;
        pub fn setsockopt(sockfd: i32, level: i32, optname: i32, optval: *const u8, optlen: u32) -> i32;
        pub fn inet_addr(cp: *const u8) -> u32;
        pub fn htons(hostshort: u16) -> u16;
        pub fn ntohl(netlong: u32) -> u32;
        pub fn htonl(hostlong: u32) -> u32;
        pub fn gethostbyname(name: *const u8) -> *mut HostEnt;
    }

    #[repr(C)]
    pub struct HostEnt {
        pub h_name: *mut u8,
        pub h_aliases: *mut *mut u8,
        pub h_addrtype: i32,
        pub h_length: i32,
        pub h_addr_list: *mut *mut u8,
    }

    pub fn init_winsock() {} // no-op on POSIX

    pub fn resolve_host(host: &str) -> Option<u32> {
        let mut h = host.to_string();
        h.push('\0');
        unsafe {
            // Try direct IPv4 first
            let addr = inet_addr(h.as_ptr());
            if addr != 0xffffffff_u32 {
                return Some(addr);
            }
            // DNS lookup
            let he = gethostbyname(h.as_ptr());
            if he.is_null() { return None; }
            let list = (*he).h_addr_list;
            if list.is_null() || (*list).is_null() { return None; }
            let addr_ptr = *list as *const u32;
            Some(*addr_ptr)
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────

/// Connect a TCP socket to host:port.
/// Returns socket fd (>= 0) on success, -1 on failure.
pub fn tcp_connect(host: &str, port: u16) -> i64 {
    os::init_winsock();
    let addr_bytes = match os::resolve_host(host) {
        Some(a) => a,
        None => return -1,
    };
    unsafe {
        #[cfg(target_os = "windows")]
        let fd = os::socket(os::AF_INET, os::SOCK_STREAM, os::IPPROTO_TCP);
        #[cfg(not(target_os = "windows"))]
        let fd = os::socket(os::AF_INET, os::SOCK_STREAM, os::IPPROTO_TCP) as usize;

        if fd == os::INVALID_SOCKET { return -1; }

        let sa = os::SockAddrIn {
            sin_family: os::AF_INET as u16,
            sin_port:   os::htons(port),
            sin_addr:   addr_bytes,
            sin_zero:   [0u8; 8],
        };

        #[cfg(target_os = "windows")]
        let rc = os::connect(fd, &sa, std::mem::size_of::<os::SockAddrIn>() as i32);
        #[cfg(not(target_os = "windows"))]
        let rc = os::connect(fd as i32, &sa, std::mem::size_of::<os::SockAddrIn>() as u32);

        if rc == os::SOCKET_ERROR {
            socket_close(fd as i64);
            return -1;
        }
        fd as i64
    }
}

/// Send data over a connected TCP socket.
/// Returns bytes sent (>= 0) or -1 on error.
pub fn tcp_send(fd: i64, data: &str) -> i64 {
    let bytes = data.as_bytes();
    unsafe {
        #[cfg(target_os = "windows")]
        let n = os::send(fd as usize, bytes.as_ptr(), bytes.len() as i32, 0);
        #[cfg(not(target_os = "windows"))]
        let n = os::send(fd as i32, bytes.as_ptr(), bytes.len(), 0);

        if n < 0 { -1 } else { n as i64 }
    }
}

/// Receive up to `max_bytes` from a connected TCP socket.
/// Returns the received string (may be shorter than max_bytes).
pub fn tcp_recv(fd: i64, max_bytes: usize) -> Result<String, String> {
    let max = max_bytes.min(65536);
    let mut buf = vec![0u8; max];
    let n = unsafe {
        #[cfg(target_os = "windows")]
        { os::recv(fd as usize, buf.as_mut_ptr(), buf.len() as i32, 0) }
        #[cfg(not(target_os = "windows"))]
        { os::recv(fd as i32, buf.as_mut_ptr(), buf.len(), 0) as i32 }
    };
    if n < 0 { return Err("tcp_recv failed".into()); }
    buf.truncate(n as usize);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Close a socket (TCP or UDP).
pub fn socket_close(fd: i64) {
    unsafe {
        #[cfg(target_os = "windows")]
        { os::closesocket(fd as usize); }
        #[cfg(not(target_os = "windows"))]
        { os::close(fd as i32); }
    }
}

/// Create a TCP listening socket bound to port.
/// Returns listener fd or -1 on failure. Requires --allow-net.
pub fn tcp_listen(port: u16) -> i64 {
    os::init_winsock();
    unsafe {
        #[cfg(target_os = "windows")]
        let fd = os::socket(os::AF_INET, os::SOCK_STREAM, os::IPPROTO_TCP);
        #[cfg(not(target_os = "windows"))]
        let fd = os::socket(os::AF_INET, os::SOCK_STREAM, os::IPPROTO_TCP) as usize;

        if fd == os::INVALID_SOCKET { return -1; }

        // SO_REUSEADDR — allow immediate rebind after Ctrl-C (no TIME_WAIT block)
        let one: i32 = 1;
        #[cfg(target_os = "windows")]
        { os::setsockopt(fd, os::SOL_SOCKET, os::SO_REUSEADDR, &one as *const i32 as *const u8, 4); }
        #[cfg(not(target_os = "windows"))]
        { os::setsockopt(fd as i32, os::SOL_SOCKET, os::SO_REUSEADDR, &one as *const i32 as *const u8, 4); }

        let sa = os::SockAddrIn {
            sin_family: os::AF_INET as u16,
            sin_port:   os::htons(port),
            sin_addr:   os::htonl(0), // INADDR_ANY
            sin_zero:   [0u8; 8],
        };

        #[cfg(target_os = "windows")]
        { if os::bind(fd, &sa, std::mem::size_of::<os::SockAddrIn>() as i32) == os::SOCKET_ERROR { socket_close(fd as i64); return -1; } }
        #[cfg(not(target_os = "windows"))]
        { if os::bind(fd as i32, &sa, std::mem::size_of::<os::SockAddrIn>() as u32) == os::SOCKET_ERROR { socket_close(fd as i64); return -1; } }

        #[cfg(target_os = "windows")]
        { if os::listen(fd, os::SOMAXCONN) == os::SOCKET_ERROR { socket_close(fd as i64); return -1; } }
        #[cfg(not(target_os = "windows"))]
        { if os::listen(fd as i32, os::SOMAXCONN) == os::SOCKET_ERROR { socket_close(fd as i64); return -1; } }

        fd as i64
    }
}

/// Accept a connection on a listening socket.
/// Returns client fd or -1 on failure.
pub fn tcp_accept(listener: i64) -> i64 {
    let mut client_addr: os::SockAddrIn = unsafe { std::mem::zeroed() };
    unsafe {
        #[cfg(target_os = "windows")]
        {
            let mut addr_len = std::mem::size_of::<os::SockAddrIn>() as i32;
            let client = os::accept(listener as usize, &mut client_addr, &mut addr_len);
            if client == os::INVALID_SOCKET { -1 } else { client as i64 }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut addr_len = std::mem::size_of::<os::SockAddrIn>() as u32;
            let client = os::accept(listener as i32, &mut client_addr, &mut addr_len);
            if client < 0 { -1 } else { client as i64 }
        }
    }
}

/// Non-blocking accept: returns client fd or -1 immediately if no pending client.
pub fn tcp_accept_nonblock(listener: i64) -> i64 {
    unsafe {
        // Set non-blocking
        #[cfg(target_os = "windows")]
        {
            let mut mode: u32 = 1;
            os::ioctlsocket(listener as usize, os::FIONBIO, &mut mode);
        }
        #[cfg(not(target_os = "windows"))]
        {
            let flags = os::fcntl(listener as i32, os::F_GETFL, 0);
            os::fcntl(listener as i32, os::F_SETFL, flags | os::O_NONBLOCK);
        }

        // Try accept
        let mut client_addr: os::SockAddrIn = std::mem::zeroed();
        #[cfg(target_os = "windows")]
        let client = {
            let mut addr_len = std::mem::size_of::<os::SockAddrIn>() as i32;
            os::accept(listener as usize, &mut client_addr, &mut addr_len)
        };
        #[cfg(not(target_os = "windows"))]
        let client = {
            let mut addr_len = std::mem::size_of::<os::SockAddrIn>() as u32;
            os::accept(listener as i32, &mut client_addr, &mut addr_len)
        };

        // Reset to blocking
        #[cfg(target_os = "windows")]
        {
            let mut mode: u32 = 0;
            os::ioctlsocket(listener as usize, os::FIONBIO, &mut mode);
        }
        #[cfg(not(target_os = "windows"))]
        {
            let flags = os::fcntl(listener as i32, os::F_GETFL, 0);
            os::fcntl(listener as i32, os::F_SETFL, flags & !os::O_NONBLOCK);
        }

        // Check result
        #[cfg(target_os = "windows")]
        {
            if client == os::INVALID_SOCKET { -1 } else { client as i64 }
        }
        #[cfg(not(target_os = "windows"))]
        {
            if client < 0 { -1 } else { client as i64 }
        }
    }
}

/// Send raw bytes over a connected TCP socket.
/// Returns bytes sent (>= 0) or -1 on error.
pub fn tcp_send_bytes(fd: i64, data: &[u8]) -> i64 {
    unsafe {
        #[cfg(target_os = "windows")]
        let n = os::send(fd as usize, data.as_ptr(), data.len() as i32, 0);
        #[cfg(not(target_os = "windows"))]
        let n = os::send(fd as i32, data.as_ptr(), data.len(), 0);

        if n < 0 { -1 } else { n as i64 }
    }
}

/// Create a UDP socket. Returns fd or -1 on failure.
pub fn udp_socket() -> i64 {
    os::init_winsock();
    unsafe {
        #[cfg(target_os = "windows")]
        let fd = os::socket(os::AF_INET, os::SOCK_DGRAM, os::IPPROTO_UDP);
        #[cfg(not(target_os = "windows"))]
        let fd = os::socket(os::AF_INET, os::SOCK_DGRAM, os::IPPROTO_UDP) as usize;

        if fd == os::INVALID_SOCKET { -1 } else { fd as i64 }
    }
}

/// Send UDP datagram to host:port.
pub fn udp_send_to(fd: i64, host: &str, port: u16, data: &str) -> i64 {
    let addr_bytes = match os::resolve_host(host) {
        Some(a) => a,
        None => return -1,
    };
    let bytes = data.as_bytes();
    let sa = os::SockAddrIn {
        sin_family: os::AF_INET as u16,
        sin_port:   unsafe { os::htons(port) },
        sin_addr:   addr_bytes,
        sin_zero:   [0u8; 8],
    };
    let n = unsafe {
        #[cfg(target_os = "windows")]
        { os::sendto(fd as usize, bytes.as_ptr(), bytes.len() as i32, 0, &sa,
                     std::mem::size_of::<os::SockAddrIn>() as i32) }
        #[cfg(not(target_os = "windows"))]
        { os::sendto(fd as i32, bytes.as_ptr(), bytes.len(), 0, &sa,
                     std::mem::size_of::<os::SockAddrIn>() as u32) as i32 }
    };
    if n < 0 { -1 } else { n as i64 }
}

/// Receive a UDP datagram (up to max_bytes). Returns (data, from_addr).
pub fn udp_recv_from(fd: i64, max_bytes: usize) -> Result<(String, String), String> {
    let max = max_bytes.min(65536);
    let mut buf = vec![0u8; max];
    let mut from: os::SockAddrIn = unsafe { std::mem::zeroed() };
    let n = unsafe {
        #[cfg(target_os = "windows")]
        {
            let mut from_len = std::mem::size_of::<os::SockAddrIn>() as i32;
            os::recvfrom(fd as usize, buf.as_mut_ptr(), buf.len() as i32, 0, &mut from, &mut from_len)
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut from_len = std::mem::size_of::<os::SockAddrIn>() as u32;
            os::recvfrom(fd as i32, buf.as_mut_ptr(), buf.len(), 0, &mut from, &mut from_len) as i32
        }
    };
    if n < 0 { return Err("udp_recv_from failed".into()); }
    buf.truncate(n as usize);
    let data = String::from_utf8_lossy(&buf).into_owned();
    // Format source address as "A.B.C.D:port"
    let ip = unsafe { os::ntohl(from.sin_addr) };
    let port = u16::from_be(from.sin_port);
    let from_str = format!("{}.{}.{}.{}:{}",
        (ip >> 24) & 0xff, (ip >> 16) & 0xff, (ip >> 8) & 0xff, ip & 0xff, port);
    Ok((data, from_str))
}
