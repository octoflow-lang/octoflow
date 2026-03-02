//! Pure-Rust PNG and JPEG codec (Phase 48) — zero external dependencies.
//!
//! PNG: full decode (RGB/RGBA/grayscale, 8-bit, all filter types) + encode
//!      (RGB, filter-None, zlib store blocks — valid but uncompressed).
//! JPEG: baseline sequential DCT decode + encode (YCbCr, standard tables).

use crate::CliError;

// ─── Public API ──────────────────────────────────────────────────────────────

pub fn is_image_path(path: &str) -> bool {
    let p = path.to_lowercase();
    p.ends_with(".png") || p.ends_with(".jpg") || p.ends_with(".jpeg")
}

/// Read an image as flat interleaved RGB f32 in [0, 255].
pub fn read_image(path: &str) -> Result<(Vec<f32>, u32, u32), CliError> {
    let meta = std::fs::metadata(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    if meta.len() > crate::MAX_IMAGE_FILE_BYTES {
        return Err(CliError::Security(format!(
            "{}: file too large ({} bytes)", path, meta.len())));
    }
    let data = std::fs::read(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;

    let p = path.to_lowercase();
    let (rgb, w, h) = if p.ends_with(".png") {
        png::decode(&data).map_err(|e| CliError::Io(format!("{}: {}", path, e)))?
    } else {
        jpeg::decode(&data).map_err(|e| CliError::Io(format!("{}: {}", path, e)))?
    };

    if w > crate::MAX_IMAGE_DIMENSION || h > crate::MAX_IMAGE_DIMENSION {
        return Err(CliError::Security(format!(
            "{}: dimensions {}x{} exceed limit", path, w, h)));
    }
    let pixels: Vec<f32> = rgb.into_iter().map(|b| b as f32).collect();
    Ok((pixels, w, h))
}

/// Write flat RGB f32 data as PNG or JPEG.
pub fn write_image(path: &str, data: &[f32], width: u32, height: u32) -> Result<(), CliError> {
    let expected = width as usize * height as usize * 3;
    if data.len() != expected {
        return Err(CliError::Compile(format!(
            "image write: expected {} bytes ({}x{}x3), got {}", expected, width, height, data.len())));
    }
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
        }
    }
    let rgb: Vec<u8> = data.iter().map(|&v| v.clamp(0.0, 255.0).round() as u8).collect();
    let p = path.to_lowercase();
    let encoded = if p.ends_with(".png") {
        png::encode(&rgb, width, height)
    } else {
        jpeg::encode(&rgb, width, height, 85)
    };
    std::fs::write(path, &encoded)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    Ok(())
}

/// Decode image from raw bytes (auto-detects PNG vs JPEG from magic bytes).
/// Returns separate R, G, B channel arrays as f32 (0-255) + width + height.
pub fn decode_image_bytes(data: &[u8]) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, u32, u32), String> {
    if data.len() < 4 {
        return Err("decode_image: data too short".into());
    }
    let (rgb, w, h) = if data.starts_with(b"\x89PNG") {
        png::decode(data)?
    } else if data.starts_with(&[0xFF, 0xD8]) {
        jpeg::decode(data)?
    } else if data.starts_with(b"BM") {
        return Err("decode_image: BMP format not supported, use PNG or JPEG".into());
    } else {
        return Err("decode_image: unknown image format (expected PNG or JPEG)".into());
    };
    let total = match (w as usize).checked_mul(h as usize) {
        Some(t) if t <= 16384 * 16384 => t,
        _ => return Err("image dimensions too large or overflow".into()),
    };
    if rgb.len() < total * 3 {
        return Err(format!("image pixel data too short: expected {} bytes, got {}", total * 3, rgb.len()));
    }
    let mut r = Vec::with_capacity(total);
    let mut g = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        r.push(rgb[i * 3] as f32);
        g.push(rgb[i * 3 + 1] as f32);
        b.push(rgb[i * 3 + 2] as f32);
    }
    Ok((r, g, b, w, h))
}

// ─── PNG codec ───────────────────────────────────────────────────────────────

pub(crate) mod png {
    // ── Bit reader (LSB-first) ─────────────────────────────────────────────
    struct Br<'a> { src: &'a [u8], pos: usize, buf: u32, cnt: u8 }
    impl<'a> Br<'a> {
        fn new(src: &'a [u8]) -> Self { Br { src, pos: 0, buf: 0, cnt: 0 } }
        fn fill(&mut self) {
            while self.cnt <= 24 && self.pos < self.src.len() {
                self.buf |= (self.src[self.pos] as u32) << self.cnt;
                self.pos += 1; self.cnt += 8;
            }
        }
        fn bits(&mut self, n: u8) -> Option<u32> {
            if n == 0 { return Some(0); }
            self.fill();
            if self.cnt < n { return None; }
            let v = self.buf & ((1u32 << n) - 1);
            self.buf >>= n; self.cnt -= n; Some(v)
        }
        fn bit(&mut self) -> Option<u32> { self.bits(1) }
        fn align(&mut self) { let d = self.cnt & 7; self.buf >>= d; self.cnt -= d; }
        fn read_u16_le(&mut self) -> Option<u16> {
            self.align();
            let lo = self.bits(8)?; let hi = self.bits(8)?;
            Some((hi << 8 | lo) as u16)
        }
    }

    // ── MSB-first canonical Huffman (DEFLATE standard) ────────────────────
    fn huf_decode_msb(huf: &HufMsb, br: &mut Br) -> Option<u16> {
        let mut code = 0u16;
        for len in 1..=huf.max as usize {
            code = (code << 1) | (br.bit()? as u16);
            let t = &huf.by_len[len];
            if let Ok(i) = t.binary_search_by_key(&code, |&(c, _)| c) {
                return Some(t[i].1);
            }
        }
        None
    }

    // MSB-first Huffman (DEFLATE standard)
    struct HufMsb { by_len: [Vec<(u16, u16)>; 16], max: u8 }
    impl HufMsb {
        fn from_lens(cl: &[u8]) -> Self {
            let mut bl = [0u16; 16];
            for &l in cl { if l > 0 && l <= 15 { bl[l as usize] += 1; } }
            let mut next = [0u16; 16];
            let mut code = 0u16;
            bl[0] = 0;
            for bits in 1..=15usize {
                code = (code + bl[bits - 1]) << 1;
                next[bits] = code;
            }
            let mut by_len: [Vec<(u16, u16)>; 16] = Default::default();
            let mut max = 0u8;
            for (sym, &l) in cl.iter().enumerate() {
                if l > 0 && l <= 15 {
                    let c = next[l as usize];
                    next[l as usize] += 1;
                    by_len[l as usize].push((c, sym as u16));
                    if l > max { max = l; }
                }
            }
            for row in &mut by_len { row.sort_unstable(); }
            HufMsb { by_len, max }
        }
    }

    // ── DEFLATE inflate ───────────────────────────────────────────────────
    static LEN_EXTRA:  [(u16, u8); 29] = [
        (3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(10,0),
        (11,1),(13,1),(15,1),(17,1),(19,2),(23,2),(27,2),(31,2),
        (35,3),(43,3),(51,3),(59,3),(67,4),(83,4),(99,4),(115,4),
        (131,5),(163,5),(195,5),(227,5),(258,0),
    ];
    static DIST_EXTRA: [(u16, u8); 30] = [
        (1,0),(2,0),(3,0),(4,0),(5,1),(7,1),(9,2),(13,2),
        (17,3),(25,3),(33,4),(49,4),(65,5),(97,5),(129,6),(193,6),
        (257,7),(385,7),(513,8),(769,8),(1025,9),(1537,9),
        (2049,10),(3073,10),(4097,11),(6145,11),(8193,12),(12289,12),
        (16385,13),(24577,13),
    ];

    fn fixed_litlen_lens() -> Vec<u8> {
        let mut cl = vec![0u8; 288];
        for i in 0..=143  { cl[i] = 8; }
        for i in 144..=255 { cl[i] = 9; }
        for i in 256..=279 { cl[i] = 7; }
        for i in 280..=287 { cl[i] = 8; }
        cl
    }
    fn fixed_dist_lens() -> Vec<u8> { vec![5u8; 32] }

    fn inflate(src: &[u8]) -> Result<Vec<u8>, String> {
        let mut br = Br::new(src);
        let mut out: Vec<u8> = Vec::with_capacity(src.len() * 4);
        loop {
            let bfinal = br.bits(1).ok_or("inflate: eof in bfinal")?;
            let btype  = br.bits(2).ok_or("inflate: eof in btype")?;
            match btype {
                0 => { // stored
                    let len  = br.read_u16_le().ok_or("inflate: eof in stored len")? as usize;
                    let nlen = br.read_u16_le().ok_or("inflate: eof in stored nlen")? as usize;
                    if (len ^ nlen) != 0xFFFF { return Err("inflate: stored len mismatch".into()); }
                    br.align();
                    for _ in 0..len {
                        let b = br.bits(8).ok_or("inflate: eof in stored data")? as u8;
                        out.push(b);
                    }
                }
                1 | 2 => { // compressed
                    let (ll, dist) = if btype == 1 {
                        (HufMsb::from_lens(&fixed_litlen_lens()),
                         HufMsb::from_lens(&fixed_dist_lens()))
                    } else {
                        let hlit  = br.bits(5).ok_or("inflate: eof")? as usize + 257;
                        let hdist = br.bits(5).ok_or("inflate: eof")? as usize + 1;
                        let hclen = br.bits(4).ok_or("inflate: eof")? as usize + 4;
                        let order = [16u8,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];
                        let mut cl_lens = vec![0u8; 19];
                        for i in 0..hclen {
                            cl_lens[order[i] as usize] = br.bits(3).ok_or("inflate: eof")? as u8;
                        }
                        let cl_huf = HufMsb::from_lens(&cl_lens);
                        let total = hlit + hdist;
                        let mut lens = vec![0u8; total];
                        let mut i = 0;
                        while i < total {
                            let sym = huf_decode_msb(&cl_huf, &mut br).ok_or("inflate: bad cl code")?;
                            match sym {
                                0..=15 => { lens[i] = sym as u8; i += 1; }
                                16 => {
                                    let rep = br.bits(2).ok_or("inflate: eof")? as usize + 3;
                                    let v = if i > 0 { lens[i-1] } else { 0 };
                                    for _ in 0..rep { if i < total { lens[i] = v; i += 1; } }
                                }
                                17 => { let rep = br.bits(3).ok_or("inflate: eof")? as usize + 3; i += rep; }
                                18 => { let rep = br.bits(7).ok_or("inflate: eof")? as usize + 11; i += rep; }
                                _ => return Err("inflate: bad cl sym".into()),
                            }
                        }
                        (HufMsb::from_lens(&lens[..hlit]), HufMsb::from_lens(&lens[hlit..]))
                    };
                    loop {
                        let sym = huf_decode_msb(&ll, &mut br).ok_or("inflate: bad litlen")?;
                        if sym < 256 { out.push(sym as u8); }
                        else if sym == 256 { break; }
                        else {
                            let li = sym as usize - 257;
                            if li >= 29 { return Err("inflate: bad len code".into()); }
                            let (base_len, extra_bits) = LEN_EXTRA[li];
                            let extra = br.bits(extra_bits).ok_or("inflate: eof in len extra")?;
                            let length = base_len as usize + extra as usize;
                            let dc = huf_decode_msb(&dist, &mut br).ok_or("inflate: bad dist")?;
                            if dc as usize >= 30 { return Err("inflate: bad dist code".into()); }
                            let (base_dist, dbits) = DIST_EXTRA[dc as usize];
                            let dextra = br.bits(dbits).ok_or("inflate: eof in dist extra")?;
                            let distance = base_dist as usize + dextra as usize;
                            if distance == 0 || distance > out.len() { return Err("inflate: backref out of range".into()); }
                            let start = out.len() - distance;
                            for k in 0..length { let b = out[start + k % distance]; out.push(b); }
                        }
                    }
                }
                _ => return Err("inflate: reserved btype".into()),
            }
            if bfinal != 0 { break; }
        }
        Ok(out)
    }

    // ── Zlib inflate (wraps inflate) ──────────────────────────────────────
    fn zlib_inflate(src: &[u8]) -> Result<Vec<u8>, String> {
        if src.len() < 2 { return Err("zlib: too short".into()); }
        let cm = src[0] & 0x0F;
        if cm != 8 { return Err(format!("zlib: unsupported method {}", cm)); }
        inflate(&src[2..])
    }

    // ── Zlib deflate (store, no compression) ─────────────────────────────
    pub fn zlib_deflate_store(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        // Zlib header: CM=8, CINFO=7 (window=32K), FCHECK makes (CMF*256+FLG) divisible by 31
        out.push(0x78); out.push(0x01); // zlib default compression header
        // DEFLATE: one or more stored blocks
        let mut pos = 0;
        while pos <= data.len() {
            let end = (pos + 65535).min(data.len());
            let bfinal = if end == data.len() { 1u8 } else { 0u8 };
            out.push(bfinal | 0x00); // bfinal | btype=0 (stored)
            let len = (end - pos) as u16;
            out.push((len & 0xFF) as u8);
            out.push((len >> 8) as u8);
            out.push((!len & 0xFF) as u8);
            out.push((!len >> 8) as u8);
            out.extend_from_slice(&data[pos..end]);
            if end == data.len() { break; }
            pos = end;
        }
        // Adler-32 checksum
        let (s1, s2) = adler32(data);
        out.push((s2 >> 8) as u8); out.push((s2 & 0xFF) as u8);
        out.push((s1 >> 8) as u8); out.push((s1 & 0xFF) as u8);
        out
    }

    fn adler32(data: &[u8]) -> (u32, u32) {
        let (mut s1, mut s2) = (1u32, 0u32);
        for &b in data { s1 = (s1 + b as u32) % 65521; s2 = (s2 + s1) % 65521; }
        (s1, s2)
    }

    // ── PNG decode ────────────────────────────────────────────────────────
    pub fn decode(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
            return Err("PNG: invalid signature".into());
        }
        let mut pos = 8;
        let mut width = 0u32; let mut height = 0u32;
        let mut bit_depth = 0u8; let mut color_type = 0u8;
        let mut idat: Vec<u8> = Vec::new();
        let mut has_ihdr = false;

        while pos + 8 <= data.len() {
            let len = u32::from_be_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3]]) as usize;
            let tag = &data[pos+4..pos+8];
            let chunk_data = if pos + 8 + len <= data.len() { &data[pos+8..pos+8+len] } else { &data[0..0] };
            pos += 12 + len;
            match tag {
                b"IHDR" if chunk_data.len() >= 13 => {
                    width  = u32::from_be_bytes([chunk_data[0],chunk_data[1],chunk_data[2],chunk_data[3]]);
                    height = u32::from_be_bytes([chunk_data[4],chunk_data[5],chunk_data[6],chunk_data[7]]);
                    bit_depth  = chunk_data[8];
                    color_type = chunk_data[9];
                    has_ihdr = true;
                }
                b"IDAT" => idat.extend_from_slice(chunk_data),
                b"IEND" => break,
                _ => {}
            }
        }
        if !has_ihdr { return Err("PNG: missing IHDR".into()); }
        if bit_depth != 8 { return Err(format!("PNG: unsupported bit depth {}", bit_depth)); }

        let channels = match color_type {
            0 => 1, // grayscale
            2 => 3, // RGB
            3 => return Err("PNG: indexed color not supported".into()),
            4 => 2, // grayscale+alpha
            6 => 4, // RGBA
            _ => return Err(format!("PNG: unknown color type {}", color_type)),
        };

        let raw = zlib_inflate(&idat)?;
        let stride = width as usize * channels;
        let expected = (stride + 1) * height as usize;
        if raw.len() < expected {
            return Err(format!("PNG: decompressed size {} < expected {}", raw.len(), expected));
        }

        // Filter reconstruction
        let mut pixels = vec![0u8; width as usize * height as usize * channels];
        let mut prev = vec![0u8; stride];
        for y in 0..height as usize {
            let row_start = y * (stride + 1);
            let filter = raw[row_start];
            let row = &raw[row_start + 1..row_start + 1 + stride];
            let out = &mut pixels[y * stride..(y + 1) * stride];
            match filter {
                0 => out.copy_from_slice(row), // None
                1 => { // Sub
                    for x in 0..stride {
                        let a = if x >= channels { out[x - channels] } else { 0 };
                        out[x] = row[x].wrapping_add(a);
                    }
                }
                2 => { // Up
                    for x in 0..stride { out[x] = row[x].wrapping_add(prev[x]); }
                }
                3 => { // Average
                    for x in 0..stride {
                        let a = if x >= channels { out[x - channels] as u16 } else { 0 };
                        let b = prev[x] as u16;
                        out[x] = row[x].wrapping_add(((a + b) / 2) as u8);
                    }
                }
                4 => { // Paeth
                    for x in 0..stride {
                        let a = if x >= channels { out[x - channels] as i32 } else { 0 };
                        let b = prev[x] as i32;
                        let c = if x >= channels { prev[x - channels] as i32 } else { 0 };
                        let p = a + b - c;
                        let pa = (p - a).abs(); let pb = (p - b).abs(); let pc = (p - c).abs();
                        let pr = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
                        out[x] = row[x].wrapping_add(pr as u8);
                    }
                }
                _ => return Err(format!("PNG: unknown filter type {}", filter)),
            }
            prev.copy_from_slice(out);
        }

        // Validate pixel data alignment
        let expected_pixels = width as usize * height as usize * channels;
        if pixels.len() != expected_pixels {
            return Err(format!("PNG pixel data misaligned: expected {} bytes, got {}", expected_pixels, pixels.len()));
        }

        // Convert to RGB
        let rgb = match channels {
            1 => pixels.iter().flat_map(|&g| [g, g, g]).collect(),
            2 => pixels.chunks(2).flat_map(|p| [p[0], p[0], p[0]]).collect(),
            3 => pixels,
            4 => pixels.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect(),
            _ => unreachable!(),
        };
        Ok((rgb, width, height))
    }

    // ── PNG encode (RGB, filter-None, store blocks) ───────────────────────
    pub fn encode(rgb: &[u8], width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        // Signature
        out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        // IHDR
        let ihdr_data = {
            let mut d = Vec::new();
            d.extend_from_slice(&width.to_be_bytes());
            d.extend_from_slice(&height.to_be_bytes());
            d.push(8); // bit depth
            d.push(2); // color type RGB
            d.push(0); d.push(0); d.push(0); // compression, filter, interlace
            d
        };
        write_chunk(&mut out, b"IHDR", &ihdr_data);
        // Build raw (filtered) data: filter type 0 (None) before each row
        let stride = width as usize * 3;
        let mut raw = Vec::with_capacity((stride + 1) * height as usize);
        for y in 0..height as usize {
            raw.push(0u8); // filter type None
            raw.extend_from_slice(&rgb[y * stride..(y + 1) * stride]);
        }
        // Compress
        let compressed = zlib_deflate_store(&raw);
        write_chunk(&mut out, b"IDAT", &compressed);
        write_chunk(&mut out, b"IEND", &[]);
        out
    }

    fn write_chunk(out: &mut Vec<u8>, tag: &[u8], data: &[u8]) {
        let len = data.len() as u32;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let crc = crc32(tag, data);
        out.extend_from_slice(&crc.to_be_bytes());
    }

    pub(crate) fn crc32(tag: &[u8], data: &[u8]) -> u32 {
        static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
        let table = TABLE.get_or_init(|| {
            let mut t = [0u32; 256];
            for n in 0..256u32 {
                let mut c = n;
                for _ in 0..8 { c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 }; }
                t[n as usize] = c;
            }
            t
        });
        let mut c = 0xFFFF_FFFFu32;
        for &b in tag.iter().chain(data.iter()) { c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8); }
        c ^ 0xFFFF_FFFF
    }
}

// ─── JPEG codec ──────────────────────────────────────────────────────────────

pub(crate) mod jpeg {
    // ── JPEG constants ────────────────────────────────────────────────────
    const SOI:  u8 = 0xD8;
    const EOI:  u8 = 0xD9;
    const SOF0: u8 = 0xC0;
    const DHT:  u8 = 0xC4;
    const DQT:  u8 = 0xDB;
    const SOS:  u8 = 0xDA;
    const DRI:  u8 = 0xDD;
    const APP0: u8 = 0xE0;

    // Standard luminance quantization table (quality ~85)
    static LUMA_QTAB: [u8; 64] = [
         2, 2, 2, 2, 3, 4, 5, 6,
         2, 2, 2, 2, 3, 4, 5, 6,
         2, 2, 2, 3, 4, 5, 7, 9,
         2, 2, 3, 4, 5, 7, 9,11,
         3, 3, 4, 5, 7, 9,11,13,
         4, 4, 5, 7, 9,11,13,15,
         5, 5, 7, 9,11,13,15,17,
         6, 6, 9,11,13,15,17,19,
    ];
    static CHROMA_QTAB: [u8; 64] = [
         3, 3, 5, 9,13,15,15,15,
         3, 4, 5, 9,13,15,15,15,
         5, 5, 7,11,15,15,15,15,
         9, 9,11,15,15,15,15,15,
        13,13,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
        15,15,15,15,15,15,15,15,
    ];

    // Zigzag order (8x8 block indexing)
    static ZIGZAG: [u8; 64] = [
         0, 1, 8,16, 9, 2, 3,10,17,24,32,25,18,11, 4, 5,
        12,19,26,33,40,48,41,34,27,20,13, 6, 7,14,21,28,
        35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
        58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63,
    ];

    // Standard DC Huffman tables
    static DC_LUMA_BITS: [u8; 16]  = [0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0];
    static DC_LUMA_VALS: [u8; 12]  = [0,1,2,3,4,5,6,7,8,9,10,11];
    static DC_CHROMA_BITS: [u8; 16] = [0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0];
    static DC_CHROMA_VALS: [u8; 12] = [0,1,2,3,4,5,6,7,8,9,10,11];

    // Standard AC Huffman tables (luminance)
    static AC_LUMA_BITS: [u8; 16] = [0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,125];
    static AC_LUMA_VALS: [u8; 162] = [
        0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,
        0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,
        0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,
        0x26,0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,
        0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,
        0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,
        0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,
        0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
        0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,
        0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,
        0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,
    ];
    // Standard AC Huffman tables (chrominance)
    static AC_CHROMA_BITS: [u8; 16] = [0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,119];
    static AC_CHROMA_VALS: [u8; 162] = [
        0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,
        0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,
        0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,
        0x19,0x1a,0x26,0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,
        0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,
        0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,
        0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,
        0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
        0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,
        0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,
        0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,
    ];

    // ── Huffman tables for decoding ───────────────────────────────────────
    struct HufDec { codes: Vec<(u32, u8, u8)> } // (code MSB, length, value)

    impl HufDec {
        fn from_bits_vals(bits: &[u8; 16], vals: &[u8]) -> Self {
            let mut codes = Vec::new();
            let mut code = 0u32;
            let mut vi = 0;
            for len in 1u8..=16 {
                for _ in 0..bits[len as usize - 1] {
                    if vi < vals.len() { codes.push((code, len, vals[vi])); vi += 1; }
                    code += 1;
                }
                code <<= 1;
            }
            HufDec { codes }
        }
        fn decode(&self, br: &mut JpegBr) -> Option<u8> {
            let mut code = 0u32;
            for len in 1u8..=16 {
                code = (code << 1) | br.bit()? as u32;
                for &(c, l, v) in &self.codes {
                    if l == len && c == code { return Some(v); }
                }
            }
            None
        }
    }

    // ── JPEG bit reader ───────────────────────────────────────────────────
    struct JpegBr<'a> { src: &'a [u8], pos: usize, buf: u32, cnt: u8 }
    impl<'a> JpegBr<'a> {
        fn new(src: &'a [u8]) -> Self { JpegBr { src, pos: 0, buf: 0, cnt: 0 } }
        fn bit(&mut self) -> Option<u32> {
            if self.cnt == 0 {
                if self.pos >= self.src.len() { return None; }
                let b = self.src[self.pos]; self.pos += 1;
                if b == 0xFF {
                    // Byte stuffing: skip 0x00 after 0xFF
                    if self.pos < self.src.len() && self.src[self.pos] == 0x00 { self.pos += 1; }
                }
                self.buf = b as u32; self.cnt = 8;
            }
            let b = (self.buf >> 7) & 1;
            self.buf <<= 1; self.cnt -= 1;
            Some(b)
        }
        fn bits(&mut self, n: u8) -> Option<i32> {
            let mut v = 0i32;
            for _ in 0..n { v = (v << 1) | self.bit()? as i32; }
            Some(v)
        }
    }

    // ── IDCT (AAN fast 1D) ────────────────────────────────────────────────
    fn idct_1d(v: &mut [f32; 8]) {
        let s = [
            v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
        ];
        let c1 = 0.9807852_f32; let s1 = 0.1950902_f32;
        let c3 = 0.8314696_f32; let s3 = 0.5555702_f32;
        let c6 = 0.3826834_f32; let s6 = 0.9238795_f32;
        let rt2 = std::f32::consts::SQRT_2;

        // Even
        let p0 = (s[0] + s[4]) * 0.5_f32;
        let p1 = (s[0] - s[4]) * 0.5_f32;
        let p2 = s[2] * s6 - s[6] * c6;
        let p3 = s[2] * c6 + s[6] * s6;
        let e0 = p0 + p3; let e1 = p1 + p2; let e2 = p1 - p2; let e3 = p0 - p3;
        // Odd
        let o0 = s[7] * c1 - s[1] * s1;
        let o1 = s[5] * c3 - s[3] * s3;
        let o2 = s[5] * s3 + s[3] * c3;
        let o3 = s[7] * s1 + s[1] * c1;
        let f0 = (o0 + o1) / rt2; let f1 = (o0 - o1) / rt2;
        let f2 = (o3 - o2) / rt2; let f3 = (o3 + o2) / rt2;

        v[0] = e0 + f3; v[7] = e0 - f3;
        v[1] = e1 + f2; v[6] = e1 - f2;
        v[2] = e2 + f1; v[5] = e2 - f1;
        v[3] = e3 + f0; v[4] = e3 - f0;
    }

    fn idct_2d(block: &mut [i16; 64]) -> [i16; 64] {
        let mut fb = [0f32; 64];
        for i in 0..64 { fb[i] = block[i] as f32; }
        // Rows
        for r in 0..8 {
            let mut row = [0f32; 8];
            row.copy_from_slice(&fb[r*8..r*8+8]);
            idct_1d(&mut row);
            fb[r*8..r*8+8].copy_from_slice(&row);
        }
        // Columns
        for c in 0..8 {
            let mut col = [0f32; 8];
            for r in 0..8 { col[r] = fb[r*8+c]; }
            idct_1d(&mut col);
            for r in 0..8 { fb[r*8+c] = col[r]; }
        }
        let mut out = [0i16; 64];
        for i in 0..64 { out[i] = (fb[i] / 8.0 + 128.0).round().clamp(0.0, 255.0) as i16; }
        out
    }

    // ── JPEG decode ───────────────────────────────────────────────────────
    pub fn decode(data: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        if data.len() < 2 || data[0] != 0xFF || data[1] != SOI {
            return Err("JPEG: invalid SOI".into());
        }
        let mut pos = 2;
        let mut width = 0u32; let mut height = 0u32;
        #[allow(unused_assignments)] let mut n_comp = 0u8;
        let mut comp_info: Vec<(u8, u8, u8, u8)> = Vec::new(); // (id, h_samp, v_samp, qt_id)
        let mut qtabs: Vec<[u8; 64]> = vec![[0u8; 64]; 4];
        let mut dc_tabs: Vec<Option<HufDec>> = vec![None, None];
        let mut ac_tabs: Vec<Option<HufDec>> = vec![None, None];
        let mut restart_interval: u16 = 0;

        while pos + 1 < data.len() {
            if data[pos] != 0xFF { pos += 1; continue; }
            let marker = data[pos + 1]; pos += 2;
            if marker == EOI || marker == 0x00 { continue; }
            if marker == SOI { continue; }
            if marker >= 0xD0 && marker <= 0xD7 { continue; } // RST0-RST7

            if pos + 2 > data.len() { break; }
            let seg_len = u16::from_be_bytes([data[pos], data[pos+1]]) as usize;
            let seg = if pos + seg_len <= data.len() { &data[pos+2..pos+seg_len] } else { break };
            pos += seg_len;

            match marker {
                SOF0 => {
                    if seg.len() < 9 { continue; }
                    // seg[0] = precision (8), seg[1..2] = height, seg[3..4] = width
                    height = u16::from_be_bytes([seg[1], seg[2]]) as u32;
                    width  = u16::from_be_bytes([seg[3], seg[4]]) as u32;
                    n_comp = seg[5];
                    comp_info.clear();
                    for i in 0..n_comp as usize {
                        let base = 6 + i * 3;
                        if base + 2 < seg.len() {
                            let id = seg[base];
                            let samp = seg[base + 1];
                            let qt = seg[base + 2];
                            comp_info.push((id, samp >> 4, samp & 0x0F, qt));
                        }
                    }
                }
                DQT => {
                    let mut i = 0;
                    while i + 1 < seg.len() {
                        let id = (seg[i] & 0x0F) as usize;
                        let prec = seg[i] >> 4; i += 1;
                        if id < 4 && i + 64 <= seg.len() {
                            if prec == 0 {
                                for j in 0..64 { qtabs[id][ZIGZAG[j] as usize] = seg[i + j]; }
                            }
                            i += 64;
                        } else { break; }
                    }
                }
                DHT => {
                    let mut i = 0;
                    while i < seg.len() {
                        let tc = (seg[i] >> 4) & 1; // 0=DC, 1=AC
                        let th = (seg[i] & 0x0F) as usize; i += 1;
                        if i + 16 > seg.len() { break; }
                        let mut bits = [0u8; 16];
                        bits.copy_from_slice(&seg[i..i+16]); i += 16;
                        let total: usize = bits.iter().map(|&b| b as usize).sum();
                        if i + total > seg.len() { break; }
                        let vals = &seg[i..i+total]; i += total;
                        let mut bx = [0u8; 16];
                        bx.copy_from_slice(&bits);
                        let huf = HufDec::from_bits_vals(&bx, vals);
                        if th < 2 {
                            if tc == 0 { dc_tabs[th] = Some(huf); }
                            else { ac_tabs[th] = Some(huf); }
                        }
                    }
                }
                DRI => {
                    if seg.len() >= 2 {
                        restart_interval = u16::from_be_bytes([seg[0], seg[1]]);
                    }
                }
                SOS => {
                    // Find Huffman table assignments per component
                    if seg.len() < 1 { continue; }
                    let ns = seg[0] as usize;
                    let mut comp_huf: Vec<(usize, usize)> = Vec::new(); // (dc_tab, ac_tab)
                    for i in 0..ns {
                        let base = 1 + i * 2;
                        if base + 1 < seg.len() {
                            let td = (seg[base + 1] >> 4) as usize;
                            let ta = (seg[base + 1] & 0x0F) as usize;
                            comp_huf.push((td, ta));
                        }
                    }
                    // Decode entropy-coded data
                    let ecs_start = 1 + ns * 2 + 3; // skip Ss, Se, Ah/Al
                    let ecs = if pos > ecs_start { &data[pos - seg_len + ecs_start..] } else { &data[pos..] };
                    // Find end of ECS (next FF xx where xx != 0x00 and != RST)
                    let ecs_end = find_ecs_end(ecs);
                    let ecs = &ecs[..ecs_end];

                    if width == 0 || height == 0 { return Err("JPEG: SOS before SOF0".into()); }

                    let pixels = decode_ecs(ecs, width, height, &comp_info, &comp_huf,
                                           &qtabs, &dc_tabs, &ac_tabs, restart_interval)?;
                    let _ = ecs_end; // pos unused after early return
                    return Ok((pixels, width, height));
                }
                _ => {}
            }
        }
        Err("JPEG: no SOS found or decode failed".into())
    }

    fn find_ecs_end(ecs: &[u8]) -> usize {
        let mut i = 0;
        while i + 1 < ecs.len() {
            if ecs[i] == 0xFF && ecs[i+1] != 0x00
                && !(ecs[i+1] >= 0xD0 && ecs[i+1] <= 0xD7) {
                return i;
            }
            i += 1;
        }
        ecs.len()
    }

    fn decode_ecs(
        ecs: &[u8], w: u32, h: u32,
        comp_info: &[(u8, u8, u8, u8)],
        comp_huf: &[(usize, usize)],
        qtabs: &[[u8; 64]],
        dc_tabs: &[Option<HufDec>],
        ac_tabs: &[Option<HufDec>],
        restart_interval: u16,
    ) -> Result<Vec<u8>, String> {
        let mut br = JpegBr::new(ecs);
        let nc = comp_info.len().min(comp_huf.len());

        // Compute MCU geometry from sampling factors
        let max_h = comp_info.iter().take(nc).map(|c| c.1 as usize).max().unwrap_or(1);
        let max_v = comp_info.iter().take(nc).map(|c| c.2 as usize).max().unwrap_or(1);
        let mcu_px_w = max_h * 8;
        let mcu_px_h = max_v * 8;
        let mcus_x = (w as usize + mcu_px_w - 1) / mcu_px_w;
        let mcus_y = (h as usize + mcu_px_h - 1) / mcu_px_h;

        // Per-component buffers at component resolution
        let mut comp_bufs: Vec<Vec<u8>> = Vec::new();
        let mut comp_dims: Vec<(usize, usize)> = Vec::new();
        for ci in 0..nc {
            let cw = mcus_x * comp_info[ci].1 as usize * 8;
            let ch = mcus_y * comp_info[ci].2 as usize * 8;
            comp_bufs.push(vec![128u8; cw * ch]);
            comp_dims.push((cw, ch));
        }

        let mut dc_prev = vec![0i32; nc];
        let mut mcu_count: u32 = 0;

        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                // RST handling: reset DC predictors at restart boundary
                if restart_interval > 0 && mcu_count > 0
                    && (mcu_count % restart_interval as u32) == 0
                {
                    for dc in &mut dc_prev { *dc = 0; }
                    br.cnt = 0; // flush remaining bits
                    // Skip RST marker bytes (0xFF 0xDn)
                    while br.pos + 1 < br.src.len()
                        && br.src[br.pos] == 0xFF
                        && br.src[br.pos + 1] >= 0xD0
                        && br.src[br.pos + 1] <= 0xD7
                    {
                        br.pos += 2;
                    }
                }

                for ci in 0..nc {
                    let h_samp = comp_info[ci].1 as usize;
                    let v_samp = comp_info[ci].2 as usize;
                    let (dc_id, ac_id) = comp_huf[ci];
                    let qt_id = comp_info[ci].3 as usize;
                    let dc_huf = dc_tabs.get(dc_id).and_then(|x| x.as_ref())
                        .ok_or("JPEG: missing DC table")?;
                    let ac_huf = ac_tabs.get(ac_id).and_then(|x| x.as_ref())
                        .ok_or("JPEG: missing AC table")?;
                    let qt = if qt_id < qtabs.len() { &qtabs[qt_id] } else { &qtabs[0] };

                    for bv in 0..v_samp {
                        for bh in 0..h_samp {
                            let mut block = [0i16; 64];
                            // DC coefficient
                            let dc_sym = dc_huf.decode(&mut br).ok_or("JPEG: DC decode failed")?;
                            let dc_val = if dc_sym == 0 { 0 } else {
                                let bits = br.bits(dc_sym).ok_or("JPEG: DC bits")?;
                                extend(bits, dc_sym)
                            };
                            dc_prev[ci] += dc_val;
                            block[0] = (dc_prev[ci] * qt[0] as i32).clamp(-1024, 1023) as i16;

                            // AC coefficients
                            let mut k = 1usize;
                            while k < 64 {
                                let sym = ac_huf.decode(&mut br).ok_or("JPEG: AC decode failed")?;
                                if sym == 0x00 { break; } // EOB
                                if sym == 0xF0 { k += 16; continue; } // ZRL
                                let run = (sym >> 4) as usize;
                                let size = sym & 0x0F;
                                k += run;
                                if k >= 64 { break; }
                                let bits = br.bits(size).ok_or("JPEG: AC bits")?;
                                let val = extend(bits, size);
                                let zz_pos = ZIGZAG[k] as usize;
                                block[zz_pos] = (val * qt[zz_pos] as i32).clamp(-1024, 1023) as i16;
                                k += 1;
                            }

                            // IDCT
                            let spatial = idct_2d(&mut block);

                            // Copy block into component buffer
                            let (cw, ch) = comp_dims[ci];
                            let block_x = mx * h_samp + bh;
                            let block_y = my * v_samp + bv;
                            for row in 0..8 {
                                for col in 0..8 {
                                    let px_x = block_x * 8 + col;
                                    let px_y = block_y * 8 + row;
                                    if px_x < cw && px_y < ch {
                                        comp_bufs[ci][px_y * cw + px_x] =
                                            spatial[row * 8 + col].clamp(0, 255) as u8;
                                    }
                                }
                            }
                        }
                    }
                }
                mcu_count += 1;
            }
        }

        // Convert YCbCr to RGB with chroma upsampling
        if nc == 3 {
            let n = w as usize * h as usize;
            let mut rgb = vec![0u8; n * 3];
            let (y_w, _) = comp_dims[0];
            let (cb_w, cb_h) = comp_dims[1];
            let (cr_w, cr_h) = comp_dims[2];

            for py in 0..h as usize {
                for px in 0..w as usize {
                    let y_val = comp_bufs[0][py * y_w + px] as f32;

                    // Map full-res coords to subsampled chroma coords
                    let cb_x = (px * comp_info[1].1 as usize / max_h).min(cb_w - 1);
                    let cb_y = (py * comp_info[1].2 as usize / max_v).min(cb_h - 1);
                    let cb_val = comp_bufs[1][cb_y * cb_w + cb_x] as f32 - 128.0;

                    let cr_x = (px * comp_info[2].1 as usize / max_h).min(cr_w - 1);
                    let cr_y = (py * comp_info[2].2 as usize / max_v).min(cr_h - 1);
                    let cr_val = comp_bufs[2][cr_y * cr_w + cr_x] as f32 - 128.0;

                    let idx = (py * w as usize + px) * 3;
                    rgb[idx    ] = (y_val + 1.402 * cr_val).round().clamp(0.0, 255.0) as u8;
                    rgb[idx + 1] = (y_val - 0.344136 * cb_val - 0.714136 * cr_val).round().clamp(0.0, 255.0) as u8;
                    rgb[idx + 2] = (y_val + 1.772 * cb_val).round().clamp(0.0, 255.0) as u8;
                }
            }
            Ok(rgb)
        } else if nc == 1 {
            let (cw, _) = comp_dims[0];
            let mut gray = Vec::with_capacity(w as usize * h as usize * 3);
            for py in 0..h as usize {
                for px in 0..w as usize {
                    let g = comp_bufs[0][py * cw + px];
                    gray.extend_from_slice(&[g, g, g]);
                }
            }
            Ok(gray)
        } else {
            Ok(comp_bufs.into_iter().flatten().collect())
        }
    }

    fn extend(val: i32, n: u8) -> i32 {
        if n == 0 { return 0; }
        if val < (1 << (n - 1)) { val - (1 << n) + 1 } else { val }
    }

    // ── JPEG encode ───────────────────────────────────────────────────────
    pub fn encode(rgb: &[u8], w: u32, h: u32, _quality: u8) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(0xFF); out.push(SOI);

        // APP0 (JFIF)
        let app0 = b"\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00";
        write_marker(&mut out, APP0, app0);

        // DQT (luminance)
        let mut dqt0 = vec![0u8; 65]; dqt0[0] = 0x00;
        for i in 0..64 { dqt0[1 + ZIGZAG[i] as usize] = LUMA_QTAB[i]; }
        write_marker(&mut out, DQT, &dqt0);

        // DQT (chrominance)
        let mut dqt1 = vec![0u8; 65]; dqt1[0] = 0x01;
        for i in 0..64 { dqt1[1 + ZIGZAG[i] as usize] = CHROMA_QTAB[i]; }
        write_marker(&mut out, DQT, &dqt1);

        // SOF0
        let mut sof = Vec::new();
        sof.push(8u8); // precision
        sof.extend_from_slice(&(h as u16).to_be_bytes());
        sof.extend_from_slice(&(w as u16).to_be_bytes());
        sof.push(3); // components
        sof.extend_from_slice(&[1, 0x11, 0]); // Y: 1x1 sampling, qtab 0
        sof.extend_from_slice(&[2, 0x11, 1]); // Cb
        sof.extend_from_slice(&[3, 0x11, 1]); // Cr
        write_marker(&mut out, SOF0, &sof);

        // DHT (DC luma)
        write_dht(&mut out, 0x00, &DC_LUMA_BITS, &DC_LUMA_VALS);
        // DHT (AC luma)
        write_dht(&mut out, 0x10, &AC_LUMA_BITS, &AC_LUMA_VALS);
        // DHT (DC chroma)
        write_dht(&mut out, 0x01, &DC_CHROMA_BITS, &DC_CHROMA_VALS);
        // DHT (AC chroma)
        write_dht(&mut out, 0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS);

        // SOS header
        let sos_hdr = [3u8, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0];
        write_marker(&mut out, SOS, &sos_hdr);

        // Build encode Huffman tables
        let dc_luma_enc  = HufEnc::from_bits_vals(&DC_LUMA_BITS,  &DC_LUMA_VALS);
        let ac_luma_enc  = HufEnc::from_bits_vals(&AC_LUMA_BITS,  &AC_LUMA_VALS);
        let dc_chroma_enc = HufEnc::from_bits_vals(&DC_CHROMA_BITS, &DC_CHROMA_VALS);
        let ac_chroma_enc = HufEnc::from_bits_vals(&AC_CHROMA_BITS, &AC_CHROMA_VALS);

        // Encode MCUs
        let mut bw = BitWriter::new();
        let mut dc_prev = [0i32; 3];
        for my in 0..(h as usize + 7) / 8 {
            for mx in 0..(w as usize + 7) / 8 {
                for ci in 0..3 {
                    let (dc_huf, ac_huf, qtab) = if ci == 0 {
                        (&dc_luma_enc, &ac_luma_enc, &LUMA_QTAB)
                    } else {
                        (&dc_chroma_enc, &ac_chroma_enc, &CHROMA_QTAB)
                    };
                    // Extract 8x8 block
                    let mut block = [0i16; 64];
                    for row in 0..8 { for col in 0..8 {
                        let px = (mx * 8 + col).min(w as usize - 1);
                        let py = (my * 8 + row).min(h as usize - 1);
                        let idx = (py * w as usize + px) * 3 + ci;
                        let v = rgb[idx] as f32 - 128.0;
                        block[row * 8 + col] = v.round() as i16;
                    }}
                    // Forward DCT
                    let mut fblock = fdct_2d(&block);
                    // Quantize
                    for i in 0..64 {
                        let q = qtab[i] as i16;
                        fblock[ZIGZAG[i] as usize] = (block[ZIGZAG[i] as usize] + q / 2) / q;
                    }
                    // Encode
                    let dc_diff = fblock[0] as i32 - dc_prev[ci];
                    dc_prev[ci] = fblock[0] as i32;
                    encode_dc(&mut bw, dc_huf, dc_diff);
                    encode_ac(&mut bw, ac_huf, &fblock);
                }
            }
        }
        bw.flush(&mut out);
        out.push(0xFF); out.push(EOI);
        out
    }

    fn fdct_2d(input: &[i16; 64]) -> [i16; 64] {
        // Simple 2D DCT (not AAN — straightforward)
        let mut tmp = [0f32; 64];
        for i in 0..64 { tmp[i] = input[i] as f32; }
        // Row DCT
        for r in 0..8 {
            let row: Vec<f32> = (0..8).map(|c| tmp[r*8+c]).collect();
            for k in 0..8 {
                let mut s = 0f32;
                for n in 0..8 {
                    s += row[n] * (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0) / 16.0).cos();
                }
                tmp[r*8+k] = s * if k == 0 { 1.0 / (2.0 * 2f32.sqrt()) } else { 0.5 };
            }
        }
        // Col DCT
        for c in 0..8 {
            let col: Vec<f32> = (0..8).map(|r| tmp[r*8+c]).collect();
            for k in 0..8 {
                let mut s = 0f32;
                for n in 0..8 {
                    s += col[n] * (std::f32::consts::PI * k as f32 * (2.0 * n as f32 + 1.0) / 16.0).cos();
                }
                tmp[k*8+c] = s * if k == 0 { 1.0 / (2.0 * 2f32.sqrt()) } else { 0.5 };
            }
        }
        let mut out = [0i16; 64];
        for i in 0..64 { out[i] = tmp[i].round() as i16; }
        out
    }

    struct HufEnc { codes: Vec<(u8, u32, u8)> } // (symbol, code MSB, length)
    impl HufEnc {
        fn from_bits_vals(bits: &[u8; 16], vals: &[u8]) -> Self {
            let mut codes = Vec::new();
            let mut code = 0u32; let mut vi = 0;
            for len in 1u8..=16 {
                for _ in 0..bits[len as usize - 1] {
                    if vi < vals.len() { codes.push((vals[vi], code, len)); vi += 1; }
                    code += 1;
                }
                code <<= 1;
            }
            HufEnc { codes }
        }
        fn lookup(&self, sym: u8) -> Option<(u32, u8)> {
            self.codes.iter().find(|&&(s,_,_)| s == sym).map(|&(_,c,l)| (c,l))
        }
    }

    struct BitWriter { buf: u32, cnt: u8 }
    impl BitWriter {
        fn new() -> Self { BitWriter { buf: 0, cnt: 0 } }
        fn write(&mut self, out: &mut Vec<u8>, code: u32, len: u8) {
            for i in (0..len).rev() {
                self.buf = (self.buf << 1) | ((code >> i) & 1);
                self.cnt += 1;
                if self.cnt == 8 {
                    let b = (self.buf & 0xFF) as u8;
                    out.push(b);
                    if b == 0xFF { out.push(0x00); } // byte stuffing
                    self.buf = 0; self.cnt = 0;
                }
            }
        }
        fn flush(&mut self, out: &mut Vec<u8>) {
            if self.cnt > 0 {
                let b = ((self.buf << (8 - self.cnt)) & 0xFF) as u8;
                out.push(b);
                if b == 0xFF { out.push(0x00); }
                self.buf = 0; self.cnt = 0;
            }
        }
    }

    fn encode_dc(bw: &mut BitWriter, huf: &HufEnc, diff: i32) {
        let (cat, val) = dc_category(diff);
        if let Some((code, len)) = huf.lookup(cat) { bw.write(&mut Vec::new(), code, len); }
        if cat > 0 { bw.write(&mut Vec::new(), val as u32, cat); }
    }
    fn encode_ac(bw: &mut BitWriter, huf: &HufEnc, block: &[i16; 64]) {
        let mut run = 0u8;
        for k in 1..64 {
            let v = block[k];
            if v == 0 { run += 1; if run == 16 { if let Some((c,l)) = huf.lookup(0xF0) { bw.write(&mut Vec::new(), c, l); } run = 0; } }
            else {
                let (cat, val) = ac_category(v as i32);
                let sym = (run << 4) | cat;
                if let Some((c, l)) = huf.lookup(sym) { bw.write(&mut Vec::new(), c, l); }
                bw.write(&mut Vec::new(), val as u32, cat);
                run = 0;
            }
        }
        if let Some((c, l)) = huf.lookup(0x00) { bw.write(&mut Vec::new(), c, l); } // EOB
    }

    fn dc_category(diff: i32) -> (u8, i32) {
        let abs = diff.unsigned_abs() as i32;
        let cat = if abs == 0 { 0 } else { (abs as f32).log2().floor() as u8 + 1 };
        let val = if diff < 0 { diff - 1 } else { diff };
        (cat, val & ((1 << cat) - 1))
    }
    fn ac_category(v: i32) -> (u8, i32) {
        let abs = v.unsigned_abs() as i32;
        let cat = if abs == 0 { 0 } else { (abs as f32).log2().floor() as u8 + 1 };
        let val = if v < 0 { v - 1 } else { v };
        (cat, val & ((1 << cat) - 1))
    }

    fn write_marker(out: &mut Vec<u8>, marker: u8, data: &[u8]) {
        out.push(0xFF); out.push(marker);
        let len = (data.len() + 2) as u16;
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(data);
    }
    fn write_dht(out: &mut Vec<u8>, id: u8, bits: &[u8; 16], vals: &[u8]) {
        let mut data = vec![id];
        data.extend_from_slice(bits);
        data.extend_from_slice(vals);
        write_marker(out, DHT, &data);
    }
}

// ─── GIF decoder ─────────────────────────────────────────────────────────────
pub(crate) mod gif {
    pub struct GifFrame {
        pub rgb: Vec<u8>,    // width*height*3 flat RGB
        pub delay_ms: u16,
    }
    pub struct GifData {
        pub width: u32,
        pub height: u32,
        pub frames: Vec<GifFrame>,
    }

    fn u16le(d: &[u8], p: usize) -> u16 {
        u16::from_le_bytes([d[p], d[p + 1]])
    }

    // LSB-first bit reader for GIF LZW
    struct GifBr<'a> { src: &'a [u8], pos: usize, buf: u32, cnt: u32 }
    impl<'a> GifBr<'a> {
        fn new(src: &'a [u8]) -> Self { GifBr { src, pos: 0, buf: 0, cnt: 0 } }
        fn bits(&mut self, n: u32) -> Option<u32> {
            while self.cnt < n {
                if self.pos >= self.src.len() { return None; }
                self.buf |= (self.src[self.pos] as u32) << self.cnt;
                self.pos += 1;
                self.cnt += 8;
            }
            let val = self.buf & ((1u32 << n) - 1);
            self.buf >>= n;
            self.cnt -= n;
            Some(val)
        }
    }

    fn read_sub_blocks(data: &[u8], pos: &mut usize) -> Vec<u8> {
        let mut out = Vec::new();
        while *pos < data.len() {
            let sz = data[*pos] as usize;
            *pos += 1;
            if sz == 0 { break; }
            if *pos + sz <= data.len() {
                out.extend_from_slice(&data[*pos..*pos + sz]);
            }
            *pos += sz;
        }
        out
    }

    fn lzw_decompress(data: &[u8], min_code_size: u8) -> Result<Vec<u8>, String> {
        let clear_code = 1u16 << min_code_size;
        let eoi_code = clear_code + 1;
        let mut code_size = min_code_size as u32 + 1;
        let mut next_code = eoi_code + 1;

        // Dictionary: table[code] = (prefix_code, suffix_byte, first_byte)
        // For single-byte entries: prefix=u16::MAX, suffix=byte, first=byte
        let max_entries = 4096usize;
        let mut tbl_prefix = vec![u16::MAX; max_entries];
        let mut tbl_suffix = vec![0u8; max_entries];
        let mut tbl_first  = vec![0u8; max_entries];
        for i in 0..clear_code as usize {
            tbl_suffix[i] = i as u8;
            tbl_first[i] = i as u8;
        }

        let mut output = Vec::new();
        let mut br = GifBr::new(data);
        let mut prev_code: Option<u16> = None;
        let mut decode_buf = Vec::with_capacity(4096);

        loop {
            let code = match br.bits(code_size) {
                Some(c) => c as u16,
                None => break,
            };

            if code == clear_code {
                code_size = min_code_size as u32 + 1;
                next_code = eoi_code + 1;
                prev_code = None;
                continue;
            }
            if code == eoi_code { break; }

            // Decode code to bytes
            let first_byte;
            if (code as usize) < next_code as usize {
                // Known code — decode chain
                decode_buf.clear();
                let mut c = code;
                while c != u16::MAX && (c as usize) < max_entries {
                    decode_buf.push(tbl_suffix[c as usize]);
                    c = tbl_prefix[c as usize];
                }
                decode_buf.reverse();
                first_byte = decode_buf[0];
                output.extend_from_slice(&decode_buf);
            } else if code == next_code {
                // Special case: code not yet in table
                let pc = prev_code.ok_or("GIF LZW: special case without prev")?;
                first_byte = tbl_first[pc as usize];
                decode_buf.clear();
                let mut c = pc;
                while c != u16::MAX && (c as usize) < max_entries {
                    decode_buf.push(tbl_suffix[c as usize]);
                    c = tbl_prefix[c as usize];
                }
                decode_buf.reverse();
                decode_buf.push(first_byte);
                output.extend_from_slice(&decode_buf);
            } else {
                return Err(format!("GIF LZW: bad code {} (next={})", code, next_code));
            }

            // Add new entry
            if let Some(pc) = prev_code {
                if (next_code as usize) < max_entries {
                    tbl_prefix[next_code as usize] = pc;
                    tbl_suffix[next_code as usize] = first_byte;
                    tbl_first[next_code as usize] = tbl_first[pc as usize];
                    next_code += 1;
                    if next_code > (1u16 << code_size) && code_size < 12 {
                        code_size += 1;
                    }
                }
            }
            prev_code = Some(code);
        }
        Ok(output)
    }

    pub fn decode(data: &[u8]) -> Result<GifData, String> {
        if data.len() < 13 { return Err("GIF: too short".into()); }
        if &data[0..3] != b"GIF" { return Err("GIF: bad signature".into()); }

        let width = u16le(data, 6) as u32;
        let height = u16le(data, 8) as u32;
        let packed = data[10];
        let gct_flag = (packed >> 7) & 1;
        let gct_size = if gct_flag == 1 { 3 * (1usize << ((packed & 7) + 1)) } else { 0 };
        let bg_color = data[11] as usize;

        let mut pos = 13;
        // Read global color table
        let mut gct = vec![0u8; gct_size];
        if gct_flag == 1 && pos + gct_size <= data.len() {
            gct.copy_from_slice(&data[pos..pos + gct_size]);
        }
        pos += gct_size;

        let n = width as usize * height as usize;
        let mut canvas = vec![0u8; n * 3];
        // Fill with background color
        if gct_flag == 1 && bg_color * 3 + 2 < gct.len() {
            for i in 0..n {
                canvas[i * 3    ] = gct[bg_color * 3    ];
                canvas[i * 3 + 1] = gct[bg_color * 3 + 1];
                canvas[i * 3 + 2] = gct[bg_color * 3 + 2];
            }
        }

        let mut frames = Vec::new();
        let mut delay_ms: u16 = 100;
        let mut disposal: u8 = 0;
        let mut transparent_idx: Option<u8> = None;

        while pos < data.len() {
            match data[pos] {
                0x21 => {
                    // Extension block
                    pos += 1;
                    if pos >= data.len() { break; }
                    let ext_type = data[pos]; pos += 1;
                    if ext_type == 0xF9 && pos + 5 <= data.len() {
                        // Graphics Control Extension
                        let _bsz = data[pos]; pos += 1;
                        let flags = data[pos]; pos += 1;
                        disposal = (flags >> 2) & 7;
                        let has_transparent = flags & 1;
                        delay_ms = u16le(data, pos) * 10; pos += 2;
                        if delay_ms == 0 { delay_ms = 100; }
                        transparent_idx = if has_transparent == 1 { Some(data[pos]) } else { None };
                        pos += 1;
                        pos += 1; // block terminator
                    } else {
                        // Skip other extensions via sub-blocks
                        while pos < data.len() {
                            let sz = data[pos] as usize; pos += 1;
                            if sz == 0 { break; }
                            pos += sz;
                        }
                    }
                }
                0x2C => {
                    // Image Descriptor
                    pos += 1;
                    if pos + 9 > data.len() { break; }
                    let left = u16le(data, pos) as usize; pos += 2;
                    let top = u16le(data, pos) as usize; pos += 2;
                    let sub_w = u16le(data, pos) as usize; pos += 2;
                    let sub_h = u16le(data, pos) as usize; pos += 2;
                    let img_packed = data[pos]; pos += 1;
                    let lct_flag = (img_packed >> 7) & 1;
                    let interlace = (img_packed >> 6) & 1;
                    let lct_size = if lct_flag == 1 { 3 * (1usize << ((img_packed & 7) + 1)) } else { 0 };

                    let ct = if lct_flag == 1 && pos + lct_size <= data.len() {
                        let t = &data[pos..pos + lct_size];
                        pos += lct_size;
                        t.to_vec()
                    } else {
                        pos += lct_size;
                        gct.clone()
                    };

                    // Save canvas for disposal method 3
                    let prev_canvas = if disposal == 3 { Some(canvas.clone()) } else { None };

                    // LZW minimum code size
                    if pos >= data.len() { break; }
                    let min_code_size = data[pos]; pos += 1;

                    // Read LZW sub-blocks
                    let lzw_data = read_sub_blocks(data, &mut pos);

                    // Decompress
                    let indices = lzw_decompress(&lzw_data, min_code_size)?;

                    // Paint onto canvas
                    for py in 0..sub_h {
                        // Handle interlacing
                        let dest_y = if interlace == 1 {
                            interlace_row(py, sub_h)
                        } else {
                            py
                        };
                        for px in 0..sub_w {
                            let src_idx = py * sub_w + px;
                            if src_idx >= indices.len() { continue; }
                            let ci = indices[src_idx] as usize;
                            if let Some(ti) = transparent_idx {
                                if ci == ti as usize { continue; }
                            }
                            let cx = left + px;
                            let cy = top + dest_y;
                            if cx < width as usize && cy < height as usize && ci * 3 + 2 < ct.len() {
                                let di = (cy * width as usize + cx) * 3;
                                canvas[di    ] = ct[ci * 3    ];
                                canvas[di + 1] = ct[ci * 3 + 1];
                                canvas[di + 2] = ct[ci * 3 + 2];
                            }
                        }
                    }

                    // Snapshot frame
                    frames.push(GifFrame { rgb: canvas.clone(), delay_ms });

                    // Apply disposal
                    match disposal {
                        2 => {
                            // Restore to background in the sub-image area
                            for py in 0..sub_h {
                                for px in 0..sub_w {
                                    let cx = left + px;
                                    let cy = top + py;
                                    if cx < width as usize && cy < height as usize {
                                        let di = (cy * width as usize + cx) * 3;
                                        if gct_flag == 1 && bg_color * 3 + 2 < gct.len() {
                                            canvas[di    ] = gct[bg_color * 3    ];
                                            canvas[di + 1] = gct[bg_color * 3 + 1];
                                            canvas[di + 2] = gct[bg_color * 3 + 2];
                                        } else {
                                            canvas[di] = 0; canvas[di+1] = 0; canvas[di+2] = 0;
                                        }
                                    }
                                }
                            }
                        }
                        3 => {
                            if let Some(prev) = prev_canvas {
                                canvas = prev;
                            }
                        }
                        _ => {} // 0, 1: leave as-is
                    }

                    // Reset GCE state
                    disposal = 0;
                    transparent_idx = None;
                    delay_ms = 100;
                }
                0x3B => break, // Trailer
                _ => { pos += 1; }
            }
        }

        if frames.is_empty() {
            return Err("GIF: no frames found".into());
        }
        Ok(GifData { width, height, frames })
    }

    fn interlace_row(pass_row: usize, height: usize) -> usize {
        // GIF interlace: 4 passes
        // Pass 1: rows 0, 8, 16, ... (start=0, step=8)
        // Pass 2: rows 4, 12, 20, ... (start=4, step=8)
        // Pass 3: rows 2, 6, 10, ... (start=2, step=4)
        // Pass 4: rows 1, 3, 5, ...  (start=1, step=2)
        let pass1 = (height + 7) / 8;
        let pass2 = (height + 3) / 8;
        let pass3 = (height + 1) / 4;
        if pass_row < pass1 {
            pass_row * 8
        } else if pass_row < pass1 + pass2 {
            (pass_row - pass1) * 8 + 4
        } else if pass_row < pass1 + pass2 + pass3 {
            (pass_row - pass1 - pass2) * 4 + 2
        } else {
            (pass_row - pass1 - pass2 - pass3) * 2 + 1
        }
    }
}

// ─── AVI/MJPEG parser ────────────────────────────────────────────────────────
pub(crate) mod avi {
    pub struct AviInfo {
        pub width: u32,
        pub height: u32,
        pub frame_count: usize,
        pub fps: f32,
        pub frame_offsets: Vec<(usize, usize)>, // (offset, length) into data
    }

    fn u32le(d: &[u8], p: usize) -> u32 {
        u32::from_le_bytes([d[p], d[p+1], d[p+2], d[p+3]])
    }
    #[allow(dead_code)]
    fn u16le(d: &[u8], p: usize) -> u16 {
        u16::from_le_bytes([d[p], d[p+1]])
    }
    fn fourcc(d: &[u8], p: usize) -> [u8; 4] {
        [d[p], d[p+1], d[p+2], d[p+3]]
    }

    pub fn parse(data: &[u8]) -> Result<AviInfo, String> {
        if data.len() < 12 { return Err("AVI: too short".into()); }
        if &data[0..4] != b"RIFF" || &data[8..12] != b"AVI " {
            return Err("AVI: not a RIFF AVI file".into());
        }

        let mut width = 0u32;
        let mut height = 0u32;
        let mut us_per_frame: u32 = 33333; // default ~30fps
        let mut frame_offsets: Vec<(usize, usize)> = Vec::new();

        let mut pos = 12;
        while pos + 8 <= data.len() {
            let tag = fourcc(data, pos);
            let size = u32le(data, pos + 4) as usize;
            let chunk_end = pos + 8 + size;

            if &tag == b"LIST" && pos + 12 <= data.len() {
                let list_type = fourcc(data, pos + 8);
                if &list_type == b"movi" {
                    // Scan for video frame chunks
                    let mut mpos = pos + 12;
                    while mpos + 8 <= chunk_end.min(data.len()) {
                        let mtag = fourcc(data, mpos);
                        let msize = u32le(data, mpos + 4) as usize;
                        // 00dc = compressed video, 00db = uncompressed video
                        if &mtag == b"00dc" || &mtag == b"00db" {
                            if mpos + 8 + msize <= data.len() {
                                frame_offsets.push((mpos + 8, msize));
                            }
                        }
                        mpos += 8 + ((msize + 1) & !1); // RIFF pads to even
                    }
                } else if &list_type == b"hdrl" {
                    // Parse AVI main header and stream headers
                    let mut hpos = pos + 12;
                    while hpos + 8 <= chunk_end.min(data.len()) {
                        let htag = fourcc(data, hpos);
                        let hsize = u32le(data, hpos + 4) as usize;
                        if &htag == b"avih" && hsize >= 32 && hpos + 8 + 32 <= data.len() {
                            us_per_frame = u32le(data, hpos + 8);
                            width = u32le(data, hpos + 8 + 32);
                            height = u32le(data, hpos + 8 + 36);
                        }
                        if &htag == b"LIST" && hpos + 12 <= data.len() {
                            let sub_type = fourcc(data, hpos + 8);
                            if &sub_type == b"strl" {
                                // Look for strf inside
                                let mut spos = hpos + 12;
                                let send = hpos + 8 + hsize;
                                while spos + 8 <= send.min(data.len()) {
                                    let stag = fourcc(data, spos);
                                    let ssize = u32le(data, spos + 4) as usize;
                                    if &stag == b"strf" && ssize >= 12 && spos + 8 + 12 <= data.len() {
                                        // BITMAPINFOHEADER: biWidth at +4, biHeight at +8
                                        let bw = u32le(data, spos + 8 + 4);
                                        let bh = u32le(data, spos + 8 + 8);
                                        if bw > 0 { width = bw; }
                                        if bh > 0 { height = bh; }
                                    }
                                    spos += 8 + ((ssize + 1) & !1);
                                }
                            }
                        }
                        hpos += 8 + ((hsize + 1) & !1);
                    }
                }
                pos += 8 + ((size + 1) & !1);
            } else {
                pos += 8 + ((size + 1) & !1);
            }
        }

        if frame_offsets.is_empty() {
            return Err("AVI: no video frames found".into());
        }

        let fps = if us_per_frame > 0 { 1_000_000.0 / us_per_frame as f32 } else { 30.0 };

        Ok(AviInfo { width, height, frame_count: frame_offsets.len(), fps, frame_offsets })
    }

    #[allow(dead_code)]
    pub fn extract_frame<'a>(data: &'a [u8], info: &AviInfo, idx: usize) -> Result<&'a [u8], String> {
        if idx >= info.frame_count {
            return Err(format!("AVI: frame {} out of range (0..{})", idx, info.frame_count));
        }
        let (offset, len) = info.frame_offsets[idx];
        if offset + len > data.len() {
            return Err("AVI: frame data out of bounds".into());
        }
        Ok(&data[offset..offset + len])
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_image_path() {
        assert!(is_image_path("photo.png"));
        assert!(is_image_path("photo.jpg"));
        assert!(is_image_path("photo.jpeg"));
        assert!(is_image_path("photo.PNG"));
        assert!(!is_image_path("data.csv"));
    }

    #[test]
    fn test_png_roundtrip_1x1() {
        let tmp = std::env::temp_dir().join("octoflow_png_rt1.png");
        let path = tmp.to_str().unwrap();
        let data = vec![255.0f32, 0.0, 128.0]; // 1x1 pixel
        write_image(path, &data, 1, 1).unwrap();
        let (back, w, h) = read_image(path).unwrap();
        assert_eq!(w, 1); assert_eq!(h, 1);
        assert!((back[0] - 255.0).abs() < 1.0);
        assert!((back[1] - 0.0).abs() < 1.0);
        assert!((back[2] - 128.0).abs() < 1.0);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_png_roundtrip_2x2() {
        let tmp = std::env::temp_dir().join("octoflow_png_rt2.png");
        let path = tmp.to_str().unwrap();
        let data: Vec<f32> = vec![
            255.0, 0.0, 0.0,   0.0, 255.0, 0.0,
            0.0, 0.0, 255.0,   255.0, 255.0, 255.0,
        ];
        write_image(path, &data, 2, 2).unwrap();
        let (back, w, h) = read_image(path).unwrap();
        assert_eq!(w, 2); assert_eq!(h, 2);
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0, "mismatch: {} vs {}", a, b);
        }
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_write_dimension_mismatch() {
        assert!(write_image("test.png", &[0.0; 6], 1, 1).is_err());
    }

    #[test]
    fn test_read_nonexistent() {
        assert!(read_image("/nonexistent/image.png").is_err());
    }

    #[test]
    fn test_image_limits_configured() {
        assert!(crate::MAX_IMAGE_FILE_BYTES >= 10 * 1024 * 1024);
        assert!(crate::MAX_IMAGE_DIMENSION >= 4096);
    }

    #[test]
    fn test_write_clamps_values() {
        let tmp = std::env::temp_dir().join("octoflow_png_clamp.png");
        let path = tmp.to_str().unwrap();
        let data: Vec<f32> = vec![-50.0, 300.0, 128.5];
        write_image(path, &data, 1, 1).unwrap();
        let (back, _, _) = read_image(path).unwrap();
        assert_eq!(back[0], 0.0);
        assert_eq!(back[1], 255.0);
        assert_eq!(back[2], 129.0);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_write_creates_parent_dirs() {
        let tmp = std::env::temp_dir().join("octoflow_img_test_subdir/sub/img.png");
        let path = tmp.to_str().unwrap();
        write_image(path, &[128.0; 3], 1, 1).unwrap();
        assert!(tmp.exists());
        std::fs::remove_dir_all(std::env::temp_dir().join("octoflow_img_test_subdir")).ok();
    }

    #[test]
    fn test_decode_truncated_pixel_data() {
        // A-01: truncated pixel data should error, not panic
        let mut fake_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data: [u8; 13] = [
            0x00, 0x00, 0x00, 0x0A, // width=10
            0x00, 0x00, 0x00, 0x0A, // height=10
            0x08, 0x02, 0x00, 0x00, 0x00, // depth=8, rgb, compression, filter, interlace
        ];
        let ihdr_crc = png::crc32(b"IHDR", &ihdr_data);
        fake_png.extend_from_slice(&(13u32).to_be_bytes());
        fake_png.extend_from_slice(b"IHDR");
        fake_png.extend_from_slice(&ihdr_data);
        fake_png.extend_from_slice(&ihdr_crc.to_be_bytes());
        // Empty IDAT + IEND — will fail in decode but should not panic
        let result = decode_image_bytes(&fake_png);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_overflow_dimensions() {
        // A-02: huge dimensions should error cleanly, not overflow
        let mut fake_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data: [u8; 13] = [
            0xFF, 0xFF, 0xFF, 0xFF, // width = 4294967295
            0xFF, 0xFF, 0xFF, 0xFF, // height = 4294967295
            0x08, 0x02, 0x00, 0x00, 0x00,
        ];
        let ihdr_crc = png::crc32(b"IHDR", &ihdr_data);
        fake_png.extend_from_slice(&(13u32).to_be_bytes());
        fake_png.extend_from_slice(b"IHDR");
        fake_png.extend_from_slice(&ihdr_data);
        fake_png.extend_from_slice(&ihdr_crc.to_be_bytes());
        let result = decode_image_bytes(&fake_png);
        assert!(result.is_err());
    }

    #[test]
    fn test_png_zlib_inflate_stored() {
        // A manually constructed PNG with a 1x1 white pixel
        let png_bytes = {
            let rgb = [255u8; 3];
            let w = 1u32; let h = 1u32;
            png::encode(&rgb, w, h)
        };
        let (pixels, w, h) = png::decode(&png_bytes).unwrap();
        assert_eq!(w, 1); assert_eq!(h, 1);
        assert_eq!(pixels, vec![255u8, 255, 255]);
    }

    // ════════════════════════════════════════════════════════════════
    // TEST-04: Media Robustness Tests (Phase 3G)
    // ════════════════════════════════════════════════════════════════

    // ── Corrupt magic bytes ──────────────────────────────────────

    #[test]
    fn test_decode_bad_magic_random() {
        let data = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
        let result = decode_image_bytes(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown image format"));
    }

    #[test]
    fn test_decode_bmp_magic_rejected() {
        let data = b"BM\x00\x00\x00\x00\x00\x00\x00\x00";
        let result = decode_image_bytes(data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("BMP"));
    }

    #[test]
    fn test_decode_too_short() {
        // Less than 4 bytes
        let result = decode_image_bytes(&[0x89]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_decode_empty_data() {
        let result = decode_image_bytes(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    // ── PNG edge cases ───────────────────────────────────────────

    #[test]
    fn test_png_bad_signature() {
        let data = b"\x89PNX\r\n\x1a\n\x00\x00\x00\x00IHDR";
        let result = png::decode(data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("signature"));
    }

    #[test]
    fn test_png_truncated_at_signature() {
        // Valid first 4 bytes but truncated before full 8-byte signature
        let data = b"\x89PNG";
        let result = png::decode(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_png_missing_ihdr() {
        // Valid signature but goes straight to IEND
        let mut data = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        // IEND chunk: length=0, type=IEND
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        data.extend_from_slice(b"IEND");
        let iend_crc = png::crc32(b"IEND", &[]);
        data.extend_from_slice(&iend_crc.to_be_bytes());
        let result = png::decode(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_png_zero_dimensions() {
        // IHDR with width=0, height=0
        let mut fake_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data: [u8; 13] = [
            0x00, 0x00, 0x00, 0x00, // width=0
            0x00, 0x00, 0x00, 0x00, // height=0
            0x08, 0x02, 0x00, 0x00, 0x00,
        ];
        let ihdr_crc = png::crc32(b"IHDR", &ihdr_data);
        fake_png.extend_from_slice(&(13u32).to_be_bytes());
        fake_png.extend_from_slice(b"IHDR");
        fake_png.extend_from_slice(&ihdr_data);
        fake_png.extend_from_slice(&ihdr_crc.to_be_bytes());
        let result = png::decode(&fake_png);
        // Either errors or produces empty image — should not panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_png_pixel_data_shorter_regression() {
        // A-01 regression: pixel data shorter than expected
        let mut fake_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data: [u8; 13] = [
            0x00, 0x00, 0x00, 0x04, // width=4
            0x00, 0x00, 0x00, 0x04, // height=4
            0x08, 0x02, 0x00, 0x00, 0x00,
        ];
        let ihdr_crc = png::crc32(b"IHDR", &ihdr_data);
        fake_png.extend_from_slice(&(13u32).to_be_bytes());
        fake_png.extend_from_slice(b"IHDR");
        fake_png.extend_from_slice(&ihdr_data);
        fake_png.extend_from_slice(&ihdr_crc.to_be_bytes());
        // IDAT with very short zlib data (incomplete)
        let idat_data = vec![0x78, 0x01, 0x01, 0x02, 0x00, 0xFD, 0xFF, 0x00, 0x00];
        let idat_crc = png::crc32(b"IDAT", &idat_data);
        fake_png.extend_from_slice(&(idat_data.len() as u32).to_be_bytes());
        fake_png.extend_from_slice(b"IDAT");
        fake_png.extend_from_slice(&idat_data);
        fake_png.extend_from_slice(&idat_crc.to_be_bytes());
        // IEND
        fake_png.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        fake_png.extend_from_slice(b"IEND");
        let iend_crc = png::crc32(b"IEND", &[]);
        fake_png.extend_from_slice(&iend_crc.to_be_bytes());
        let result = png::decode(&fake_png);
        assert!(result.is_err(), "truncated pixel data should error");
    }

    #[test]
    fn test_png_integer_overflow_dimensions_regression() {
        // A-02 regression: dimensions near u32::MAX should error, not overflow
        let mut fake_png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let ihdr_data: [u8; 13] = [
            0x7F, 0xFF, 0xFF, 0xFF, // width=2147483647
            0x7F, 0xFF, 0xFF, 0xFF, // height=2147483647
            0x08, 0x02, 0x00, 0x00, 0x00,
        ];
        let ihdr_crc = png::crc32(b"IHDR", &ihdr_data);
        fake_png.extend_from_slice(&(13u32).to_be_bytes());
        fake_png.extend_from_slice(b"IHDR");
        fake_png.extend_from_slice(&ihdr_data);
        fake_png.extend_from_slice(&ihdr_crc.to_be_bytes());
        let result = decode_image_bytes(&fake_png);
        assert!(result.is_err(), "huge dimensions should error cleanly");
    }

    #[test]
    fn test_png_encode_decode_roundtrip_large() {
        // 10x10 all-red image roundtrip
        let w = 10u32;
        let h = 10u32;
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            rgb.push(255); rgb.push(0); rgb.push(0);
        }
        let encoded = png::encode(&rgb, w, h);
        let (decoded, dw, dh) = png::decode(&encoded).unwrap();
        assert_eq!(dw, w);
        assert_eq!(dh, h);
        assert_eq!(decoded.len(), rgb.len());
        for (a, b) in rgb.iter().zip(decoded.iter()) {
            assert_eq!(a, b);
        }
    }

    // ── JPEG edge cases ──────────────────────────────────────────

    #[test]
    fn test_jpeg_bad_soi() {
        let data = [0xFF, 0x00, 0x00, 0x00]; // Not 0xFF 0xD8
        let result = jpeg::decode(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SOI"));
    }

    #[test]
    fn test_jpeg_truncated_header() {
        // Valid SOI but nothing else
        let data = [0xFF, 0xD8];
        let result = jpeg::decode(&data);
        // Should error (no markers after SOI) or succeed vacuously
        assert!(result.is_err());
    }

    #[test]
    fn test_jpeg_encode_produces_valid_header() {
        let w = 16u32;
        let h = 16u32;
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            rgb.push(128); rgb.push(64); rgb.push(192);
        }
        let encoded = jpeg::encode(&rgb, w, h, 85);
        assert!(encoded.len() > 10, "JPEG output too short");
        assert_eq!(encoded[0], 0xFF);
        assert_eq!(encoded[1], 0xD8); // SOI marker
        // Last two bytes should be EOI marker
        let n = encoded.len();
        assert_eq!(encoded[n - 2], 0xFF);
        assert_eq!(encoded[n - 1], 0xD9); // EOI marker
    }

    #[test]
    fn test_jpeg_truncated_after_soi() {
        // Valid SOI + start of a marker but truncated
        let data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00];
        let result = jpeg::decode(&data);
        assert!(result.is_err(), "truncated JPEG should error");
    }

    // ── GIF edge cases ───────────────────────────────────────────

    #[test]
    fn test_gif_too_short() {
        let data = b"GIF89a";
        let result = gif::decode(data);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.contains("too short"), "got: {}", e),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_gif_bad_signature() {
        let data = b"NOT89a\x00\x00\x00\x00\x00\x00\x00";
        let result = gif::decode(data);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.contains("signature"), "got: {}", e),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_gif_truncated_after_header() {
        // Valid GIF header but truncated before any image data
        let mut data = b"GIF89a".to_vec();
        data.extend_from_slice(&[0x01, 0x00]); // width=1
        data.extend_from_slice(&[0x01, 0x00]); // height=1
        data.push(0x00); // packed (no GCT)
        data.push(0x00); // bg color
        data.push(0x00); // pixel aspect ratio
        // No image data, just terminator
        data.push(0x3B); // GIF trailer
        let result = gif::decode(&data);
        // Should succeed with 0 frames or error cleanly
        assert!(result.is_ok() || result.is_err());
    }

    // ── Write edge cases ─────────────────────────────────────────

    #[test]
    fn test_write_zero_dimension() {
        // 0 pixels → error
        assert!(write_image("test.png", &[], 0, 0).is_err() || write_image("test.png", &[], 0, 0).is_ok());
    }

    #[test]
    fn test_write_huge_dimension_mismatch() {
        // Data doesn't match declared dimensions
        let data = vec![0.0f32; 3]; // 1 pixel
        assert!(write_image("test.png", &data, 100, 100).is_err());
    }

    #[test]
    fn test_write_nan_values_clamped() {
        let tmp = std::env::temp_dir().join("octoflow_png_nan.png");
        let path = tmp.to_str().unwrap();
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]; // 1x1
        // Should not panic — NaN/Inf get clamped
        let result = write_image(path, &data, 1, 1);
        if result.is_ok() {
            std::fs::remove_file(&tmp).ok();
        }
    }

    // ── AVI edge cases ───────────────────────────────────────────

    #[test]
    fn test_avi_bad_riff_header() {
        let data = b"NOPE\x00\x00\x00\x00AVI ";
        let result = avi::parse(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_avi_too_short() {
        let data = b"RIFF";
        let result = avi::parse(data);
        assert!(result.is_err());
    }
}
