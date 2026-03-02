//! .octo binary storage format for GPU-friendly columnar data.
//!
//! Binary format designed for zero-copy GPU buffer upload:
//! - Magic "OCTO" header with version, column count, row count
//! - Column descriptors with name, type, encoding, offset
//! - Column data aligned for direct DMA to GPU VRAM
//! - Encoding: raw f32 (zero-parse) or delta-encoded f32
//! - Footer with per-column statistics (min, max)
//!
//! **10-50x faster than CSV** — binary f32 requires no parsing.

use crate::CliError;
use std::io::{Cursor, Read};

const MAGIC: [u8; 4] = *b"OCTO";
const VERSION: u16 = 1;

/// Data type: f32 (the only type for now).
const DTYPE_F32: u8 = 0x01;

/// Raw f32 little-endian — zero-parse, GPU-ready.
const ENCODING_RAW: u8 = 0x00;
/// Delta-encoded f32 — base + deltas, reconstruct via prefix sum.
const ENCODING_DELTA: u8 = 0x01;

/// Check if a path has the `.octo` extension.
pub fn is_octo_path(path: &str) -> bool {
    path.to_lowercase().ends_with(".octo")
}

/// Column descriptor for the .octo format.
#[derive(Debug, Clone)]
pub struct ColumnDescriptor {
    pub name: String,
    pub dtype: u8,
    pub compression: u8,
    pub data_offset: u64,
    pub data_size: u64,
}

/// Per-column statistics stored in the footer.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub min: f32,
    pub max: f32,
}

/// Metadata about a .octo file.
#[derive(Debug)]
pub struct OctoInfo {
    pub version: u16,
    pub row_count: usize,
    pub columns: Vec<ColumnDescriptor>,
    pub file_size: usize,
}

/// Read a .octo file, returning the first column as f32 data.
pub fn read_octo(path: &str) -> Result<Vec<f32>, CliError> {
    read_octo_column(path, 0)
}

/// Read a specific column from a .octo file by index.
pub fn read_octo_column(path: &str, column_index: usize) -> Result<Vec<f32>, CliError> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    if metadata.len() > crate::MAX_OCTO_FILE_BYTES {
        return Err(CliError::Security(format!(
            "{}: file size {} bytes exceeds .octo limit ({} MB)",
            path,
            metadata.len(),
            crate::MAX_OCTO_FILE_BYTES / (1024 * 1024)
        )));
    }

    let bytes = std::fs::read(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;

    let (row_count, descriptors) = read_header(&bytes, path)?;

    if column_index >= descriptors.len() {
        return Err(CliError::Io(format!(
            "{}: column index {} out of range (file has {} columns)",
            path, column_index, descriptors.len()
        )));
    }

    let desc = &descriptors[column_index];
    if desc.dtype != DTYPE_F32 {
        return Err(CliError::Io(format!(
            "{}: unsupported data type 0x{:02x}",
            path, desc.dtype
        )));
    }

    let data_start = desc.data_offset as usize;
    let data_end = data_start + desc.data_size as usize;
    if data_end > bytes.len() {
        return Err(CliError::Io(format!(
            "{}: column data extends beyond file",
            path
        )));
    }
    let col_bytes = &bytes[data_start..data_end];

    match desc.compression {
        ENCODING_RAW => decode_raw(col_bytes, row_count),
        ENCODING_DELTA => decode_delta(col_bytes, row_count),
        _ => Err(CliError::Io(format!(
            "{}: unsupported encoding 0x{:02x}",
            path, desc.compression
        ))),
    }
}

/// Write a single column of f32 data using raw encoding.
pub fn write_octo(path: &str, data: &[f32]) -> Result<(), CliError> {
    write_octo_columns(path, &[("data", data, ENCODING_RAW)])
}

/// Write a single column of f32 data using delta encoding.
pub fn write_octo_delta(path: &str, data: &[f32]) -> Result<(), CliError> {
    write_octo_columns(path, &[("data", data, ENCODING_DELTA)])
}

/// Write multiple named columns to a .octo file.
///
/// Each entry: `(name, data, encoding)` where encoding is `0x00` (raw) or `0x01` (delta).
pub fn write_octo_columns(
    path: &str,
    columns: &[(&str, &[f32], u8)],
) -> Result<(), CliError> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
        }
    }

    if columns.is_empty() {
        return Err(CliError::Io(
            "cannot write .octo file with zero columns".into(),
        ));
    }

    let row_count = columns[0].1.len();
    for (name, data, _) in columns {
        if data.len() != row_count {
            return Err(CliError::Io(format!(
                "column '{}' has {} rows, expected {}",
                name,
                data.len(),
                row_count
            )));
        }
    }

    let mut buf = Vec::new();

    // ── Header (18 bytes) ───────────────────────────────────────────
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&(columns.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(row_count as u64).to_le_bytes());

    // Calculate descriptor section size to find data start offset
    let mut descriptor_size = 0usize;
    for (name, _, _) in columns {
        // name_len(2) + name + dtype(1) + compression(1) + offset(8) + size(8)
        descriptor_size += 2 + name.len() + 1 + 1 + 8 + 8;
    }

    // Data starts after header + descriptors, aligned to 16 bytes
    let header_and_desc_end = 18 + descriptor_size;
    let data_start = (header_and_desc_end + 15) & !15;

    // Encode all columns and compute offsets
    let mut encoded_columns: Vec<Vec<u8>> = Vec::new();
    let mut offsets: Vec<(usize, usize)> = Vec::new(); // (offset, size)
    let mut current_offset = data_start;

    for (_, data, encoding) in columns {
        let encoded = match *encoding {
            ENCODING_RAW => encode_raw(data),
            ENCODING_DELTA => encode_delta(data),
            _ => {
                return Err(CliError::Io(format!(
                    "unsupported encoding 0x{:02x}",
                    encoding
                )))
            }
        };
        offsets.push((current_offset, encoded.len()));
        current_offset += encoded.len();
        current_offset = (current_offset + 15) & !15; // align next column
        encoded_columns.push(encoded);
    }

    // ── Column descriptors ──────────────────────────────────────────
    for (i, (name, _, encoding)) in columns.iter().enumerate() {
        let (offset, size) = offsets[i];
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.push(DTYPE_F32);
        buf.push(*encoding);
        buf.extend_from_slice(&(offset as u64).to_le_bytes());
        buf.extend_from_slice(&(size as u64).to_le_bytes());
    }

    // ── Padding to data alignment ───────────────────────────────────
    buf.resize(data_start, 0);

    // ── Column data ─────────────────────────────────────────────────
    for (i, encoded) in encoded_columns.iter().enumerate() {
        let expected_offset = offsets[i].0;
        buf.resize(expected_offset, 0);
        buf.extend_from_slice(encoded);
    }

    // ── Footer: per-column statistics ───────────────────────────────
    for (_, data, _) in columns {
        let stats = compute_stats(data);
        buf.extend_from_slice(&stats.min.to_le_bytes());
        buf.extend_from_slice(&stats.max.to_le_bytes());
    }
    buf.extend_from_slice(&MAGIC); // footer sentinel

    std::fs::write(path, &buf)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;

    Ok(())
}

/// Get metadata about a .octo file without reading all column data.
pub fn octo_info(path: &str) -> Result<OctoInfo, CliError> {
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    let (row_count, columns) = read_header(&bytes, path)?;
    Ok(OctoInfo {
        version: VERSION,
        row_count,
        columns,
        file_size: bytes.len(),
    })
}

// ── Header parsing (shared by read_octo_column and octo_info) ──────

fn read_header(bytes: &[u8], path: &str) -> Result<(usize, Vec<ColumnDescriptor>), CliError> {
    let mut cursor = Cursor::new(bytes);

    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
    if magic != MAGIC {
        return Err(CliError::Io(format!(
            "{}: not a valid .octo file (bad magic)",
            path
        )));
    }

    let version = read_u16_le(&mut cursor)?;
    if version != VERSION {
        return Err(CliError::Io(format!(
            "{}: unsupported .octo version {}",
            path, version
        )));
    }

    let column_count = read_u32_le(&mut cursor)? as usize;
    let row_count = read_u64_le(&mut cursor)? as usize;

    let mut descriptors = Vec::with_capacity(column_count);
    for _ in 0..column_count {
        let name_len = read_u16_le(&mut cursor)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        cursor
            .read_exact(&mut name_bytes)
            .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;
        let name = String::from_utf8(name_bytes)
            .map_err(|_| CliError::Io(format!("{}: invalid UTF-8 column name", path)))?;

        let mut meta = [0u8; 18]; // dtype(1) + compression(1) + offset(8) + size(8)
        cursor
            .read_exact(&mut meta)
            .map_err(|e| CliError::Io(format!("{}: {}", path, e)))?;

        descriptors.push(ColumnDescriptor {
            name,
            dtype: meta[0],
            compression: meta[1],
            data_offset: u64::from_le_bytes(meta[2..10].try_into().unwrap()),
            data_size: u64::from_le_bytes(meta[10..18].try_into().unwrap()),
        });
    }

    Ok((row_count, descriptors))
}

// ── Encoding / Decoding ─────────────────────────────────────────────

fn encode_raw(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &val in data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

fn decode_raw(bytes: &[u8], row_count: usize) -> Result<Vec<f32>, CliError> {
    if bytes.len() < row_count * 4 {
        return Err(CliError::Io("raw column data too short".into()));
    }
    let mut data = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let o = i * 4;
        let val = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        data.push(val);
    }
    Ok(data)
}

fn encode_delta(data: &[f32]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    bytes.extend_from_slice(&data[0].to_le_bytes()); // base value
    for i in 1..data.len() {
        let delta = data[i] - data[i - 1];
        bytes.extend_from_slice(&delta.to_le_bytes());
    }
    bytes
}

fn decode_delta(bytes: &[u8], row_count: usize) -> Result<Vec<f32>, CliError> {
    if row_count == 0 {
        return Ok(Vec::new());
    }
    if bytes.len() < row_count * 4 {
        return Err(CliError::Io("delta column data too short".into()));
    }
    let mut data = Vec::with_capacity(row_count);
    let base = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    data.push(base);
    for i in 1..row_count {
        let o = i * 4;
        let delta = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        data.push(data[i - 1] + delta);
    }
    Ok(data)
}

fn compute_stats(data: &[f32]) -> ColumnStats {
    if data.is_empty() {
        return ColumnStats { min: 0.0, max: 0.0 };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &val in data {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }
    ColumnStats { min, max }
}

// ── Binary read helpers ─────────────────────────────────────────────

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, CliError> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)
        .map_err(|e| CliError::Io(format!("read error: {}", e)))?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32_le<R: Read>(r: &mut R) -> Result<u32, CliError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .map_err(|e| CliError::Io(format!("read error: {}", e)))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le<R: Read>(r: &mut R) -> Result<u64, CliError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)
        .map_err(|e| CliError::Io(format!("read error: {}", e)))?;
    Ok(u64::from_le_bytes(buf))
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> String {
        let dir = std::env::temp_dir().join("octoflow_octo_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn test_is_octo_path() {
        assert!(is_octo_path("data.octo"));
        assert!(is_octo_path("path/to/DATA.OCTO"));
        assert!(is_octo_path("file.Octo"));
        assert!(!is_octo_path("data.csv"));
        assert!(!is_octo_path("data.octopus"));
    }

    #[test]
    fn test_write_read_raw_roundtrip() {
        let path = temp_path("raw_roundtrip.octo");
        let data = vec![1.0, 2.5, 3.14, -0.5, 100.0, 0.0, -999.99];

        write_octo(&path, &data).unwrap();
        let read_back = read_octo(&path).unwrap();

        assert_eq!(data.len(), read_back.len());
        for (a, b) in data.iter().zip(read_back.iter()) {
            assert_eq!(*a, *b, "raw roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_write_read_delta_roundtrip() {
        let path = temp_path("delta_roundtrip.octo");
        // Simulated time-series prices (sequential, small deltas)
        let data = vec![2650.0, 2650.5, 2651.2, 2650.8, 2651.5, 2652.0];

        write_octo_delta(&path, &data).unwrap();
        let read_back = read_octo(&path).unwrap();

        assert_eq!(data.len(), read_back.len());
        for (a, b) in data.iter().zip(read_back.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "delta roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_multi_column_roundtrip() {
        let path = temp_path("multi_col.octo");
        let open = vec![100.0, 101.0, 102.0, 103.0];
        let high = vec![101.5, 102.5, 103.5, 104.5];
        let low = vec![99.5, 100.5, 101.5, 102.5];
        let close = vec![101.0, 102.0, 103.0, 104.0];

        write_octo_columns(
            &path,
            &[
                ("open", &open, ENCODING_RAW),
                ("high", &high, ENCODING_RAW),
                ("low", &low, ENCODING_RAW),
                ("close", &close, ENCODING_DELTA),
            ],
        )
        .unwrap();

        // Read each column
        for (i, expected) in [&open, &high, &low, &close].iter().enumerate() {
            let read_back = read_octo_column(&path, i).unwrap();
            assert_eq!(expected.len(), read_back.len(), "column {} length mismatch", i);
            for (a, b) in expected.iter().zip(read_back.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "column {} mismatch: {} vs {}",
                    i,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_octo_info() {
        let path = temp_path("info_test.octo");
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        write_octo(&path, &data).unwrap();

        let info = octo_info(&path).unwrap();
        assert_eq!(info.version, 1);
        assert_eq!(info.row_count, 5);
        assert_eq!(info.columns.len(), 1);
        assert_eq!(info.columns[0].name, "data");
        assert_eq!(info.columns[0].compression, ENCODING_RAW);
    }

    #[test]
    fn test_read_nonexistent() {
        let result = read_octo("/nonexistent/path/file.octo");
        assert!(result.is_err());
    }

    #[test]
    fn test_bad_magic() {
        let path = temp_path("bad_magic.octo");
        std::fs::write(&path, b"NOPE_not_octo").unwrap();
        let result = read_octo(&path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("bad magic"), "got: {}", msg);
    }

    #[test]
    fn test_column_index_out_of_range() {
        let path = temp_path("col_range.octo");
        write_octo(&path, &[1.0, 2.0]).unwrap();
        let result = read_octo_column(&path, 5);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("out of range"), "got: {}", msg);
    }

    #[test]
    fn test_empty_data() {
        let path = temp_path("empty.octo");
        write_octo(&path, &[]).unwrap();
        let read_back = read_octo(&path).unwrap();
        assert!(read_back.is_empty());
    }

    #[test]
    fn test_large_dataset() {
        let path = temp_path("large.octo");
        let data: Vec<f32> = (0..100_000).map(|i| i as f32 * 0.01).collect();
        write_octo(&path, &data).unwrap();
        let read_back = read_octo(&path).unwrap();
        assert_eq!(data.len(), read_back.len());
        // Verify first, middle, and last
        assert_eq!(data[0], read_back[0]);
        assert_eq!(data[50_000], read_back[50_000]);
        assert_eq!(data[99_999], read_back[99_999]);
    }

    #[test]
    fn test_octo_limits_configured() {
        assert!(crate::MAX_OCTO_FILE_BYTES > 0);
        assert!(crate::MAX_OCTO_FILE_BYTES >= 100 * 1024 * 1024);
    }
}
