use std::fmt;

use wincode::config::{Configuration, PREALLOCATION_SIZE_LIMIT_DISABLED};
use wincode::{ReadError, SchemaRead, SchemaWrite, WriteError};

pub const NATIVE_MAGIC: &[u8; 4] = b"FQTW";
pub const NATIVE_FORMAT_VERSION: u16 = 1;
pub const NATIVE_KIND_POINT: u8 = 1;
pub const NATIVE_KIND_RECT: u8 = 2;

const NATIVE_HEADER_LEN: usize = 8;
pub const DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES: usize = 64 * 1024 * 1024;

pub type NativeEncodingConfig =
    Configuration<true, { PREALLOCATION_SIZE_LIMIT_DISABLED }>;
pub type NativeDecodeConfig<const LIMIT: usize> = Configuration<true, LIMIT>;

const fn native_encoding_config() -> NativeEncodingConfig {
    Configuration::default().disable_preallocation_size_limit()
}

const fn native_decode_config<const LIMIT: usize>() -> NativeDecodeConfig<LIMIT> {
    Configuration::default().with_preallocation_size_limit::<LIMIT>()
}

#[derive(Debug)]
pub enum SerializationError {
    Encode(WriteError),
    Decode(ReadError),
    InvalidMagic,
    UnsupportedVersion(u16),
    UnexpectedKind { expected: u8, found: u8 },
    UnsupportedFlags(u8),
    TruncatedHeader,
    RuntimePreallocationLimitTooLarge(usize),
}

impl fmt::Display for SerializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationError::Encode(err) => write!(f, "encode failed: {err}"),
            SerializationError::Decode(err) => write!(f, "decode failed: {err}"),
            SerializationError::InvalidMagic => {
                write!(f, "invalid fastquadtree native serialization header")
            }
            SerializationError::UnsupportedVersion(version) => {
                write!(
                    f,
                    "unsupported fastquadtree native serialization version {version}"
                )
            }
            SerializationError::UnexpectedKind { expected, found } => {
                write!(
                    f,
                    "unexpected fastquadtree native tree kind {found}; expected {expected}"
                )
            }
            SerializationError::UnsupportedFlags(flags) => {
                write!(
                    f,
                    "unsupported fastquadtree native serialization flags {flags:#04x}"
                )
            }
            SerializationError::TruncatedHeader => {
                write!(f, "truncated fastquadtree native serialization header")
            }
            SerializationError::RuntimePreallocationLimitTooLarge(limit) => {
                write!(
                    f,
                    "preallocation limit {limit} is too large for runtime dispatch; use explicit unlimited decoding for trusted data"
                )
            }
        }
    }
}

impl std::error::Error for SerializationError {}

impl From<WriteError> for SerializationError {
    fn from(err: WriteError) -> Self {
        SerializationError::Encode(err)
    }
}

impl From<ReadError> for SerializationError {
    fn from(err: ReadError) -> Self {
        SerializationError::Decode(err)
    }
}

pub fn encode_native<T>(value: &T, kind: u8) -> Result<Vec<u8>, SerializationError>
where
    T: SchemaWrite<NativeEncodingConfig, Src = T> + ?Sized,
{
    let body = wincode::config::serialize(value, native_encoding_config())?;
    let mut out = Vec::with_capacity(NATIVE_HEADER_LEN + body.len());
    out.extend_from_slice(NATIVE_MAGIC);
    out.extend_from_slice(&NATIVE_FORMAT_VERSION.to_le_bytes());
    out.push(kind);
    out.push(0);
    out.extend_from_slice(&body);
    Ok(out)
}

pub fn decode_native<T>(bytes: &[u8], expected_kind: u8) -> Result<T, SerializationError>
where
    T: for<'de> SchemaRead<
        'de,
        NativeDecodeConfig<{ DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES }>,
        Dst = T,
    >,
{
    decode_native_with_preallocation_limit::<T, { DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES }>(
        bytes,
        expected_kind,
    )
}

pub fn decode_native_with_preallocation_limit<T, const LIMIT: usize>(
    bytes: &[u8],
    expected_kind: u8,
) -> Result<T, SerializationError>
where
    T: for<'de> SchemaRead<'de, NativeDecodeConfig<LIMIT>, Dst = T>,
{
    if bytes.len() < NATIVE_HEADER_LEN {
        return Err(SerializationError::TruncatedHeader);
    }
    if &bytes[..4] != NATIVE_MAGIC {
        return Err(SerializationError::InvalidMagic);
    }

    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != NATIVE_FORMAT_VERSION {
        return Err(SerializationError::UnsupportedVersion(version));
    }

    let found_kind = bytes[6];
    if found_kind != expected_kind {
        return Err(SerializationError::UnexpectedKind {
            expected: expected_kind,
            found: found_kind,
        });
    }

    let flags = bytes[7];
    if flags != 0 {
        return Err(SerializationError::UnsupportedFlags(flags));
    }

    Ok(wincode::config::deserialize_exact(
        &bytes[NATIVE_HEADER_LEN..],
        native_decode_config::<LIMIT>(),
    )?)
}

pub fn decode_native_unlimited<T>(
    bytes: &[u8],
    expected_kind: u8,
) -> Result<T, SerializationError>
where
    T: for<'de> SchemaRead<
        'de,
        NativeDecodeConfig<{ PREALLOCATION_SIZE_LIMIT_DISABLED }>,
        Dst = T,
    >,
{
    decode_native_with_preallocation_limit::<T, { PREALLOCATION_SIZE_LIMIT_DISABLED }>(
        bytes,
        expected_kind,
    )
}

pub fn decode_native_with_runtime_preallocation_limit<T>(
    bytes: &[u8],
    expected_kind: u8,
    preallocation_limit_bytes: Option<usize>,
    disable_preallocation_limit: bool,
) -> Result<T, SerializationError>
where
    T: for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<1024>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ 1024 * 1024 }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ 4 * 1024 * 1024 }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ 16 * 1024 * 1024 }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ 256 * 1024 * 1024 }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ 1024 * 1024 * 1024 }>,
            Dst = T,
        > + for<'de> SchemaRead<
            'de,
            NativeDecodeConfig<{ PREALLOCATION_SIZE_LIMIT_DISABLED }>,
            Dst = T,
        >,
{
    if disable_preallocation_limit {
        return decode_native_unlimited(bytes, expected_kind);
    }

    let limit = preallocation_limit_bytes.unwrap_or(DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES);
    if limit <= 1024 {
        decode_native_with_preallocation_limit::<T, 1024>(bytes, expected_kind)
    } else if limit <= 1024 * 1024 {
        decode_native_with_preallocation_limit::<T, { 1024 * 1024 }>(bytes, expected_kind)
    } else if limit <= 4 * 1024 * 1024 {
        decode_native_with_preallocation_limit::<T, { 4 * 1024 * 1024 }>(bytes, expected_kind)
    } else if limit <= 16 * 1024 * 1024 {
        decode_native_with_preallocation_limit::<T, { 16 * 1024 * 1024 }>(bytes, expected_kind)
    } else if limit <= DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES {
        decode_native_with_preallocation_limit::<T, { DEFAULT_NATIVE_PREALLOCATION_LIMIT_BYTES }>(
            bytes,
            expected_kind,
        )
    } else if limit <= 256 * 1024 * 1024 {
        decode_native_with_preallocation_limit::<T, { 256 * 1024 * 1024 }>(bytes, expected_kind)
    } else if limit <= 1024 * 1024 * 1024 {
        decode_native_with_preallocation_limit::<T, { 1024 * 1024 * 1024 }>(bytes, expected_kind)
    } else {
        Err(SerializationError::RuntimePreallocationLimitTooLarge(limit))
    }
}
