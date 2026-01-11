/**
 * ZIP Export Utilities for Campaign Analytics
 *
 * Provides utility functions for creating ZIP archives from multiple files
 * without external dependencies. Uses the native CompressionStream API
 * with fallback to raw store (no compression) for maximum compatibility.
 *
 * @module lib/utils/zip-export
 */

// =============================================================================
// Types
// =============================================================================

/**
 * File entry for ZIP archive.
 */
export interface ZipFileEntry {
  /** Filename including path (e.g., "charts/success-rate.png") */
  filename: string;
  /** File content as Blob, ArrayBuffer, or Uint8Array */
  content: Blob | ArrayBuffer | Uint8Array;
  /** Optional MIME type for the file */
  mimeType?: string;
  /** Optional last modified date */
  lastModified?: Date;
}

/**
 * Result of ZIP generation.
 */
export interface ZipGenerateResult {
  /** Whether generation was successful */
  success: boolean;
  /** Generated ZIP blob (if successful) */
  blob?: Blob;
  /** Total files included */
  fileCount?: number;
  /** Total uncompressed size in bytes */
  totalSize?: number;
  /** Error message (if failed) */
  error?: string;
}

/**
 * Options for ZIP generation.
 */
export interface ZipGenerateOptions {
  /** Comment for the ZIP archive */
  comment?: string;
  /** Whether to compress files (default: true) */
  compress?: boolean;
  /** Compression level (1-9, default: 6) */
  compressionLevel?: number;
}

// =============================================================================
// Constants
// =============================================================================

/** ZIP local file header signature */
const LOCAL_FILE_HEADER_SIG = 0x04034b50;

/** ZIP central directory header signature */
const CENTRAL_DIR_HEADER_SIG = 0x02014b50;

/** ZIP end of central directory signature */
const END_OF_CENTRAL_DIR_SIG = 0x06054b50;

/** ZIP version made by (2.0 for deflate) */
const VERSION_MADE_BY = 0x0014;

/** ZIP version needed to extract */
const VERSION_NEEDED = 0x0014;

/** No compression */
const COMPRESSION_STORE = 0;

/** DEFLATE compression */
const COMPRESSION_DEFLATE = 8;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Converts a Date to DOS date/time format.
 */
function dateToDos(date: Date): { date: number; time: number } {
  const dosTime =
    ((date.getHours() & 0x1f) << 11) |
    ((date.getMinutes() & 0x3f) << 5) |
    ((date.getSeconds() >> 1) & 0x1f);

  const dosDate =
    (((date.getFullYear() - 1980) & 0x7f) << 9) |
    (((date.getMonth() + 1) & 0x0f) << 5) |
    (date.getDate() & 0x1f);

  return { date: dosDate, time: dosTime };
}

/**
 * Calculates CRC-32 checksum for a Uint8Array.
 */
function crc32(data: Uint8Array): number {
  // Pre-computed CRC-32 table
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }

  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Converts various content types to Uint8Array.
 */
async function contentToBytes(
  content: Blob | ArrayBuffer | Uint8Array
): Promise<Uint8Array> {
  if (content instanceof Uint8Array) {
    return content;
  }

  if (content instanceof ArrayBuffer) {
    return new Uint8Array(content);
  }

  // It's a Blob
  const buffer = await content.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Encodes a string to UTF-8 bytes.
 */
function encodeUTF8(str: string): Uint8Array {
  return new TextEncoder().encode(str);
}

/**
 * Writes a 16-bit little-endian value to a DataView.
 */
function writeUint16LE(view: DataView, offset: number, value: number): void {
  view.setUint16(offset, value, true);
}

/**
 * Writes a 32-bit little-endian value to a DataView.
 */
function writeUint32LE(view: DataView, offset: number, value: number): void {
  view.setUint32(offset, value, true);
}

/**
 * Compresses data using DEFLATE via CompressionStream.
 * Falls back to uncompressed if not supported.
 */
async function compressData(
  data: Uint8Array
): Promise<{ compressed: Uint8Array; method: number }> {
  // Check if CompressionStream is available
  if (typeof CompressionStream === "undefined") {
    return { compressed: data, method: COMPRESSION_STORE };
  }

  try {
    const stream = new CompressionStream("deflate-raw");
    const writer = stream.writable.getWriter();
    const reader = stream.readable.getReader();

    // Write data
    writer.write(data);
    writer.close();

    // Read compressed data
    const chunks: Uint8Array[] = [];
    let totalLength = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      totalLength += value.length;
    }

    // Combine chunks
    const compressed = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      compressed.set(chunk, offset);
      offset += chunk.length;
    }

    // Only use compression if it actually reduces size
    if (compressed.length < data.length) {
      return { compressed, method: COMPRESSION_DEFLATE };
    }

    return { compressed: data, method: COMPRESSION_STORE };
  } catch {
    // Fallback to no compression
    return { compressed: data, method: COMPRESSION_STORE };
  }
}

// =============================================================================
// ZIP File Structure
// =============================================================================

interface LocalFileHeader {
  signature: number;
  versionNeeded: number;
  flags: number;
  compression: number;
  modTime: number;
  modDate: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  filenameLength: number;
  extraFieldLength: number;
  filename: Uint8Array;
  extraField: Uint8Array;
  data: Uint8Array;
}

interface CentralDirEntry {
  signature: number;
  versionMadeBy: number;
  versionNeeded: number;
  flags: number;
  compression: number;
  modTime: number;
  modDate: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  filenameLength: number;
  extraFieldLength: number;
  commentLength: number;
  diskNumberStart: number;
  internalAttrs: number;
  externalAttrs: number;
  localHeaderOffset: number;
  filename: Uint8Array;
  extraField: Uint8Array;
  comment: Uint8Array;
}

interface EndOfCentralDir {
  signature: number;
  diskNumber: number;
  diskWithCentralDir: number;
  entriesOnDisk: number;
  totalEntries: number;
  centralDirSize: number;
  centralDirOffset: number;
  commentLength: number;
  comment: Uint8Array;
}

/**
 * Creates a local file header structure.
 */
function createLocalFileHeader(
  filename: Uint8Array,
  data: Uint8Array,
  crc: number,
  compression: number,
  modDate: Date
): LocalFileHeader {
  const dos = dateToDos(modDate);

  return {
    signature: LOCAL_FILE_HEADER_SIG,
    versionNeeded: VERSION_NEEDED,
    flags: 0x0800, // UTF-8 filenames
    compression,
    modTime: dos.time,
    modDate: dos.date,
    crc32: crc,
    compressedSize: data.length,
    uncompressedSize: data.length, // Will be updated for compression
    filenameLength: filename.length,
    extraFieldLength: 0,
    filename,
    extraField: new Uint8Array(0),
    data,
  };
}

/**
 * Creates a central directory entry.
 */
function createCentralDirEntry(
  filename: Uint8Array,
  crc: number,
  compressedSize: number,
  uncompressedSize: number,
  compression: number,
  modDate: Date,
  localHeaderOffset: number
): CentralDirEntry {
  const dos = dateToDos(modDate);

  return {
    signature: CENTRAL_DIR_HEADER_SIG,
    versionMadeBy: VERSION_MADE_BY,
    versionNeeded: VERSION_NEEDED,
    flags: 0x0800, // UTF-8 filenames
    compression,
    modTime: dos.time,
    modDate: dos.date,
    crc32: crc,
    compressedSize,
    uncompressedSize,
    filenameLength: filename.length,
    extraFieldLength: 0,
    commentLength: 0,
    diskNumberStart: 0,
    internalAttrs: 0,
    externalAttrs: 0,
    localHeaderOffset,
    filename,
    extraField: new Uint8Array(0),
    comment: new Uint8Array(0),
  };
}

/**
 * Creates end of central directory record.
 */
function createEndOfCentralDir(
  numEntries: number,
  centralDirSize: number,
  centralDirOffset: number,
  comment: Uint8Array
): EndOfCentralDir {
  return {
    signature: END_OF_CENTRAL_DIR_SIG,
    diskNumber: 0,
    diskWithCentralDir: 0,
    entriesOnDisk: numEntries,
    totalEntries: numEntries,
    centralDirSize,
    centralDirOffset,
    commentLength: comment.length,
    comment,
  };
}

/**
 * Serializes a local file header to bytes.
 */
function serializeLocalFileHeader(header: LocalFileHeader): Uint8Array {
  const size = 30 + header.filenameLength + header.extraFieldLength;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);

  writeUint32LE(view, 0, header.signature);
  writeUint16LE(view, 4, header.versionNeeded);
  writeUint16LE(view, 6, header.flags);
  writeUint16LE(view, 8, header.compression);
  writeUint16LE(view, 10, header.modTime);
  writeUint16LE(view, 12, header.modDate);
  writeUint32LE(view, 14, header.crc32);
  writeUint32LE(view, 18, header.compressedSize);
  writeUint32LE(view, 22, header.uncompressedSize);
  writeUint16LE(view, 26, header.filenameLength);
  writeUint16LE(view, 28, header.extraFieldLength);

  const bytes = new Uint8Array(buffer);
  bytes.set(header.filename, 30);

  return bytes;
}

/**
 * Serializes a central directory entry to bytes.
 */
function serializeCentralDirEntry(entry: CentralDirEntry): Uint8Array {
  const size =
    46 + entry.filenameLength + entry.extraFieldLength + entry.commentLength;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);

  writeUint32LE(view, 0, entry.signature);
  writeUint16LE(view, 4, entry.versionMadeBy);
  writeUint16LE(view, 6, entry.versionNeeded);
  writeUint16LE(view, 8, entry.flags);
  writeUint16LE(view, 10, entry.compression);
  writeUint16LE(view, 12, entry.modTime);
  writeUint16LE(view, 14, entry.modDate);
  writeUint32LE(view, 18, entry.crc32);
  writeUint32LE(view, 22, entry.compressedSize);
  writeUint32LE(view, 26, entry.uncompressedSize);
  writeUint16LE(view, 30, entry.filenameLength);
  writeUint16LE(view, 32, entry.extraFieldLength);
  writeUint16LE(view, 34, entry.commentLength);
  writeUint16LE(view, 36, entry.diskNumberStart);
  writeUint16LE(view, 38, entry.internalAttrs);
  writeUint32LE(view, 40, entry.externalAttrs);
  writeUint32LE(view, 44, entry.localHeaderOffset);

  const bytes = new Uint8Array(buffer);
  bytes.set(entry.filename, 46);

  return bytes;
}

/**
 * Serializes end of central directory to bytes.
 */
function serializeEndOfCentralDir(eocd: EndOfCentralDir): Uint8Array {
  const size = 22 + eocd.commentLength;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);

  writeUint32LE(view, 0, eocd.signature);
  writeUint16LE(view, 4, eocd.diskNumber);
  writeUint16LE(view, 6, eocd.diskWithCentralDir);
  writeUint16LE(view, 8, eocd.entriesOnDisk);
  writeUint16LE(view, 10, eocd.totalEntries);
  writeUint32LE(view, 12, eocd.centralDirSize);
  writeUint32LE(view, 16, eocd.centralDirOffset);
  writeUint16LE(view, 20, eocd.commentLength);

  const bytes = new Uint8Array(buffer);
  if (eocd.commentLength > 0) {
    bytes.set(eocd.comment, 22);
  }

  return bytes;
}

// =============================================================================
// Main Export Functions
// =============================================================================

/**
 * Generates a ZIP archive from multiple files.
 *
 * @param files - Array of files to include in the archive
 * @param options - ZIP generation options
 * @returns Promise resolving to generation result
 *
 * @example
 * ```tsx
 * const files: ZipFileEntry[] = [
 *   { filename: "chart.png", content: pngBlob },
 *   { filename: "data.csv", content: new TextEncoder().encode(csvString) },
 * ];
 *
 * const result = await generateZip(files, { comment: "Campaign Export" });
 * if (result.success && result.blob) {
 *   downloadFile(result.blob, "campaign-export.zip");
 * }
 * ```
 */
export async function generateZip(
  files: ZipFileEntry[],
  options: ZipGenerateOptions = {}
): Promise<ZipGenerateResult> {
  const { comment = "", compress = true } = options;

  try {
    if (files.length === 0) {
      return {
        success: true,
        blob: new Blob([], { type: "application/zip" }),
        fileCount: 0,
        totalSize: 0,
      };
    }

    const parts: Uint8Array[] = [];
    const centralDirEntries: CentralDirEntry[] = [];
    let offset = 0;
    let totalUncompressedSize = 0;

    // Process each file
    for (const file of files) {
      const filenameBytes = encodeUTF8(file.filename);
      const uncompressedData = await contentToBytes(file.content);
      const crc = crc32(uncompressedData);
      const modDate = file.lastModified ?? new Date();

      totalUncompressedSize += uncompressedData.length;

      // Compress if enabled
      let compressedData: Uint8Array;
      let compressionMethod: number;

      if (compress) {
        const result = await compressData(uncompressedData);
        compressedData = result.compressed;
        compressionMethod = result.method;
      } else {
        compressedData = uncompressedData;
        compressionMethod = COMPRESSION_STORE;
      }

      // Create local file header
      const localHeader = createLocalFileHeader(
        filenameBytes,
        compressedData,
        crc,
        compressionMethod,
        modDate
      );

      // Update sizes for compression
      localHeader.compressedSize = compressedData.length;
      localHeader.uncompressedSize = uncompressedData.length;

      // Serialize and add to parts
      const headerBytes = serializeLocalFileHeader(localHeader);
      parts.push(headerBytes);
      parts.push(compressedData);

      // Create central directory entry
      const centralEntry = createCentralDirEntry(
        filenameBytes,
        crc,
        compressedData.length,
        uncompressedData.length,
        compressionMethod,
        modDate,
        offset
      );
      centralDirEntries.push(centralEntry);

      // Update offset
      offset += headerBytes.length + compressedData.length;
    }

    // Write central directory
    const centralDirOffset = offset;
    let centralDirSize = 0;

    for (const entry of centralDirEntries) {
      const entryBytes = serializeCentralDirEntry(entry);
      parts.push(entryBytes);
      centralDirSize += entryBytes.length;
    }

    // Write end of central directory
    const commentBytes = encodeUTF8(comment);
    const eocd = createEndOfCentralDir(
      files.length,
      centralDirSize,
      centralDirOffset,
      commentBytes
    );
    parts.push(serializeEndOfCentralDir(eocd));

    // Combine all parts
    const totalLength = parts.reduce((sum, p) => sum + p.length, 0);
    const zipData = new Uint8Array(totalLength);
    let pos = 0;
    for (const part of parts) {
      zipData.set(part, pos);
      pos += part.length;
    }

    return {
      success: true,
      blob: new Blob([zipData], { type: "application/zip" }),
      fileCount: files.length,
      totalSize: totalUncompressedSize,
    };
  } catch (error) {
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Unknown error during ZIP generation",
    };
  }
}

/**
 * Downloads a ZIP file to the user's device.
 *
 * @param blob - ZIP blob to download
 * @param filename - Filename for the download
 * @returns Promise resolving to success status
 */
export async function downloadZip(
  blob: Blob,
  filename: string
): Promise<boolean> {
  try {
    // Ensure filename has .zip extension
    const finalFilename = filename.endsWith(".zip")
      ? filename
      : `${filename}.zip`;

    // Create object URL for the blob
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor element
    const link = document.createElement("a");
    link.href = url;
    link.download = finalFilename;

    // Append to body (required for Firefox)
    document.body.appendChild(link);

    // Trigger download
    link.click();

    // Cleanup
    document.body.removeChild(link);

    // Revoke object URL after a small delay to ensure download starts
    setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 100);

    return true;
  } catch {
    return false;
  }
}

/**
 * Generates and downloads a ZIP file in one operation.
 *
 * @param files - Array of files to include
 * @param filename - Filename for the download
 * @param options - ZIP generation options
 * @returns Promise resolving to generation result with download status
 */
export async function generateAndDownloadZip(
  files: ZipFileEntry[],
  filename: string,
  options?: ZipGenerateOptions
): Promise<ZipGenerateResult & { downloaded: boolean }> {
  const result = await generateZip(files, options);

  if (!result.success || !result.blob) {
    return { ...result, downloaded: false };
  }

  const downloaded = await downloadZip(result.blob, filename);

  return { ...result, downloaded };
}

/**
 * Generates a filename for ZIP export with timestamp.
 *
 * @param baseName - Base name for the file
 * @param includeTimestamp - Whether to include timestamp (default: true)
 * @returns Generated filename with .zip extension
 */
export function generateZipFilename(
  baseName: string,
  includeTimestamp: boolean = true
): string {
  // Sanitize base name (remove unsafe characters)
  const sanitized = baseName
    .toLowerCase()
    .replace(/[^a-z0-9-_]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");

  if (includeTimestamp) {
    const timestamp = new Date()
      .toISOString()
      .replace(/:/g, "-")
      .replace(/\..+/, "");
    return `${sanitized}-${timestamp}.zip`;
  }

  return `${sanitized}.zip`;
}

/**
 * Checks if ZIP export is supported in the current browser.
 *
 * @returns Object indicating support for ZIP export features
 */
export function checkZipExportSupport(): {
  blob: boolean;
  download: boolean;
  compression: boolean;
  overall: boolean;
} {
  const blob = typeof Blob !== "undefined";
  const download =
    typeof document !== "undefined" &&
    "download" in document.createElement("a");
  const compression = typeof CompressionStream !== "undefined";

  return {
    blob,
    download,
    compression,
    overall: blob && download,
  };
}
