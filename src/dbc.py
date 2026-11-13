"""Pure-Python reader for DATASUS ``.dbc`` files (Brazil's SINAN surveillance microdata).

A ``.dbc`` file is a dBase (DBF) table whose records are compressed with the PKWare DCL "implode"
algorithm. The usual readers (pyreaddbc, PySUS) need a C compiler; this module needs nothing but the
standard library, so it works on any platform and Python version.

Layout of a .dbc file:
    [DBF header, uncompressed][4-byte CRC][DCL-imploded DBF records]

The decompressor is a port of Mark Adler's ``blast.c`` (zlib/contrib/blast). Records are decoded while
streaming, so a 1 GB table never has to fit in memory.

Usage:
    for row in read_dbc("CHIKBR23.dbc", columns=["CLASSI_FIN", "FEBRE"]):
        ...
"""
import struct

MAXBITS = 13

# Compact code-length tables of the DCL format (from blast.c): each byte = (repeat - 1) << 4 | bit length.
_LITLEN = bytes([
    11, 124, 8, 7, 28, 7, 188, 13, 76, 4, 10, 8, 12, 10, 12, 10, 8, 23, 8,
    9, 7, 6, 7, 8, 7, 6, 55, 8, 23, 24, 12, 11, 7, 9, 11, 12, 6, 7, 22, 5,
    7, 24, 6, 11, 9, 6, 7, 22, 7, 11, 38, 7, 9, 8, 25, 11, 8, 11, 9, 12,
    8, 12, 5, 38, 5, 38, 5, 11, 7, 5, 6, 21, 6, 10, 53, 8, 7, 24, 10, 27,
    44, 253, 253, 253, 252, 252, 252, 13, 12, 45, 12, 45, 12, 61, 12, 45,
    44, 173])
_LENLEN = bytes([2, 35, 36, 53, 38, 23])
_DISTLEN = bytes([2, 20, 53, 230, 247, 151, 248])
_BASE = (3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264)
_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def _decode_table(compact):
    """Lookup table indexed by the next MAXBITS input bits (LSB first) -> (symbol, code length).

    DCL Huffman codes are canonical, stored bit-inverted and read most-significant bit first.
    """
    lengths = []
    for byte in compact:
        lengths += [byte & 15] * ((byte >> 4) + 1)
    table = [None] * (1 << MAXBITS)
    code = 0
    for length in range(1, MAXBITS + 1):
        for symbol, sym_len in enumerate(lengths):
            if sym_len != length:
                continue
            raw = 0  # the bits as they appear in the stream (first bit = bit 0), inverted
            for i in range(length):
                raw |= (1 - ((code >> (length - 1 - i)) & 1)) << i
            for high in range(1 << (MAXBITS - length)):
                table[raw | (high << length)] = (symbol, length)
            code += 1
        code <<= 1
    return table


_LITCODE = _decode_table(_LITLEN)
_LENCODE = _decode_table(_LENLEN)
_DISTCODE = _decode_table(_DISTLEN)


def _blast(data, start):
    """Generator of decompressed chunks (bytes) of a DCL-imploded stream beginning at ``data[start]``."""
    pos, bitbuf, bitcnt = start, 0, 0
    n = len(data)

    def need(k):
        nonlocal pos, bitbuf, bitcnt
        while bitcnt < k:
            bitbuf |= (data[pos] if pos < n else 0) << bitcnt
            pos += 1
            bitcnt += 8

    def bits(k):
        nonlocal bitbuf, bitcnt
        need(k)
        value = bitbuf & ((1 << k) - 1)
        bitbuf >>= k
        bitcnt -= k
        return value

    def decode(table):
        nonlocal bitbuf, bitcnt
        need(MAXBITS)
        symbol, length = table[bitbuf & ((1 << MAXBITS) - 1)]
        bitbuf >>= length
        bitcnt -= length
        return symbol

    lit, dict_bits = bits(8), bits(8)
    if lit > 1 or not 4 <= dict_bits <= 6:
        raise ValueError("Not a PKWare DCL imploded stream")

    out = bytearray()
    while True:
        if bits(1):
            symbol = decode(_LENCODE)
            length = _BASE[symbol] + bits(_EXTRA[symbol])
            if length == 519:  # end of stream
                break
            shift = 2 if length == 2 else dict_bits
            dist = (decode(_DISTCODE) << shift) + bits(shift) + 1
            if dist > len(out):
                raise ValueError("Corrupt stream: distance too far back")
            if dist >= length:
                out += out[-dist:len(out) - dist + length]
            else:  # overlapping copy (repeats the last `dist` bytes)
                for _ in range(length):
                    out.append(out[-dist])
        else:
            out.append(decode(_LITCODE) if lit else bits(8))
        if len(out) > 1 << 16:  # hand over everything except the 4 KB window needed for back-references
            yield bytes(out[:-4096])
            del out[:-4096]
    yield bytes(out)


def _fields(header):
    """Parse the DBF field descriptors: list of (name, offset in record, width)."""
    fields, offset = [], 1  # byte 0 of each record is the deletion flag
    for i in range(32, len(header) - 1, 32):
        if header[i] == 0x0D:
            break
        name = header[i:i + 11].split(b"\x00")[0].decode("latin-1")
        width = header[i + 16]
        fields.append((name, offset, width))
        offset += width
    return fields


def dbc_columns(path):
    """Column names of a .dbc file (reads only the header)."""
    with open(path, "rb") as f:
        head = f.read(32)
        header_len = struct.unpack("<H", head[8:10])[0]
        return [name for name, _, _ in _fields(head + f.read(header_len - 32))]


def read_dbc(path, columns=None):
    """Yield each record of a .dbc file as a dict {column: stripped string}.

    columns: optional list of column names to keep (faster and lighter than keeping all of them).
    """
    with open(path, "rb") as f:
        data = f.read()
    n_records = struct.unpack("<I", data[4:8])[0]
    header_len, record_len = struct.unpack("<HH", data[8:12])
    fields = _fields(data[:header_len])
    if columns is not None:
        wanted = set(columns)
        fields = [fld for fld in fields if fld[0] in wanted]

    buffer, emitted = bytearray(), 0
    for chunk in _blast(data, header_len + 4):
        buffer += chunk
        usable = len(buffer) - len(buffer) % record_len
        for start in range(0, usable, record_len):
            if emitted == n_records:
                return
            record = buffer[start:start + record_len]
            if record[0] != 0x2A:  # "*" marks a deleted record
                yield {name: record[off:off + width].decode("latin-1").strip() for name, off, width in fields}
            emitted += 1
        del buffer[:usable]
