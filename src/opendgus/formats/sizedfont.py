"""DGUS II multi-size bitmap fonts ("DGUS_2" ``N.bin`` files used by text_display / data_var).

Layout (reverse engineered from DWIN fonts that ship in DWIN_SET folders, see
docs/re/03-assets-icl-fonts.md section 3.7):

* 32-byte header: ``"DGUS_2"``, byte 12 = 2, byte 13 = number of sub-fonts;
* from 0x20, 8-byte size records ``[5A][w][h][01][base u32 BE]`` up to the first
  sub-font. The record for a control's ``font_x`` is ``font_x - 4``; records that
  share a ``base`` share one sub-font whose glyph height is the largest ``h``
  among them (the first record of a sub-font may declare a smaller one);
* at ``base + 5 * code_point``: ``[valid_width u8][0xF0 | absolute address u32 BE]``;
  a missing glyph has ``valid_width == 0`` or no 0xF0 marker;
* a glyph is ``valid_width`` x ``height`` pixels, 4 bits per pixel, high nibble first,
  rows not padded.

Unlike the single-size Unicode gray font in :mod:`opendgus.formats.fonts`, one file
carries many sizes and is indexed by ASCII code point.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from PIL import Image

MAGIC = b"DGUS_2"
TABLE = 0x20
RECORD = 8
SIZE_RECORD_OFFSET = 4       # record index = font_x - 4
INDEX_ENTRY = 5


@dataclass(frozen=True)
class Glyph:
    width: int               # valid width in pixels
    height: int
    mask: Image.Image        # "L" image, 0 = background .. 255 = ink


class SizedGrayFont:
    """Reader for a multi-size ``DGUS_2`` font. ``glyph(code_point, font_x)`` -> :class:`Glyph` or ``None``."""

    def __init__(self, data: bytes):
        if data[:6] != MAGIC or len(data) < TABLE + RECORD:
            raise ValueError("not a multi-size DGUS_2 font")
        self.data = data
        self.records: list[tuple[int, int, int]] = []        # (max_w, declared_h, base)
        off = TABLE
        while off + RECORD <= len(data) and data[off] == 0x5A:
            self.records.append((data[off + 1], data[off + 2], struct.unpack(">I", data[off + 4:off + 8])[0]))
            off += RECORD
        if not self.records:
            raise ValueError("no size records")
        heights: dict[int, int] = {}
        for _, h, base in self.records:
            heights[base] = max(heights.get(base, 0), h)
        self._heights = heights

    @classmethod
    def from_file(cls, path) -> "SizedGrayFont":
        with open(path, "rb") as fh:
            return cls(fh.read())

    def sizes(self) -> range:
        """Valid ``font_x`` values."""
        return range(SIZE_RECORD_OFFSET, SIZE_RECORD_OFFSET + len(self.records))

    def _entry(self, code_point: int, font_x: int):
        i = font_x - SIZE_RECORD_OFFSET
        if not 0 <= i < len(self.records) or not 0 <= code_point <= 0xFF:
            return None
        base = self.records[i][2]
        pos = base + INDEX_ENTRY * code_point
        if pos + INDEX_ENTRY > len(self.data):
            return None
        width = self.data[pos]
        raw = struct.unpack(">I", self.data[pos + 1:pos + 5])[0]
        height = self._heights[base]
        addr = raw & 0x0FFFFFFF
        if raw >> 28 != 0xF or not width or addr + (width * height + 1) // 2 > len(self.data):
            return None
        return width, height, addr

    def width(self, code_point: int, font_x: int):
        """Valid width in pixels of a glyph, or ``None`` when the font has none (cheaper than :meth:`glyph`)."""
        e = self._entry(code_point, font_x)
        return None if e is None else e[0]

    def glyph(self, code_point: int, font_x: int):
        e = self._entry(code_point, font_x)
        if e is None:
            return None
        width, height, addr = e
        packed = self.data[addr:addr + (width * height + 1) // 2]
        pixels = bytearray(width * height)
        for i in range(width * height):
            b = packed[i >> 1]
            pixels[i] = ((b >> 4) if i % 2 == 0 else (b & 15)) * 17
        return Glyph(width, height, Image.frombytes("L", (width, height), bytes(pixels)))
