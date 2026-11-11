"""Multi-size ``N.bin`` bitmap fonts: the reader (formats.sizedfont) and the emulator using them."""
import struct

import pytest
from PIL import Image

from opendgus.emulator import Device
from opendgus.formats import controls as C
from opendgus.formats.sizedfont import SizedGrayFont

SIZE_RECORDS = 252                              # font_x 4..255, as the DWIN tool writes
INDEX_BYTES = 5 * 256                           # one 5-byte entry per code point 0..255


def make_font(sub_fonts):
    """A ``DGUS_2`` multi-size font from ``{(first_font_x, height): {code_point: (width, gray4_values)}}``.

    Each sub-font serves font_x from its first value up to the next sub-font's first value.
    """
    firsts = sorted(sub_fonts)
    base = 0x20 + 8 * SIZE_RECORDS
    bases, blob = {}, bytearray()
    for key in firsts:
        bases[key] = base + len(blob)
        index, glyphs = bytearray(INDEX_BYTES), bytearray()
        for cp, (width, px) in sorted(sub_fonts[key].items()):
            assert len(px) == width * key[1]
            packed = bytearray((len(px) + 1) // 2)
            for i, v in enumerate(px):
                packed[i >> 1] |= (v << 4) if i % 2 == 0 else v
            index[5 * cp] = width
            index[5 * cp + 1:5 * cp + 5] = struct.pack(">I", 0xF0000000 | (bases[key] + INDEX_BYTES + len(glyphs)))
            glyphs += packed
        blob += index + glyphs
    table = bytearray()
    for font_x in range(4, 4 + SIZE_RECORDS):
        key = max(k for k in firsts if k[0] <= font_x)
        table += bytes([0x5A, 0, key[1], 1]) + struct.pack(">I", bases[key])
    return b"DGUS_2" + bytes(26) + bytes(table) + bytes(blob)


@pytest.fixture
def font_bytes():
    # font_x 4..9 -> 8 px tall; font_x 10.. -> 20 px tall. 'A' is a solid block, 'i' is narrow and half-grey.
    return make_font({
        (4, 8): {ord("A"): (6, [15] * 48), ord("i"): (2, [15] * 16)},
        (10, 20): {ord("A"): (14, [15] * 280), ord("i"): (4, [8] * 80)},
    })


def test_size_selects_the_sub_font_and_height(font_bytes):
    f = SizedGrayFont(font_bytes)
    small, large = f.glyph(ord("A"), 6), f.glyph(ord("A"), 12)
    assert (small.width, small.height) == (6, 8)
    assert (large.width, large.height) == (14, 20)
    assert large.mask.size == (14, 20) and large.mask.getpixel((0, 0)) == 255     # level 15 -> full ink
    assert f.glyph(ord("i"), 12).mask.getpixel((0, 0)) == 8 * 17                  # 4-bit level kept


def test_missing_glyph_or_size_is_none(font_bytes):
    f = SizedGrayFont(font_bytes)
    assert f.glyph(ord("z"), 12) is None                                          # no glyph for the code point
    assert f.glyph(0x20, 12) is None and f.glyph(0x1FF, 12) is None
    assert f.glyph(ord("A"), 3) is None and f.glyph(ord("A"), 256) is None        # outside font_x 4..255
    assert f.width(ord("A"), 12) == 14 and f.width(ord("z"), 12) is None


def test_rejects_other_bin_files():
    with pytest.raises(ValueError):
        SizedGrayFont(b"\x00" * 64)
    with pytest.raises(ValueError):
        SizedGrayFont(b"DGUS_2" + bytes(26))                                      # magic but no size records


def _folder(tmp_path, font0, with_font, font_bytes, h_align=0, hor_dis=0):
    """A 800x480 DWIN_SET with one text_display at (20, 20) showing text from VP 0x5000."""
    rec = C.text_display(0x5000, (20, 20), 0xFFFF, (20, 20, 300, 60), 8, font0, 0, 12, 24, 0,
                         h_align=h_align, hor_dis=hor_dis)
    (tmp_path / "14.bin").write_bytes(C.build_14({0: [rec]}))
    (tmp_path / "13.bin").write_bytes(C.build_13([]))
    (tmp_path / "22.bin").write_bytes(bytes(0x20000))
    if with_font:
        (tmp_path / "1.bin").write_bytes(font_bytes)
    return tmp_path


def _write_text(dev, text):
    assert len(text) % 2 == 0, "the panel ends a string with 0xFFFF on a word boundary"
    raw = text.encode().ljust(8, b"\xff") + b"\xff\xff"
    dev.write_vp(0x5000, [int.from_bytes(raw[i:i + 2], "big") for i in range(0, len(raw), 2)])


def _ink_columns(img, box):
    """x positions in `box` where any pixel differs from the page grey (the text is white, blended by glyph level)."""
    px = img.convert("L").load()
    x0, y0, x1, y1 = box
    return [x for x in range(x0, x1) if any(abs(px[x, y] - 40) > 20 for y in range(y0, y1))]


def test_text_display_draws_the_real_font_with_glyph_advance(tmp_path, font_bytes):
    dev = Device.from_folder(_folder(tmp_path, font0=1, with_font=True, font_bytes=font_bytes, hor_dis=2))
    assert 1 in dev._font_libs and not dev.warnings
    _write_text(dev, "Ai")
    cols = _ink_columns(dev.render(), (0, 0, 800, 480))
    assert cols, "nothing drawn"
    # font_x=12 selects the 20 px sub-font: 'A' is 14 wide, 'i' is 4 wide, hor_dis adds 2 between them
    assert (cols[0], cols[-1]) == (20, 20 + 14 + 2 + 4 - 1)                       # A, hor_dis gap, i
    assert any("real N.bin font" in n for n in dev.notes)


def test_text_display_without_the_font_file_falls_back_to_the_stand_in(tmp_path, font_bytes):
    dev = Device.from_folder(_folder(tmp_path, font0=1, with_font=False, font_bytes=font_bytes))
    assert not dev._font_libs
    _write_text(dev, "Ai")
    assert _ink_columns(dev.render(), (0, 0, 800, 480))                           # still draws, with the TTF
    assert any("TTF stand-in" in n for n in dev.notes) and not any("real N.bin font" in n for n in dev.notes)


def test_centred_text_uses_the_real_glyph_width_not_the_cell(tmp_path, font_bytes):
    # box 20..300 (280 wide). "AA" is 2 x 14 = 28 px in the font; the cell would give 2 x 12 = 24.
    dev = Device.from_folder(_folder(tmp_path, font0=1, with_font=True, font_bytes=font_bytes, h_align=2))
    _write_text(dev, "AA")
    cols = _ink_columns(dev.render(), (0, 0, 800, 480))
    left = 20 + (280 - 28) // 2
    assert (cols[0], cols[-1]) == (left, left + 27)
