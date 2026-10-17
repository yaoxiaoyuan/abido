#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随笔 — draw one random painting from whichever of the nine painters comes up.

The nine modules already do all the work and they already agree on an
interface:

    paint(path, size=(w, h), seed=None, quiet=False) -> (path, title, seed)

and each one picks its own subject, weather, palette and ground when it is not
told.  So the only thing left to choose is *which brush to pick up*, and this
script does nothing else.  No new drawing code lives here.

A batch is *dealt* rather than sampled: ask for nine and you get one from
each of the nine painters, in a random order, each one as unrelated to its
neighbours as two separate runs would be.  Ask for twelve and three painters
get two.  Only the counts are pinned — see `deal` for why, and for what it
deliberately does not touch.

Draw more than one and it also mounts them as a 册页 — a contact sheet, four
leaves to a row by default, each with its own title under it.  That is the
only picture this file composes itself, and even then it composes no strokes:
it reads the PNGs back off disk and shrinks them.

    python3 random_paint.py                     # one painting, random painter
    python3 random_paint.py -n 4                 # four of them + random-sheet.png
    python3 random_paint.py -n 9 --cols 3        # three to a row
    python3 random_paint.py -n 4 --no-sheet      # just the four paintings
    python3 random_paint.py -p orchid -s 909     # a named painter and seed
    python3 random_paint.py --size 600x800       # override the sheet
    python3 random_paint.py --list
"""

import argparse
import os
import random
import struct
import time
import zlib

import gongbi
import shanshui
import bamboo
import fish
import pavilion
import orchid
import grape
import crab
import mei

# name -> (module, 画科, the module's own default sheet)
PAINTERS = {
    "gongbi":   (gongbi,   "工笔花卉", (1000, 1360)),
    "shanshui": (shanshui, "山水",     (1000, 1400)),
    "bamboo":   (bamboo,   "墨竹",     (900, 1300)),
    "fish":     (fish,   "游鱼",     (900, 1200)),
    "pavilion": (pavilion, "亭景",     (900, 1200)),
    "orchid":   (orchid,   "墨兰",     (900, 1300)),
    "grape":    (grape,    "墨葡萄",   (900, 1300)),
    "crab":     (crab,     "虾蟹",     (900, 1200)),
    "mei":      (mei,      "墨梅",     (900, 1300)),
}


def deal(n, rng):
    """`n` painter names, spread as evenly over `PAINTERS` as `n` allows.

    Drawing each sheet's painter independently from its own seed is the
    obvious thing and it gives a *multinomial*, which is much lumpier than
    people expect a random batch to look: at n = 8 over eight painters only
    0.2% of batches contain one of each, and 9% hand four or more of the eight
    leaves to a single painter.  That is correct randomness and a bad 册页 --
    four 墨兰 beside no 山水 at all reads as a broken launcher, not as chance.

    So the *counts* are fixed and only the arrangement is random.  Every
    painter gets ``n // k`` sheets; the ``n % k`` left over go to that many
    distinct painters drawn without replacement, so no two counts ever differ
    by more than one and no painter is favoured for the remainder.  Then the
    whole multiset is shuffled, which makes the order a uniformly random
    permutation of it -- the most random order there is, given the counts.

    This decides *which brush*, and nothing else.  Each sheet still gets its
    own seed and the module still picks its own subject, weather, palette and
    ground from it, so the paintings themselves are as independent as before:
    a balanced batch of eight is eight unrelated paintings, not a sampler.
    """
    names = sorted(PAINTERS)
    picks = names * (n // len(names))
    picks += rng.sample(names, n % len(names))
    rng.shuffle(picks)
    return picks


def paint_random(out="out.png", seed=None, painter=None, size=None,
                 quiet=False):
    """Draw one painting.  Returns (painter_name, path, title, seed)."""
    if seed is None:
        seed = random.randrange(1 << 30)
    # With no painter named, choose one from the seed, so one number reproduces
    # the whole painting and not just its contents -- but from a *namespaced*
    # stream.  (`main` deals the painter instead and always names one, so this
    # branch is for a single sheet asked for from code, and for the test that
    # checks the namespacing below still holds.)
    #
    # `random.Random(seed).randrange(6)` was the obvious way to write this and
    # it is wrong, because the module is then handed the same `seed` and opens
    # its own `random.Random(seed)` whose first act is to pick its subject from
    # a list of three, four or six. CPython's `_randbelow(n)` takes
    # `getrandbits(n.bit_length())`, which is the *top* bits of the generator's
    # first 32-bit word -- so the launcher's choice and the painter's choice
    # read the same bits of the same word, and knowing one *is* knowing the
    # other. The result was not a weak correlation but an identity: every
    # gongbi sheet the launcher ever produced was 荷花, every orchid 幽兰, and
    # 6 of the 24 subjects were reachable at all. (The air, 皴 and palette
    # axes come from later draws and did vary, which is what hid it.)
    #
    # Seeding from a string takes `random.seed`'s sha512 path, so this stream
    # shares no state with the painter's and the correlation is gone. `seed`
    # itself is still what the module gets, so `-p orchid -s 909` is still
    # byte-identical to `orchid.py -s 909`.
    name = painter or sorted(PAINTERS)[
        random.Random("painter|%d" % seed).randrange(len(PAINTERS))]
    if name not in PAINTERS:
        raise SystemExit("unknown painter %r; try --list" % name)
    mod, cn, default_size = PAINTERS[name]

    if not quiet:
        print("%s  %s" % (name, cn))
    path, title, seed = mod.paint(out, size=size or default_size, seed=seed,
                                 quiet=quiet)
    return name, path, title, seed


# --------------------------------------------------------------------------
# 册页 — mounting a batch on one sheet
#
# Still no drawing code: the paintings are already PNGs on disk, so the
# summary is a read, a box-average down and a paste.  It is laid out as a
# mount board with the leaves on it rather than as a flush grid of
# thumbnails, because that is what a 册页 is and because the titles have to
# go somewhere.
#
# Note what this deliberately does *not* do: re-render each painting small.
# Every painter scales its composition to the sheet it is given -- stroke
# widths, how far the vine reaches, how many leaves fit -- so a 300-pixel
# re-render is a different painting, not a thumbnail of this one.  The only
# honest thumbnail of a painting is the painting, shrunk.
# --------------------------------------------------------------------------

def read_png(path):
    """(w, h, bytearray) of 8-bit RGB pixels.

    Only as general as it has to be.  Every painter in this family writes
    through `gongbi.Canvas.write_png`, which emits 8-bit truecolour, no
    interlace, one IDAT, filter 0 on every row.  The other four filter types
    are implemented anyway so that this cannot quietly become wrong if that
    writer ever changes.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("%s is not a PNG" % path)
    i, w, h, idat = 8, 0, 0, bytearray()
    while i + 8 <= len(data):
        ln = struct.unpack(">I", data[i:i + 4])[0]
        tag = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + ln]
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, inter = struct.unpack(">IIBBBBB", body)
            if (depth, ctype, inter) != (8, 2, 0):
                raise ValueError("%s: want 8-bit RGB, got depth=%d colour=%d"
                                 " interlace=%d" % (path, depth, ctype, inter))
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
        i += 12 + ln
    raw = zlib.decompress(bytes(idat))
    stride = w * 3
    out = bytearray(w * h * 3)
    prev = bytes(stride)
    pos = 0
    for y in range(h):
        ft = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if ft == 1:                                   # Sub
            for x in range(3, stride):
                line[x] = (line[x] + line[x - 3]) & 255
        elif ft == 2:                                 # Up
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif ft == 3:                                 # Average
            for x in range(stride):
                a = line[x - 3] if x >= 3 else 0
                line[x] = (line[x] + ((a + prev[x]) >> 1)) & 255
        elif ft == 4:                                 # Paeth
            for x in range(stride):
                a = line[x - 3] if x >= 3 else 0
                c = prev[x - 3] if x >= 3 else 0
                b = prev[x]
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc
                                      else (b if pb <= pc else c))) & 255
        elif ft != 0:
            raise ValueError("%s: unknown row filter %d" % (path, ft))
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return w, h, out


def _paste(cv, src, sw, sh, x0, y0, tw, th, samples=2):
    """Box-average `src` down into the tw x th box at (x0, y0).

    `samples`² taps per destination pixel rather than the whole box: these
    sheets are full of hairlines -- veins, tendrils, 皴 -- and nearest
    neighbour drops half of them and aliases the rest into dashes.  Four taps
    is enough to keep a one-pixel vein as a grey one, and costs a quarter of
    what averaging every source pixel would.
    """
    kx, ky = sw / tw, sh / th
    off = [(j + 0.5) / samples for j in range(samples)]
    n = samples * samples
    dst = cv.buf
    for ty in range(th):
        ys = [min(sh - 1, int((ty + o) * ky)) * sw for o in off]
        drow = (y0 + ty) * cv.w + x0
        for tx in range(tw):
            xs = [min(sw - 1, int((tx + o) * kx)) for o in off]
            r = g = b = 0
            for base in ys:
                for xx in xs:
                    i = (base + xx) * 3
                    r += src[i]
                    g += src[i + 1]
                    b += src[i + 2]
            j = (drow + tx) * 3
            dst[j] = r // n
            dst[j + 1] = g // n
            dst[j + 2] = b // n


def _frame(cv, x0, y0, w, h, ink, alpha):
    """One hairline rectangle — the mount's cut line round a leaf."""
    for x in range(x0, x0 + w):
        cv.blend(x, y0, ink, alpha)
        cv.blend(x, y0 + h - 1, ink, alpha)
    for y in range(y0, y0 + h):
        cv.blend(x0, y, ink, alpha)
        cv.blend(x0 + w - 1, y, ink, alpha)


def _caption(cv, rng, font, text, cx, y, s, ink):
    """One horizontal line of writing, centred on `cx`.

    `gongbi.paint_column` writes downward, which is right for an inscription
    inside a painting and wrong for a label under one.  Each painter's own
    font blob is used, because the title came out of that painter's own
    tables and is the one thing its blob is guaranteed to cover.
    """
    if font is None:
        return
    chars = [c for c in text if font.has(c)]
    if not chars:
        return
    adv = s * 1.06
    x = cx - adv * (len(chars) - 1) * 0.5 - s * 0.5
    mottle = gongbi.Noise(rng, 8, 8)
    for i, ch in enumerate(chars):
        gongbi.fill_contours(
            cv, font.outline(ch, x + i * adv + rng.uniform(-0.03, 0.03) * s,
                             y, s, rot=rng.uniform(-0.03, 0.03)),
            gongbi.jitter(ink, rng, 6), rng.uniform(0.66, 0.88), mottle)


def contact_sheet(items, out, cols=4, tile=300, quiet=False):
    """Mount `items` -- (painter, path, title, seed) -- on one PNG.

    Returns (path, width, height) or None if nothing could be read.
    """
    t0 = time.time()
    leaves = []
    for name, path, title, seed in items:
        try:
            w, h, buf = read_png(path)
        except Exception as exc:                  # a half-written file, say
            print("  (skipping %s: %s)" % (path, exc))
            continue
        leaves.append((name, title, w, h, buf))
    if not leaves:
        return None

    cols = max(1, min(cols, len(leaves)))
    rows = (len(leaves) + cols - 1) // cols
    # One cell shape for every leaf, tall enough for the tallest aspect in
    # the batch: the painters do not agree on a sheet (1000x1360 for 工笔,
    # 900x1200 for 游鱼), and a grid of differently shaped cells reads as a
    # mistake even when every cell is correct.
    asp = max(h / w for _, _, w, h, _ in leaves)
    cellh = int(round(tile * asp))
    cap = int(round(tile * 0.13))                 # the strip for the title
    gs = tile * 0.070                             # glyph size
    gut = int(round(tile * 0.085))
    mar = int(round(tile * 0.105))
    W = mar * 2 + cols * tile + (cols - 1) * gut
    H = mar * 2 + rows * (cellh + cap) + (rows - 1) * gut

    # a mount board, a shade deeper than any of the painters' papers
    cv = gongbi.Canvas(W, H, bg=(227, 219, 201))
    rng = random.Random("sheet|" + ",".join(str(s) for _, _, _, s in items))
    ink = bamboo.dilute(bamboo.I("nong"), 0.84)
    hair = bamboo.dilute(bamboo.I("zhong"), 0.52)

    for k, (name, title, sw, sh, buf) in enumerate(leaves):
        row, col = k // cols, k % cols
        # A short last row is centred rather than left-aligned. Five leaves on
        # a four-wide board otherwise hang one in the corner with three cells
        # of empty mount beside it, which reads as an accident; 册页 are bound
        # about a centre fold and a short leaf sits on it.
        in_row = min(cols, len(leaves) - row * cols)
        cx0 = (mar + (cols - in_row) * (tile + gut) // 2
               + col * (tile + gut))
        cy0 = mar + row * (cellh + cap + gut)
        scale = min(tile / sw, cellh / sh)
        tw, th = max(1, int(sw * scale)), max(1, int(sh * scale))
        x0 = cx0 + (tile - tw) // 2
        y0 = cy0 + (cellh - th) // 2
        _paste(cv, buf, sw, sh, x0, y0, tw, th)
        # 双框 — the cut line, and a second line a little out from it
        _frame(cv, x0 - 1, y0 - 1, tw + 2, th + 2, hair, 0.34)
        _frame(cv, x0 - 5, y0 - 5, tw + 10, th + 10, hair, 0.13)
        font = getattr(PAINTERS[name][0], "FONT", None)
        _caption(cv, rng, font, title, cx0 + tile * 0.5,
                 cy0 + cellh + cap * 0.12, gs, ink)

    cv.write_png(out)
    if not quiet:
        print("  %-24s 册页 %dx%d  %dx%d  %.1fs"
              % (os.path.basename(out), cols, rows, W, H, time.time() - t0))
    return out, W, H


def main(argv=None):
    ap = argparse.ArgumentParser(description="draw a random Chinese painting")
    ap.add_argument("-o", "--out", default="random.png")
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("-s", "--seed", type=int, default=None)
    ap.add_argument("-p", "--painter", default=None,
                    help="gongbi | shanshui | bamboo | fish | pavilion"
                         " | orchid | grape | crab | mei")
    ap.add_argument("--size", default=None, help="WxH, e.g. 900x1200")
    ap.add_argument("--cols", type=int, default=4,
                    help="leaves per row on the 册页 summary (default 4)")
    ap.add_argument("--tile", type=int, default=300,
                    help="width of one leaf on the summary, px (default 300)")
    ap.add_argument("--sheet", default=None,
                    help="where to write the summary (default <out>-sheet.png)")
    ap.add_argument("--no-sheet", action="store_true",
                    help="skip the summary even when drawing more than one")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        for k in sorted(PAINTERS):
            mod, cn, sz = PAINTERS[k]
            print("  %-9s %-9s %dx%d  (%s.py)" % (k, cn, sz[0], sz[1], k))
        return 0

    size = None
    if a.size:
        w, _, h = a.size.lower().partition("x")
        size = (int(w), int(h))

    stem, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    n = max(1, a.count)

    # The seeds first, then the painters.  Each sheet's seed is the only thing
    # that decides what it contains, exactly as before -- but the painter is
    # now dealt (see `deal`) instead of being derived from that seed, so a
    # batch is balanced by construction.  The deal is itself seeded from the
    # seeds, so `-s 909 -n 8` still reproduces the whole 册页, leaf order
    # included, from one number.
    seeds = [random.randrange(1 << 30) if a.seed is None else a.seed + i
             for i in range(n)]
    if a.painter:
        picks = [a.painter] * n
    else:
        picks = deal(n, random.Random(
            "deal|%d|" % n + ",".join(str(s) for s in seeds)))

    done = []
    for i, (name, seed) in enumerate(zip(picks, seeds)):
        out = a.out if n == 1 else "%s-%d%s" % (stem, i + 1, ext)
        done.append(paint_random(out, seed=seed, painter=name, size=size))

    if n > 1 and not a.no_sheet:
        contact_sheet(done, a.sheet or "%s-sheet%s" % (stem, ext),
                      cols=a.cols, tile=a.tile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
