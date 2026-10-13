#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_glyphs.py — bake the inscription vocabulary of gongbi.py into a tiny
self-contained data module (gongbi_font.py).

Reads a TrueType font (quadratic `glyf` outlines), pulls the contours of just
the characters the generator can write, flattens the curves, quantises them to
a 1000-unit em square and delta-encodes everything into one zlib+base85 blob.
The painter then needs no font installed and no third-party library.

    python3 tools/extract_glyphs.py FONT.ttf -o gongbi_font.py

Add characters by editing CHARS in gongbi.py (vocab is imported from there).
"""

import argparse
import base64
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ON_CURVE, X_SHORT, Y_SHORT, REPEAT, X_SAME, Y_SAME = 1, 2, 4, 8, 16, 32


class TTF:
    def __init__(self, path):
        self.d = open(path, "rb").read()
        d = self.d
        off = 0
        if d[:4] == b"ttcf":
            off = struct.unpack(">I", d[12:16])[0]
        num = struct.unpack(">H", d[off + 4:off + 6])[0]
        self.tables = {}
        for i in range(num):
            p = off + 12 + i * 16
            tag = d[p:p + 4]
            o, ln = struct.unpack(">II", d[p + 8:p + 16])
            self.tables[tag] = (o, ln)
        head = self.tables[b"head"][0]
        self.upem = struct.unpack(">H", d[head + 18:head + 20])[0]
        self.loc_fmt = struct.unpack(">h", d[head + 50:head + 52])[0]
        maxp = self.tables[b"maxp"][0]
        self.nglyphs = struct.unpack(">H", d[maxp + 4:maxp + 6])[0]
        self._loca()
        self._cmap()
        self._hmtx()

    def _loca(self):
        o, ln = self.tables[b"loca"]
        d, n = self.d, self.nglyphs + 1
        if self.loc_fmt == 0:
            self.loca = [v * 2 for v in struct.unpack(">%dH" % n, d[o:o + 2 * n])]
        else:
            self.loca = list(struct.unpack(">%dI" % n, d[o:o + 4 * n]))

    def _hmtx(self):
        hhea = self.tables[b"hhea"][0]
        n = struct.unpack(">H", self.d[hhea + 34:hhea + 36])[0]
        o = self.tables[b"hmtx"][0]
        self.adv = []
        for i in range(n):
            self.adv.append(struct.unpack(">H", self.d[o + i * 4:o + i * 4 + 2])[0])

    def advance(self, gid):
        return self.adv[gid] if gid < len(self.adv) else (self.adv[-1] if self.adv else self.upem)

    def _cmap(self):
        d = self.d
        o = self.tables[b"cmap"][0]
        n = struct.unpack(">H", d[o + 2:o + 4])[0]
        best, best_score = None, -1
        for i in range(n):
            pid, eid, sub = struct.unpack(">HHI", d[o + 4 + i * 8:o + 12 + i * 8])
            fmt = struct.unpack(">H", d[o + sub:o + sub + 2])[0]
            score = {12: 3, 4: 2}.get(fmt, 0) + (1 if pid == 3 else 0)
            if score > best_score:
                best, best_score = (o + sub, fmt), score
        self.cmap = {}
        sub, fmt = best
        if fmt == 4:
            segX2 = struct.unpack(">H", d[sub + 6:sub + 8])[0]
            seg = segX2 // 2
            ends = struct.unpack(">%dH" % seg, d[sub + 14:sub + 14 + segX2])
            p = sub + 16 + segX2
            starts = struct.unpack(">%dH" % seg, d[p:p + segX2])
            p += segX2
            deltas = struct.unpack(">%dh" % seg, d[p:p + segX2])
            p += segX2
            ro_base = p
            ranges = struct.unpack(">%dH" % seg, d[p:p + segX2])
            for i in range(seg):
                for c in range(starts[i], min(ends[i], 0xFFFF) + 1):
                    if ranges[i] == 0:
                        g = (c + deltas[i]) & 0xFFFF
                    else:
                        gp = ro_base + i * 2 + ranges[i] + (c - starts[i]) * 2
                        g = struct.unpack(">H", d[gp:gp + 2])[0]
                        if g:
                            g = (g + deltas[i]) & 0xFFFF
                    if g:
                        self.cmap[c] = g
        elif fmt == 12:
            ngroups = struct.unpack(">I", d[sub + 12:sub + 16])[0]
            for i in range(ngroups):
                s, e, g = struct.unpack(">III", d[sub + 16 + i * 12:sub + 28 + i * 12])
                for c in range(s, e + 1):
                    self.cmap[c] = g + (c - s)

    # --- outlines ------------------------------------------------------
    def contours(self, gid, depth=0):
        """Return [[(x, y, on_curve), ...], ...] in font units."""
        if gid is None or gid + 1 >= len(self.loca):
            return []
        o = self.tables[b"glyf"][0] + self.loca[gid]
        end = self.tables[b"glyf"][0] + self.loca[gid + 1]
        if end <= o:
            return []
        d = self.d
        nc = struct.unpack(">h", d[o:o + 2])[0]
        if nc < 0:
            return self._composite(o + 10, depth)
        ends = struct.unpack(">%dH" % nc, d[o + 10:o + 10 + nc * 2])
        p = o + 10 + nc * 2
        ilen = struct.unpack(">H", d[p:p + 2])[0]
        p += 2 + ilen
        npts = (ends[-1] + 1) if nc else 0
        flags = []
        while len(flags) < npts:
            f = d[p]; p += 1
            flags.append(f)
            if f & REPEAT:
                r = d[p]; p += 1
                flags.extend([f] * r)
        flags = flags[:npts]
        xs, v = [], 0
        for f in flags:
            if f & X_SHORT:
                dx = d[p]; p += 1
                v += dx if f & X_SAME else -dx
            elif not f & X_SAME:
                v += struct.unpack(">h", d[p:p + 2])[0]; p += 2
            xs.append(v)
        ys, v = [], 0
        for f in flags:
            if f & Y_SHORT:
                dy = d[p]; p += 1
                v += dy if f & Y_SAME else -dy
            elif not f & Y_SAME:
                v += struct.unpack(">h", d[p:p + 2])[0]; p += 2
            ys.append(v)
        out, s = [], 0
        for e in ends:
            out.append([(xs[i], ys[i], bool(flags[i] & ON_CURVE))
                        for i in range(s, e + 1)])
            s = e + 1
        return [c for c in out if len(c) > 2]

    def _composite(self, p, depth):
        if depth > 4:
            return []
        d, out = self.d, []
        while True:
            flags, gi = struct.unpack(">HH", d[p:p + 4]); p += 4
            if flags & 1:
                a1, a2 = struct.unpack(">hh", d[p:p + 4]); p += 4
            else:
                a1, a2 = struct.unpack(">bb", d[p:p + 2]); p += 2
            xx = yy = 1.0
            xy = yx = 0.0
            if flags & 8:
                xx = yy = struct.unpack(">h", d[p:p + 2])[0] / 16384.0; p += 2
            elif flags & 0x40:
                xx = struct.unpack(">h", d[p:p + 2])[0] / 16384.0
                yy = struct.unpack(">h", d[p + 2:p + 4])[0] / 16384.0; p += 4
            elif flags & 0x80:
                xx, xy, yx, yy = (v / 16384.0 for v in
                                  struct.unpack(">hhhh", d[p:p + 8])); p += 8
            dx, dy = (a1, a2) if flags & 2 else (0, 0)
            for c in self.contours(gi, depth + 1):
                out.append([(x * xx + y * yx + dx, x * xy + y * yy + dy, on)
                            for x, y, on in c])
            if not flags & 0x20:
                break
        return out


def flatten(contour, upem):
    """TrueType quadratic contour -> polyline (list of (x, y) floats)."""
    pts = list(contour)
    # make the contour start on-curve
    start = next((i for i, p in enumerate(pts) if p[2]), None)
    if start is None:                     # all off-curve: synth midpoints
        x0 = (pts[0][0] + pts[-1][0]) / 2.0
        y0 = (pts[0][1] + pts[-1][1]) / 2.0
        pts = [(x0, y0, True)] + pts
        start = 0
    pts = pts[start:] + pts[:start]
    pts.append(pts[0])

    out = [(pts[0][0], pts[0][1])]
    i = 1
    cur = (pts[0][0], pts[0][1])
    tol = upem / 220.0
    while i < len(pts):
        x, y, on = pts[i]
        if on:
            out.append((x, y))
            cur = (x, y)
            i += 1
            continue
        # off-curve control; the next on-curve point (or implied midpoint)
        nx, ny, non = pts[i + 1] if i + 1 < len(pts) else (pts[0][0], pts[0][1], True)
        if not non:
            nx, ny = (x + nx) / 2.0, (y + ny) / 2.0
            step = 1
        else:
            step = 2
        d = max(abs(nx - cur[0]), abs(ny - cur[1]))
        n = max(2, min(10, int(d / tol) + 2))
        for k in range(1, n + 1):
            t = k / n
            mt = 1 - t
            out.append((mt * mt * cur[0] + 2 * mt * t * x + t * t * nx,
                        mt * mt * cur[1] + 2 * mt * t * y + t * t * ny))
        cur = (nx, ny)
        i += step
    return out


# --- varint encoding -------------------------------------------------------

def zz(v):
    return (v << 1) if v >= 0 else ((-v << 1) - 1)


def put(buf, v):
    while True:
        b = v & 0x7F
        v >>= 7
        buf.append(b | (0x80 if v else 0))
        if not v:
            return


def build(font, chars, scale=1000):
    f = TTF(font)
    blob = bytearray()
    kept, missing = [], []
    k = scale / f.upem
    for ch in chars:
        gid = f.cmap.get(ord(ch))
        cs = f.contours(gid) if gid else []
        polys = []
        for c in cs:
            poly = flatten(c, f.upem)
            q, last = [], (0, 0)
            for x, y in poly:
                qx, qy = int(round(x * k)), int(round(y * k))
                if q and qx == last[0] and qy == last[1]:
                    continue
                q.append((qx, qy))
                last = (qx, qy)
            if len(q) > 2:
                polys.append(q)
        if not polys:
            missing.append(ch)
            continue
        kept.append(ch)
        put(blob, ord(ch))
        put(blob, int(round(f.advance(gid) * k)))
        put(blob, len(polys))
        for q in polys:
            put(blob, len(q))
            px = py = 0
            for x, y in q:
                put(blob, zz(x - px))
                put(blob, zz(y - py))
                px, py = x, y
    head = bytearray()
    put(head, len(kept))                  # glyph count, known only at the end
    return kept, missing, bytes(head) + bytes(blob), scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("font")
    ap.add_argument("-o", "--out", default="gongbi_font.py")
    ap.add_argument("-m", "--module", default="gongbi",
                    help="module whose vocabulary() lists the characters to "
                         "bake (gongbi, shanshui, ...)")
    a = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    src = __import__(a.module)
    chars = sorted(set(src.vocabulary()))
    kept, missing, body, scale = build(a.font, chars)
    comp = base64.b85encode(zlib.compress(body, 9)).decode()
    lines = [comp[i:i + 78] for i in range(0, len(comp), 78)]
    name = os.path.basename(a.font)
    with open(a.out, "w") as fh:
        fh.write('# -*- coding: utf-8 -*-\n')
        fh.write('"""Auto-generated by tools/extract_glyphs.py — do not edit.\n\n')
        fh.write("Brush-calligraphy outlines (%d glyphs) for the inscriptions of\n"
                 "%s.py, taken from %s and quantised to a %d-unit em square.\n"
                 % (len(kept), a.module, name, scale))
        fh.write('"""\n\n')
        fh.write("SOURCE = %r\nEM = %d\nCHARS = %r\n\nDATA = (\n" % (name, scale, "".join(kept)))
        for ln in lines:
            fh.write('    "%s"\n' % ln)
        fh.write(")\n")
    print("glyphs: %d   missing: %s   blob: %.1f KiB -> %s"
          % (len(kept), "".join(missing) or "-", len(comp) / 1024.0, a.out))


if __name__ == "__main__":
    main()
