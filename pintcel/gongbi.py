#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gongbi.py — 随机生成中国工笔花卉图 / Random vivid Chinese Gongbi-style flower paintings.

Gongbi (工笔) is the meticulous court style of Chinese painting. This generator
imitates its actual production order:

    绢底   silk/paper ground with woven texture and age stains
    勾线   fine, even ink contour lines (铁线勾勒)
    分染   layered washes that deepen from the base (or rim) of every petal
    罩染   a unifying transparent glaze of one pigment
    点彩   opaque gouache accents: gold stamens, white powder tips
    题款   an inscription column and vermilion seals, placed in the 留白

Everything is procedural and stdlib-only: shapes are analytic signed fields
(no polygons), so edges are anti-aliased exactly and washes follow the petal
form the way a brush does. Output is a PNG written by hand via zlib.

Usage
-----
    python3 gongbi.py                         # one random painting
    python3 gongbi.py -n 6                    # six of them
    python3 gongbi.py --species peony --seed 7
    python3 gongbi.py --size 1400x1900 --out scroll.png
    python3 gongbi.py --list                  # show the flower repertoire
"""

import argparse
import base64
import math
import os
import random
import struct
import sys
import time
import zlib

# --------------------------------------------------------------------------
# 颜料 — traditional Chinese pigments, the whole palette this painter owns
# --------------------------------------------------------------------------

PIGMENT = {
    "mo":       "#2A2724",  # 墨      ink
    "danmo":    "#6E6A62",  # 淡墨    diluted ink
    "yanzhi":   "#9E2742",  # 胭脂    carmine lake
    "zhuhong":  "#CE3A24",  # 朱红    vermilion
    "zhupiao":  "#E4674A",  # 朱膘    light vermilion
    "tenghuang":"#EFB22B",  # 藤黄    gamboge
    "shihuang": "#D98E24",  # 石黄    orpiment
    "huaqing":  "#2D5574",  # 花青    indigo
    "shiqing":  "#3B6C9B",  # 石青    azurite
    "shilv":    "#4F7E5E",  # 石绿    malachite
    "caolv":    "#7FA052",  # 草绿    sap green
    "zheshi":   "#A3663C",  # 赭石    ochre
    "baifen":   "#FCF8EF",  # 白粉    lead white
    "jin":      "#C9A227",  # 金      gold
}


def rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def mix(a, b, t):
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def P(name):
    return rgb(PIGMENT[name])


def jitter(c, rng, amt=8.0):
    return tuple(min(255.0, max(0.0, v + rng.uniform(-amt, amt))) for v in c)


def clamp01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


# --- 配色 — colour maths in HSV ---------------------------------------------
#
# Interpolating a wash from a pale stop to a deep one in RGB drags the midtones
# through grey: pink -> crimson passes through a dead mauve. Real pigment keeps
# its chroma all the way down, so every ramp is built in HSV instead, taking
# the short way around the hue circle and bowing the saturation up in the
# middle. This is the single biggest reason the colour reads as pigment.


def to_hsv(c):
    r, g, b = c[0] / 255.0, c[1] / 255.0, c[2] / 255.0
    mx = r if r > g else g
    if b > mx:
        mx = b
    mn = r if r < g else g
    if b < mn:
        mn = b
    d = mx - mn
    if d < 1e-9:
        h = 0.0
    elif mx == r:
        h = (60.0 * ((g - b) / d)) % 360.0
    elif mx == g:
        h = 60.0 * ((b - r) / d) + 120.0
    else:
        h = 60.0 * ((r - g) / d) + 240.0
    return h, (0.0 if mx < 1e-9 else d / mx), mx


def from_hsv(h, s, v):
    h = h % 360.0
    s = clamp01(s)
    v = clamp01(v)
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    rv, gv, bv = ((c, x, 0.0), (x, c, 0.0), (0.0, c, x),
                  (0.0, x, c), (x, 0.0, c), (c, 0.0, x))[int(h / 60.0) % 6]
    return ((rv + m) * 255.0, (gv + m) * 255.0, (bv + m) * 255.0)


def hue_lerp(h0, h1, t):
    """Interpolate hue the short way round the circle."""
    return (h0 + ((h1 - h0 + 540.0) % 360.0 - 180.0) * t) % 360.0


def hsv_mix(a, b, t, bow=0.0):
    """Blend two colours through HSV, optionally bowing saturation outward."""
    t = clamp01(t)
    h0, s0, v0 = to_hsv(a)
    h1, s1, v1 = to_hsv(b)
    if s0 < 0.04:                      # a neutral has no hue to travel from
        h0 = h1
    if s1 < 0.04:
        h1 = h0
    s = s0 + (s1 - s0) * t
    if bow:
        s += bow * math.sin(math.pi * t) * (1.0 - s)
    return from_hsv(hue_lerp(h0, h1, t), s, v0 + (v1 - v0) * t)


def hue_rotate(c, deg, sat=1.0, val=1.0):
    h, s, v = to_hsv(c)
    return from_hsv(h + deg, s * sat, v * val)


def saturate(c, k):
    """Push a colour's chroma; k > 1 enriches, k < 1 greys it out."""
    h, s, v = to_hsv(c)
    return from_hsv(h, s * k, v)


# --- 随机 — distributions ---------------------------------------------------

def bell(rng, lo, hi, n=3):
    """Lo..hi with most draws near the middle; a flower is mostly typical
    with a few outliers, which uniform jitter never gives you."""
    s = 0.0
    for _ in range(n):
        s += rng.random()
    return lo + (hi - lo) * s / n


# --------------------------------------------------------------------------
# Canvas — plain RGB bytearray plus a hand-rolled PNG writer
# --------------------------------------------------------------------------

class Canvas:
    def __init__(self, w, h, bg=(240, 232, 212)):
        self.w, self.h = w, h
        self.buf = bytearray(w * h * 3)
        r, g, b = (int(v) for v in bg)
        self.buf[0::3] = bytes([r]) * (w * h)
        self.buf[1::3] = bytes([g]) * (w * h)
        self.buf[2::3] = bytes([b]) * (w * h)

    def clip(self, x0, y0, x1, y1):
        return (max(0, int(x0)), max(0, int(y0)),
                min(self.w, int(x1) + 1), min(self.h, int(y1) + 1))

    def blend(self, x, y, c, a):
        if a <= 0.0 or x < 0 or y < 0 or x >= self.w or y >= self.h:
            return
        if a > 1.0:
            a = 1.0
        i = (y * self.w + x) * 3
        buf = self.buf
        buf[i] = int(buf[i] + (c[0] - buf[i]) * a)
        buf[i + 1] = int(buf[i + 1] + (c[1] - buf[i + 1]) * a)
        buf[i + 2] = int(buf[i + 2] + (c[2] - buf[i + 2]) * a)

    def write_png(self, path):
        w, h, buf = self.w, self.h, self.buf
        stride = w * 3
        raw = bytearray()
        for y in range(h):
            raw.append(0)                       # filter type 0
            raw += buf[y * stride:(y + 1) * stride]
        comp = zlib.compress(bytes(raw), 6)

        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", comp)
               + chunk(b"IEND", b""))
        with open(path, "wb") as f:
            f.write(png)


# --------------------------------------------------------------------------
# 噪点 — cheap value noise (lattice + bilinear) for silk grain and stains
# --------------------------------------------------------------------------

class Noise:
    def __init__(self, rng, cells_x, cells_y):
        self.cx, self.cy = cells_x, cells_y
        self.g = [[rng.random() for _ in range(cells_x + 2)]
                  for _ in range(cells_y + 2)]

    def at(self, fx, fy):
        """fx, fy in [0,1]."""
        x = fx * self.cx
        y = fy * self.cy
        ix, iy = int(x), int(y)
        tx, ty = x - ix, y - iy
        tx = tx * tx * (3 - 2 * tx)
        ty = ty * ty * (3 - 2 * ty)
        g = self.g
        a = g[iy][ix] + (g[iy][ix + 1] - g[iy][ix]) * tx
        b = g[iy + 1][ix] + (g[iy + 1][ix + 1] - g[iy + 1][ix]) * tx
        return a + (b - a) * ty


# --------------------------------------------------------------------------
# 绢底 — the ground: silk tone, weave, fibres, foxing stains, vignette
# --------------------------------------------------------------------------

GROUNDS = [
    ("老绢", "#E9DCBE", "#D8C69F"),   # aged silk
    ("蝉翼宣", "#F1E7D2", "#E2D3B4"),  # thin xuan paper
    ("米黄绢", "#EADFC4", "#D5C49B"),
    ("淡茶", "#E3D6B9", "#CBB890"),
    ("霜白宣", "#F4EEDE", "#E3D9C1"),
]


def paint_ground(cv, rng):
    name, lo, hi = GROUNDS[rng.randrange(len(GROUNDS))]
    c_lo, c_hi = rgb(lo), rgb(hi)
    w, h = cv.w, cv.h
    blotch = Noise(rng, 5, 7)
    grain = Noise(rng, max(8, w // 26), max(8, h // 26))
    stain = Noise(rng, 3, 4)
    stain_c = rgb("#B99C6A")

    # Pre-tabulate the silk weave so the inner loop stays multiplication-only.
    wx = [1.0 + 0.016 * math.sin(x * 1.9 + 0.4) for x in range(w)]
    wy = [1.0 + 0.016 * math.sin(y * 2.1) for y in range(h)]
    cxm, cym = w * 0.5, h * 0.5
    inv_r = 1.0 / math.hypot(cxm, cym)

    buf = cv.buf
    i = 0
    for y in range(h):
        fy = y / h
        wyy = wy[y]
        dy2 = (y - cym) ** 2
        for x in range(w):
            fx = x / w
            t = blotch.at(fx, fy) * 0.8 + grain.at(fx, fy) * 0.2
            r, g, b = mix(c_lo, c_hi, t)
            s = stain.at(fx, fy)
            if s > 0.70:                       # faint foxing / tea stains
                k = (s - 0.70) * 0.34
                r, g, b = mix((r, g, b), stain_c, k)
            f = wyy * wx[x]
            v = 1.0 - 0.15 * (math.sqrt(dy2 + (x - cxm) ** 2) * inv_r) ** 3
            f *= v                             # soft vignette toward the edge
            buf[i] = int(min(255.0, r * f))
            buf[i + 1] = int(min(255.0, g * f))
            buf[i + 2] = int(min(255.0, b * f))
            i += 3
    return name


# --------------------------------------------------------------------------
# 叶瓣 — the one renderer that draws petals, leaves, sepals, stems, wings.
#
# A "blade" is defined analytically in its own frame: u runs 0..1 from the
# anchor along the axis, and the silhouette is |v| <= hw(u). Washes, veins and
# the contour line are all functions of (u, v), so the shading follows the
# form exactly and the edge is anti-aliased without supersampling.
# --------------------------------------------------------------------------

NU = 512          # resolution of the u lookup tables
NT = 192          # resolution of the across-the-blade tables
NS = 96           # resolution of the light->deep colour ramp
SH0 = -0.34       # shade below 0 means "lit": the ramp runs past the pale stop
SHK = NS / (1.0 - SH0)

# 光 — one light direction per painting, so every form models the same way.
LIGHT = (-0.64, -0.77)


def set_light(rng):
    """Pick the painting's light. Gongbi lights from above, usually off to one
    side; the whole scene must agree or nothing reads as solid."""
    global LIGHT
    a = rng.uniform(-2.5, -0.65)          # upper-left .. upper-right
    LIGHT = (math.cos(a), math.sin(a))
    return a


class Blade:
    __slots__ = ("cx", "cy", "ang", "L", "W", "k1", "e1", "k2", "e2",
                 "edge", "eamp", "efreq", "ephase", "tip",
                 "bend", "notch", "stem_taper", "tuck",
                 "light", "deep", "ink", "line", "alpha",
                 "w_base", "p_base", "w_tip", "p_tip", "w_rim", "p_rim",
                 "w_mid", "p_mid", "hl", "hl_tip", "veins", "vein_n",
                 "vein_fan", "vein_ink", "midrib", "mode", "caps",
                 "rim_c", "w_rimc", "p_rimc", "soft",
                 "lit", "wash", "bow", "blend", "wob")

    def __init__(self, **kw):
        d = dict(
            cx=0.0, cy=0.0, ang=0.0, L=100.0, W=30.0,
            k1=2.4, e1=0.55, k2=4.0, e2=0.55,
            # edge treatment: plain | wave (ruffled petal) | serrate (teeth)
            #                 | lobe (deep palmate clefts)
            edge="plain", eamp=0.0, efreq=6.0, ephase=0.0,
            tip=0.0,                     # drawn-out acuminate point
            bend=0.0, notch=0.0, stem_taper=0.5,
            tuck=1.0,                    # close the two ends to a point

            light=(250, 240, 240), deep=(170, 40, 70), ink=P("mo"),
            line=0.55, alpha=1.0,
            w_base=0.85, p_base=1.9, w_tip=0.0, p_tip=2.0,
            w_rim=0.18, p_rim=3.0, w_mid=0.0, p_mid=2.0,
            hl=0.0, hl_tip=0.0,
            veins=0.0, vein_n=7.0, vein_fan=1.5, vein_ink=None,
            midrib=0.0, mode="blade", caps=True,
            rim_c=None, w_rimc=0.0, p_rimc=4.0,   # 复色: a second pigment
            soft=0.0,       # 没骨: widen the edge ramp into a wash, in px
            lit=0.0,        # directional light across the form (see render)
            wash="power",   # power | sigmoid (水口, a crisp wash boundary)
            bow=0.12,       # saturation bow through the middle of the ramp
            blend="over",   # over | mul (ink-on-silk, the ground shows through)
            wob=1.0,        # 手不是尺: scale of the width wobble (see render).
                            # A single fast 写意 stroke — a bamboo leaf — wants
                            # this near zero: its edge is clean by nature, and
                            # the wobble on a blade only eight pixels wide
                            # reads as serration rather than as a hand.
        )
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)

    # ---- silhouette half-width at u -------------------------------------
    def _hw(self, u):
        if self.mode == "stem":
            base = 1.0 - self.stem_taper * u
        else:
            a = 1.0 - (1.0 - u) ** self.k1
            b = 1.0 - u ** self.k2
            if a < 0.0:
                a = 0.0
            if b < 0.0:
                b = 0.0
            base = (a ** self.e1) * (b ** self.e2)
            if self.tip > 0.0:               # taper the last stretch to a point
                z = (1.0 - u) / self.tip
                if z < 1.0:
                    base *= z ** 0.7
        e = self.edge
        if e != "plain" and self.eamp:
            ph = self.efreq * u + self.ephase
            if e == "wave":
                base *= 1.0 + self.eamp * math.sin(math.pi * ph)
            elif e == "serrate":             # sawtooth teeth leaning tipward
                base *= 1.0 + self.eamp * ((ph - math.floor(ph)) - 0.45)
            elif e == "lobe":                # narrow clefts between broad lobes
                # The exponent decides which half of the sine is sharp. >1
                # keeps the term near zero for most of the period, so the
                # width only pinches in a narrow band: a round-shouldered
                # palmate lobe. <1 inverts it and gives holly spikes.
                s = abs(math.sin(math.pi * ph))
                base *= 1.0 - self.eamp * (s ** 2.2)
        return self.W * (base if base > 0.0 else 0.0)

    def render(self, cv):
        L, W = self.L, self.W
        if L < 2.0 or W < 0.6:
            return
        ca, sa = math.cos(self.ang), math.sin(self.ang)

        # --- 收根收尖 — close the two ends to a point ---------------------
        # The profile opens as u**e1 with e1 < 1, i.e. with an *infinite*
        # slope at u = 0: half a pixel in from the anchor the silhouette is
        # already four or five pixels wide, and the base cap down in the pixel
        # loop slices that across square. On an opaque near leaf the branch
        # covers the stub; on a pale 远叶 it survives as a ten-pixel ruled
        # line floating in mid-air — the straightest mark in the painting and
        # the only one nobody drew. The tip does the same thing more quietly
        # (b**e2 with e2 < 1, then the tip cap). Ramping the first and last
        # few *pixels* — not a fixed fraction of u, the defect is absolute —
        # turns both cuts into cusps, and costs nothing: it happens here in
        # the table, never in the inner loop.
        # A stalk never ramps its *root*: that end is the hinge on the wood and
        # has to meet the bark at full width or the leaf looks unglued. Its tip
        # does ramp, so the stalk ends in a point under the base of its own
        # blade rather than in a blunt little rectangle — which is exactly what
        # a pale 远叶 petiole looked like, a dark dash lying beside its leaf.
        # Bark opts out of both by passing tuck=0: consecutive segments share a
        # width at the joint, and a cusp there would pinch the branch in two.
        root = min(0.11, 7.0 / L) * (0.0 if self.mode == "stem" else self.tuck)
        peak = min(0.05, 2.6 / L) * self.tuck if not self.notch else 0.0

        # --- 手不是尺 — a hand does not draw a straight line --------------
        # Over the tapering third of a blade the half-width falls almost
        # exactly linearly, so the silhouette there is the *side of a cone*:
        # mathematically ruled, and — depending on where the rotation lands —
        # sometimes exactly horizontal or exactly vertical on screen for
        # twenty pixels at a stretch. Nothing else in the painting is that
        # straight, so the eye finds it instantly; on a pale 远叶, where the
        # contour is all it has to go on, it reads as a pencil line left in
        # by mistake. The analytic edge is *too* good.
        # The cure is to put the hand back: two incommensurate sines, of
        # wavelength about nine and twenty pixels, breathe the half-width by
        # half a pixel. Phased off the blade's own placement, so it differs
        # between blades and is identical between renders of the same seed.
        # The amplitude has to be mostly *absolute*: a purely proportional
        # wobble dies away exactly where the blade is thin, which is where
        # the cone sides — and so the ruled stretches — actually are. It is
        # applied before the end ramps below, so the two ends still close
        # cleanly instead of fraying. Branches are excluded: consecutive bark
        # segments share a width at the joint and a wobble would step there.
        if self.mode == "stem" or self.wob <= 0.0:
            wob_a = wob_r = 0.0
            wf1 = wf2 = wp1 = wp2 = 0.0
        else:
            wob_a = 0.50 * self.wob
            wob_r = 0.009 * self.wob
            wf1 = 0.72 * L
            wf2 = 0.31 * L
            wp1 = (self.cx * 0.7311 + self.cy * 0.3137) % 6.2832
            wp2 = (self.cx * 0.1733 - self.cy * 0.9173 + self.ang * 3.1) % 6.2832

        # --- lookup tables over u ---------------------------------------
        hw_t = [0.0] * (NU + 1)
        bd_t = [0.0] * (NU + 1)
        su_t = [0.0] * (NU + 1)
        hu_t = [0.0] * (NU + 1)
        hwmax = 0.0
        bdmax = 0.0
        for i in range(NU + 1):
            u = i / NU
            hw = self._hw(u)
            if wob_a:
                hw += (wob_a + wob_r * hw) * (math.sin(wf1 * u + wp1)
                                              + 0.62 * math.sin(wf2 * u + wp2))
                if hw < 0.0:
                    hw = 0.0
            if u < root:
                hw *= u / root
            elif u > 1.0 - peak:
                hw *= (1.0 - u) / peak
            hw_t[i] = hw
            if hw > hwmax:
                hwmax = hw
            bd = self.bend * L * (u * u)
            bd_t[i] = bd
            if abs(bd) > bdmax:
                bdmax = abs(bd)
            s = 0.0
            if self.w_base:
                s += self.w_base * (1.0 - u) ** self.p_base
            if self.w_tip:
                s += self.w_tip * u ** self.p_tip
            su_t[i] = s
            hu_t[i] = (self.hl * (1.0 - (1.0 - u) ** 1.2)
                       + self.hl_tip * u ** 2.2)

        # --- lookup tables over |v|/hw ----------------------------------
        st_t = [0.0] * (NT + 1)
        rc_t = [0.0] * (NT + 1)
        for j in range(NT + 1):
            t = j / NT
            s = 0.0
            if self.w_rim:
                s += self.w_rim * t ** self.p_rim
            if self.w_mid:
                s += self.w_mid * (1.0 - t) ** self.p_mid
            st_t[j] = s
            rc_t[j] = self.w_rimc * t ** self.p_rimc if self.w_rimc else 0.0

        # --- bounding box ------------------------------------------------
        pad = 3.0
        ext = hwmax + bdmax + pad
        xs, ys = [], []
        for ux, vy in ((-pad, -ext), (L + pad, -ext), (L + pad, ext), (-pad, ext)):
            xs.append(self.cx + ux * ca - vy * sa)
            ys.append(self.cy + ux * sa + vy * ca)
        x0, y0, x1, y1 = cv.clip(min(xs), min(ys), max(xs), max(ys))
        if x0 >= x1 or y0 >= y1:
            return

        light, deep, ink = self.light, self.deep, self.ink
        lr, lg, lb = light
        ir, ig, ib = ink
        white = P("baifen")
        wr, wg, wb = white[0] - lr, white[1] - lg, white[2] - lb
        rim_c = self.rim_c if self.w_rimc else None

        # --- 分染 colour ramp, built in HSV -----------------------------
        # One table maps accumulated shade -> pigment, so the inner loop does
        # a lookup instead of three lerps: faster *and* far richer, because
        # the interpolation happens in HSV (see hsv_mix). Indices below the
        # pale stop are the lit flank, carried on up into 白粉.
        sig = self.wash == "sigmoid"
        bow = self.bow
        cr_t = [0.0] * (NS + 1)
        cg_t = [0.0] * (NS + 1)
        cb_t = [0.0] * (NS + 1)
        for j in range(NS + 1):
            sh = SH0 + j / SHK
            if sh < 0.0:                      # 提亮 — the side facing the light
                c = hsv_mix(light, white, (sh / SH0) * 0.72)
            else:
                t = sh
                if sig:                       # 水口: a crisp wash boundary
                    t = 1.0 / (1.0 + math.exp(-7.0 * (t - 0.42)))
                    t = clamp01((t - 0.0503) / 0.9329)
                c = hsv_mix(light, deep, t, bow=bow)
            cr_t[j], cg_t[j], cb_t[j] = c
        if rim_c:
            rcr, rcg, rcb = rim_c
        line_k = self.line
        alpha = self.alpha
        notch = self.notch * L
        veins, vein_n, vein_fan = self.veins, self.vein_n, self.vein_fan
        radial = vein_fan < 0.0
        vink = self.vein_ink or deep
        vr, vg, vb = vink
        midrib = self.midrib
        caps = self.caps
        # 远: a soft edge is not a blurred edge, it is a *wash*. An analytic
        # one-pixel edge is a cut-out, and a dozen pale cut-outs stacked on
        # silk read as torn paper — every chord of every silhouette becomes a
        # visible line, and the straight ones become ruled ones. Spreading the
        # coverage ramp over a couple of pixels costs one multiply per pixel
        # and turns the same shape into something the ground shows through.
        isf = 1.0 / (1.0 + 2.0 * self.soft)
        dmin = -1.2 if self.soft <= 0.0 else -0.5 / isf - 0.3
        lw = 1.15 + W * 0.012          # contour line weight, 勾线
        inv_lw = 1.0 / lw
        buf = cv.buf
        cw = cv.w
        cx, cy = self.cx, self.cy
        Lx = L + 1.5
        mul = self.blend == "mul"
        # 光 — the blade's local +v axis points along (-sin, cos) in the world,
        # so this is how much that flank faces the light. A petal is not flat:
        # treating its cross-section as a cylinder and signing the shade by v
        # is what turns a decorative silhouette into something with volume.
        lk = -self.lit * (LIGHT[0] * -sa + LIGHT[1] * ca)

        for py in range(y0, y1):
            dy = py + 0.5 - cy
            row = py * cw
            dyc, dys = dy * ca, dy * sa
            for px in range(x0, x1):
                dx = px + 0.5 - cx
                x = dx * ca + dys                    # rotate by -ang
                if x < -1.5 or x > Lx:
                    continue
                y = dyc - dx * sa
                u = x / L
                if u < 0.0:
                    u = 0.0
                elif u > 1.0:
                    u = 1.0
                iu = int(u * NU)
                hw = hw_t[iu]
                if hw <= 0.0:
                    continue
                yv = y - bd_t[iu]
                ay = yv if yv >= 0.0 else -yv
                d = hw - ay
                if d < dmin:
                    continue
                if caps:
                    t_n = ay / hw
                    lim = L - notch * (1.0 - t_n * t_n) if notch else L
                    d2 = lim - x
                    if d2 < d:
                        d = d2
                    if x < d:
                        d = x
                    if d < dmin:
                        continue
                cov = d * isf + 0.5
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                a = cov * alpha
                if a <= 0.004:
                    continue

                ts = yv / hw                 # signed: which flank are we on
                if ts > 1.0:
                    ts = 1.0
                elif ts < -1.0:
                    ts = -1.0
                tt = ts if ts >= 0.0 else -ts
                it = int(tt * NT)
                sh = su_t[iu] + st_t[it] + lk * ts
                if sh > 1.0:
                    sh = 1.0
                elif sh < SH0:
                    sh = SH0
                isx = int((sh - SH0) * SHK)
                r = cr_t[isx]
                g = cg_t[isx]
                b = cb_t[isx]

                if rim_c:                    # 复色: ochre / carmine rim wash
                    k = rc_t[it]
                    if k > 0.0:
                        r += (rcr - r) * k
                        g += (rcg - g) * k
                        b += (rcb - b) * k

                hlv = hu_t[iu] * (1.0 - tt * 0.55)
                if hlv > 0.0:
                    r += wr * hlv
                    g += wg * hlv
                    b += wb * hlv

                if veins:
                    if midrib and tt < midrib:
                        k = veins * (1.0 - tt / midrib) * 0.9
                        r += (vr - r) * k
                        g += (vg - g) * k
                        b += (vb - b) * k
                    # vein_fan < 0 is the radial family: lines of constant
                    # t/u leave the petiole like the ribs of a 荷叶. The
                    # parallel family below skews toward the tip instead.
                    s = (tt / (u + 0.08) * vein_n if radial
                         else (u - 0.05) * vein_n - tt * vein_fan)
                    if s > 0.0:
                        fr = s - int(s)
                        if fr > 0.88:
                            k = veins * (fr - 0.88) * 7.0 * (1.0 - tt * 0.3)
                            r += (vr - r) * k
                            g += (vg - g) * k
                            b += (vb - b) * k

                if line_k and d < lw:
                    k = (1.0 - d * inv_lw)
                    k = k * k * line_k
                    r += (ir - r) * k
                    g += (ig - g) * k
                    b += (ib - b) * k

                i = (row + px) * 3
                if mul:
                    # 没骨 — multiply instead of cover, so the silk weave and
                    # ground tone keep showing through foliage and bark the
                    # way a transparent wash does on real paper.
                    o = buf[i]
                    buf[i] = int(o + (o * r * 0.00392 - o) * a)
                    i += 1
                    o = buf[i]
                    buf[i] = int(o + (o * g * 0.00392 - o) * a)
                    i += 1
                    o = buf[i]
                    buf[i] = int(o + (o * b * 0.00392 - o) * a)
                else:
                    o = buf[i]
                    buf[i] = int(o + (r - o) * a)
                    i += 1
                    o = buf[i]
                    buf[i] = int(o + (g - o) * a)
                    i += 1
                    o = buf[i]
                    buf[i] = int(o + (b - o) * a)


# --------------------------------------------------------------------------
# small primitives: discs (stamen dots, moss, berries) and brush segments
# --------------------------------------------------------------------------

def disc(cv, cx, cy, r, color, alpha=1.0, squash=1.0, soft=0.0,
         edge=None, edge_k=0.0):
    if r < 0.3:
        return
    ry = r * squash
    x0, y0, x1, y1 = cv.clip(cx - r - 2, cy - ry - 2, cx + r + 2, cy + ry + 2)
    inv = 1.0 / r
    invy = 1.0 / (ry if ry > 0.01 else 0.01)
    for py in range(y0, y1):
        ny = (py + 0.5 - cy) * invy
        for px in range(x0, x1):
            nx = (px + 0.5 - cx) * inv
            dn = math.sqrt(nx * nx + ny * ny)
            d = (1.0 - dn) * r
            cov = d + 0.5
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            c = color
            if soft:
                c = mix(color, c, 0.0)
            if edge and edge_k and dn > 0.45:
                c = mix(c, edge, min(1.0, (dn - 0.45) * edge_k))
            if soft:
                cov *= 1.0 - soft * dn * dn
            cv.blend(px, py, c, cov * alpha)


def brush_seg(cv, x1, y1, x2, y2, w1, w2, color, alpha=1.0):
    """Short tapered stroke — filaments, veins, calligraphic marks."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 0.4:
        disc(cv, x1, y1, max(w1, 0.5), color, alpha)
        return
    ux, uy = dx / L, dy / L
    wmax = max(w1, w2)
    x0, yy0, xx1, yy1 = cv.clip(min(x1, x2) - wmax - 2, min(y1, y2) - wmax - 2,
                                max(x1, x2) + wmax + 2, max(y1, y2) + wmax + 2)
    for py in range(yy0, yy1):
        for px in range(x0, xx1):
            vx, vy = px + 0.5 - x1, py + 0.5 - y1
            t = (vx * ux + vy * uy) / L
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            qx, qy = x1 + dx * t, y1 + dy * t
            dist = math.hypot(px + 0.5 - qx, py + 0.5 - qy)
            w = w1 + (w2 - w1) * t
            cov = (w - dist) + 0.5
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            cv.blend(px, py, color, cov * alpha)


def brush_path(cv, pts, w_start, w_end, color, alpha=1.0):
    n = len(pts) - 1
    if n < 1:
        return
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        brush_seg(cv, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                  w_start + (w_end - w_start) * t0,
                  w_start + (w_end - w_start) * t1, color, alpha)


# --------------------------------------------------------------------------
# curves —枝干 skeletons
# --------------------------------------------------------------------------

def bezier(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        a, b, c, d = mt ** 3, 3 * mt * mt * t, 3 * mt * t * t, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


def tangent(pts, i):
    j = min(len(pts) - 1, i + 1)
    k = max(0, i - 1)
    return math.atan2(pts[j][1] - pts[k][1], pts[j][0] - pts[k][0])


def knuckle(rng, pts, k=3, amp=0.18, start=0.14):
    """节 — kink a path at a few joints by rotating everything beyond them.

    A branch swept out as one bezier reads as a pipe. Real wood changes
    direction at nodes, so the gesture survives but the line stops being
    smooth. (The trick is borrowed from the joint model in `nonflowers`,
    where branch width and bend are both driven off distance to a joint.)
    """
    pts = list(pts)
    n = len(pts)
    if n < 6 or k < 1:
        return pts, []
    lo = max(1, int(n * start))
    joints = []
    for _ in range(k):
        j = rng.randrange(lo, n - 1)
        th = rng.uniform(0.35, 1.0) * amp * rng.choice((1.0, -1.0))
        ox, oy = pts[j]
        ca, sa = math.cos(th), math.sin(th)
        for i in range(j, n):
            dx, dy = pts[i][0] - ox, pts[i][1] - oy
            pts[i] = (ox + dx * ca - dy * sa, oy + dx * sa + dy * ca)
        joints.append(j)
    return pts, sorted(joints)


def draw_branch(cv, rng, pts, w0, w1, species):
    """Woody stem: butt-jointed tapered blades so one continuous contour."""
    bark_l = jitter(saturate(mix(P("zheshi"), P("baifen"), 0.20), 1.18), rng, 6)
    bark_d = jitter(mix(P("zheshi"), P("mo"), 0.72), rng, 6)
    n = len(pts) - 1

    # 节 — swell the wood where the path turns. Taking the bulge straight off
    # the local curvature means it always lands on the knuckles knuckle()
    # made; smoothing it afterwards is essential, or the sampling noise in the
    # curvature scallops the silhouette into something like bamboo.
    bg = [1.0] * (n + 1)
    for i in range(1, n):
        a0 = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
        a1 = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        d = abs((a1 - a0 + math.pi) % math.tau - math.pi)
        bg[i] = 1.0 + 0.55 * (d if d < 0.6 else 0.6)
    for _ in range(3):                       # 1-2-1 smoothing
        bg = [bg[0]] + [(bg[i - 1] + 2.0 * bg[i] + bg[i + 1]) * 0.25
                        for i in range(1, n)] + [bg[n]]

    def bulge(i):
        return bg[i] if 0 <= i <= n else 1.0

    for i in range(n):
        t = i / n
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if seg < 0.6:
            continue
        w_a = (w0 + (w1 - w0) * t) * bulge(i)
        w_b = (w0 + (w1 - w0) * ((i + 1) / n)) * bulge(i + 1)
        Blade(cx=ax, cy=ay, ang=math.atan2(by - ay, bx - ax),
              L=seg * 1.06, W=w_a, mode="stem", tuck=0.0,
              stem_taper=max(-0.6, min(0.95, 1.0 - w_b / max(w_a, 0.01))),
              light=bark_l, deep=bark_d,
              # cylinder shading: the lit flank is picked up from the global
              # light, so bark rounds off the same way the petals do.
              w_base=0.0, w_rim=0.58, p_rim=1.5, w_mid=0.0,
              lit=0.72, bow=0.16,
              # NB: bark must *not* multiply. Consecutive segments overlap by
              # 6% so the joint is covered, and a multiplying wash darkens
              # every overlap — which banded the branch like a bamboo cane.
              blend="over",
              line=0.0, caps=False,
              ).render(cv)
    # 勾线 — one continuous contour down each flank. Offsetting each segment
    # on its own normal leaves a visible tick at every joint; averaging the
    # neighbouring tangents (as `tubify` does in nonflowers) does not.
    for s in (1.0, -1.0):
        side = []
        for i in range(n + 1):
            a = tangent(pts, i)
            w = (w0 + (w1 - w0) * (i / n)) * bulge(i) * s
            side.append((pts[i][0] - math.sin(a) * w,
                         pts[i][1] + math.cos(a) * w))
        brush_path(cv, side, 0.8, 0.6, mix(bark_d, P("mo"), 0.5), 0.5)
    # 皴 — long dry drag-strokes along the grain, so bark has a surface.
    #
    # These have to be dragged *along* the curve, each sample offset on its own
    # normal and by its own share of the local half-width. Drawn as one chord
    # from pts[i] to pts[j] with a single offset vector taken at i — which is
    # what this was — the stroke is a straight line across a bend: it starts on
    # the bark, leaves it at the first node, and finishes anywhere up to twenty
    # pixels out in bare paper. Spanning four to nine nodes it only needs a
    # gentle bend to escape, and on a large canvas, where the nodes are further
    # apart, it escapes every time. That is where the faint ruled hairlines
    # running alongside every branch came from: 皴 strokes thrown clear of the
    # wood they were meant to texture, a 0.5 px line at alpha 0.2 — too pale to
    # read as a stroke, straight enough to read as a construction line, and
    # several of them stacked near-parallel, which is exactly what a pencil
    # under-drawing looks like. Following the curve keeps them on the bark,
    # where a dry drag belongs, and the fraction of the half-width keeps them
    # there through a taper too.
    for _ in range(int(n * rng.uniform(0.22, 0.5))):
        i = rng.randrange(max(1, n - 3))
        j = min(n, i + rng.randint(4, 9))
        f = rng.uniform(-0.55, 0.55)
        drag = []
        for t in range(i, j + 1):
            a = tangent(pts, t) + math.pi / 2
            o = f * (w0 + (w1 - w0) * (t / n)) * bulge(t)
            drag.append((pts[t][0] + math.cos(a) * o,
                         pts[t][1] + math.sin(a) * o))
        brush_path(cv, drag, rng.uniform(0.35, 0.7), rng.uniform(0.35, 0.7),
                   mix(bark_d, P("mo"), 0.35), rng.uniform(0.12, 0.28))
    # 苔点 — moss dots and bark texture
    moss = mix(P("shilv"), P("mo"), 0.35)
    for _ in range(int(len(pts) * rng.uniform(0.6, 1.6))):
        i = rng.randrange(len(pts))
        a = tangent(pts, i) + math.pi / 2
        o = rng.uniform(-0.8, 0.8) * (w0 + (w1 - w0) * i / n)
        r = rng.uniform(1.0, 3.0)
        c = moss if rng.random() < 0.6 else P("mo")
        disc(cv, pts[i][0] + math.cos(a) * o, pts[i][1] + math.sin(a) * o,
             r, c, rng.uniform(0.3, 0.7))


# --------------------------------------------------------------------------
# 花 — species presets.  Each entry is the painter's recipe for one flower.
# --------------------------------------------------------------------------

def _pair(light, deep, rim=None):
    """A flower's recipe: the pale first wash, the deep 分染 pigment, and an
    optional second pigment laid along the rim (复色)."""
    return (rgb(light), rgb(deep), rgb(rim) if rim else None)


SPECIES = {
    "peony": dict(
        cn="牡丹", season="春",
        rings=[(9, 1.00, 0.95), (8, 0.86, 0.66), (7, 0.70, 0.44),
               (6, 0.54, 0.27), (5, 0.40, 0.13)],
        petal=dict(k1=2.3, e1=0.55, k2=3.4, e2=0.56, wr=0.78,
                   edge="wave", eamp=0.10, efreq=5.0, notch=0.10, bend=0.34),
        colors=[                                     # 魏紫 赵粉 朱砂 姚黄 玉楼春
            _pair("#EEB6D0", "#78144A", "#B22A66"),
            _pair("#F8C2D0", "#AA1C4C", "#D44F79"),
            _pair("#F5B9A6", "#8E120B", "#CA3015"),
            _pair("#FFE49A", "#B86C04", "#E09616"),
            _pair("#FFF7EC", "#AC4763", "#D27E94"),
        ],
        shade=dict(w_base=1.15, p_base=1.10, w_rim=0.34, p_rim=2.4,
                   w_rimc=0.60, p_rimc=2.4, hl_tip=0.56, veins=0.26,
                   vein_n=9.0),
        center=dict(kind="tuft", n=30, color="jin", r=0.20),
        leaf=dict(n=(5, 8), L=(0.70, 1.00), wr=0.36, form="trifoliate",
                  edge="lobe", eamp=0.16, efreq=4.0, tip=0.22),
        branch=dict(w=(9.0, 3.5), style="smooth"),
        size=1.00, blossoms=(1, 3), buds=(1, 3),
    ),
    "lotus": dict(
        cn="荷花", season="夏",
        rings=[(8, 1.00, 0.88), (7, 0.80, 0.58), (6, 0.58, 0.32)],
        petal=dict(k1=2.0, e1=0.60, k2=1.8, e2=0.95, wr=0.42,
                   edge="wave", eamp=0.04, efreq=4.0, notch=0.0, bend=0.22),
        colors=[_pair("#FEE7E6", "#B82553", "#DB4F76"),
                _pair("#FBD7E0", "#96134A", "#C33668"),
                _pair("#FFF6DE", "#BE9426", "#DBB450")],
        shade=dict(w_base=0.30, p_base=2.2, w_tip=1.05, p_tip=1.8,
                   w_rim=0.30, p_rim=2.2, w_rimc=0.50, p_rimc=2.4, hl=0.34,
                   veins=0.46, vein_n=0.0, vein_fan=0.0),
        center=dict(kind="pod", n=22, color="tenghuang", r=0.26),
        leaf=dict(n=(2, 4), L=(0.85, 1.30), wr=1.00, form="round",
                  edge="wave", eamp=0.08, efreq=8.0, tip=0.0),
        branch=dict(w=(6.0, 4.0), style="reed"),
        size=1.12, blossoms=(1, 2), buds=(1, 2),
    ),
    "plum": dict(
        cn="梅花", season="冬",
        rings=[(5, 1.00, 0.52)],
        petal=dict(k1=2.6, e1=0.50, k2=7.0, e2=0.42, wr=0.95,
                   edge="wave", eamp=0.05, efreq=5.0, notch=0.10, bend=0.10),
        colors=[_pair("#FFF8EE", "#AC4D6C", "#D07E96"),
                _pair("#F6B8C6", "#9A1038", "#C42E57"),
                _pair("#FBDAC2", "#A82C12", "#D44E22")],
        shade=dict(w_base=1.05, p_base=1.25, w_rim=0.38, p_rim=2.0,
                   w_rimc=0.45, p_rimc=2.4, hl_tip=0.46),
        center=dict(kind="long", n=22, color="tenghuang", r=0.62),
        leaf=dict(n=(0, 3), L=(0.5, 0.8), wr=0.32, form="simple",
                  edge="serrate", eamp=0.10, efreq=13.0, tip=0.28),
        branch=dict(w=(13.0, 3.0), style="zigzag"),
        size=0.52, blossoms=(5, 9), buds=(3, 7),
    ),
    "chrysanthemum": dict(
        cn="菊花", season="秋",
        rings=[(16, 1.00, 0.72), (14, 0.86, 0.56), (12, 0.70, 0.42),
               (10, 0.54, 0.28), (8, 0.38, 0.16)],
        petal=dict(k1=3.0, e1=0.45, k2=2.6, e2=0.55, wr=0.17,
                   edge="plain", eamp=0.0, efreq=3.0, notch=0.0, bend=0.55),
        colors=[_pair("#FFE071", "#B96A02", "#E09C14"),
                # 白菊 — the pale cultivar still needs a real pigment at the
                # heart; a near-white deep stop was what made it read as paper
                _pair("#FBF4D6", "#9A6E1C", "#C59A3F"),
                _pair("#EFCCE2", "#7E2254", "#AC3E74"),
                _pair("#FFD08C", "#A83606", "#D4621C")],
        shade=dict(w_base=1.12, p_base=1.20, w_rim=0.22, p_rim=2.6,
                   w_rimc=0.40, p_rimc=2.8, hl_tip=0.50),
        center=dict(kind="none"),
        leaf=dict(n=(5, 9), L=(0.70, 1.00), wr=0.60, form="palmate",
                  edge="lobe", eamp=0.40, efreq=3.0, tip=0.16),
        branch=dict(w=(6.0, 2.6), style="smooth"),
        size=0.88, blossoms=(1, 3), buds=(1, 3),
    ),
    "magnolia": dict(
        cn="玉兰", season="春",
        rings=[(6, 1.00, 0.44), (3, 0.74, 0.20)],
        # 玉兰 tepals are long upright spoons, not round pansy petals
        petal=dict(k1=1.6, e1=0.72, k2=3.6, e2=0.52, wr=0.31,
                   edge="wave", eamp=0.03, efreq=4.0, notch=0.0, bend=0.26),
        # the pale stop stays inside the pigment family — a white first wash
        # under a strong 分染 leaves a violet petal with a plastic white rim
        colors=[_pair("#F6EDF2", "#8E6FA6", "#B59BC2"),
                _pair("#FBECEE", "#8E4E76", "#B37C9A"),
                _pair("#F1E6F6", "#60468C", "#8E74AE")],
        shade=dict(w_base=1.00, p_base=1.55, w_rim=0.30, p_rim=2.6,
                   w_rimc=0.34, p_rimc=2.8, hl=0.26, hl_tip=0.34),
        center=dict(kind="tuft", n=14, color="shihuang", r=0.14),
        leaf=dict(n=(0, 2), L=(0.6, 0.9), wr=0.42, form="simple",
                  edge="plain", eamp=0.0, efreq=4.0, tip=0.20),
        branch=dict(w=(12.0, 3.4), style="zigzag"),
        size=0.80, blossoms=(2, 4), buds=(2, 5),
    ),
    "camellia": dict(
        cn="山茶", season="冬",
        rings=[(7, 1.00, 0.80), (6, 0.80, 0.54), (5, 0.60, 0.32)],
        petal=dict(k1=2.4, e1=0.52, k2=6.0, e2=0.45, wr=0.74,
                   edge="wave", eamp=0.07, efreq=5.0, notch=0.14, bend=0.24),
        colors=[_pair("#F4B09C", "#9C0F04", "#CB2C14"),
                _pair("#F2A4B8", "#880C34", "#B82048"),
                _pair("#FFEFE6", "#B22438", "#D65666")],
        shade=dict(w_base=1.18, p_base=1.05, w_rim=0.36, p_rim=2.2,
                   w_rimc=0.55, p_rimc=2.2, hl_tip=0.52, veins=0.20,
                   vein_n=7.0),
        center=dict(kind="tuft", n=24, color="tenghuang", r=0.22),
        leaf=dict(n=(6, 10), L=(0.8, 1.10), wr=0.38, form="simple",
                  edge="serrate", eamp=0.09, efreq=16.0, tip=0.20,
                  glossy=True),
        branch=dict(w=(9.0, 3.2), style="smooth"),
        size=0.86, blossoms=(1, 3), buds=(1, 4),
    ),
}

# 罩染 — the unifying glaze brushed over a finished flower
GLAZES = [("tenghuang", 0.07), ("yanzhi", 0.06), ("huaqing", 0.05),
          ("zheshi", 0.06), (None, 0.0), (None, 0.0)]


# --------------------------------------------------------------------------
# 花头 — one blossom
# --------------------------------------------------------------------------

def glaze_palette(rng, colors):
    """Apply one 罩染 glaze so every petal of a painting shares an air."""
    light, deep, rim = colors
    name, k = GLAZES[rng.randrange(len(GLAZES))]
    if name:
        gl = P(name)
        light = mix(light, gl, k)
        deep = mix(deep, gl, k * 0.55)
        rim = mix(rim, gl, k * 0.7) if rim else None
    return light, deep, rim


def paint_blossom(cv, rng, spec, cx, cy, R, colors, openness=1.0,
                  depth_alpha=1.0):
    pr = spec["petal"]
    sh = spec["shade"]
    light, deep, rim = colors
    # 色线 — flowers are contoured in a darkened tone of their own pigment,
    # never in black ink; that is most of what makes the colour sing.
    line_c = mix(deep, P("mo"), 0.45)
    tilt = rng.uniform(0.42, 0.95)          # foreshortening of the disc
    spin = rng.uniform(0, math.tau)
    rings = spec["rings"]
    nring = max(1, len(rings) - 1)
    # One 开合 for the whole head: outer petals flare back, inner ones cup in.
    # Shared across the rings is the point — per-petal noise alone never adds
    # up to a flower that is convincingly budding or convincingly blown.
    flare = rng.uniform(0.35, 1.25)
    cup = rng.uniform(0.15, 0.95)
    wash = "sigmoid" if rng.random() < 0.45 else "power"   # 水口 or soft 分染

    for ri, (count, lscale, rr) in enumerate(rings):
        ring_t = ri / nring                 # 0 outermost .. 1 innermost
        curl = pr["bend"] * (flare * (1.0 - ring_t) - cup * ring_t)
        petals = []
        for k in range(count):
            a = spin + math.tau * (k + rng.uniform(-0.22, 0.22)) / count \
                + ri * 0.41
            dx, dy = math.cos(a), math.sin(a) * tilt
            r0 = R * rr * 0.30 * openness
            px, py = cx + dx * r0, cy + dy * r0
            ang = math.atan2(dy, dx)
            fore = 0.80 + 0.20 * (0.5 + 0.5 * math.sin(a))
            L = R * lscale * bell(rng, 0.84, 1.14) * openness * fore
            petals.append((py, px, py, ang, L, a))
        petals.sort(key=lambda p: p[0])      # far (upper) petals first

        for _, px, py, ang, L, a in petals:
            # petals turned away from the viewer are cooled and darkened (虚)
            back = 1.0 - clamp01((py - cy) / (R * 0.9) * 0.5 + 0.5)
            lt = jitter(hsv_mix(light, deep, 0.10 * back + rng.uniform(0, 0.06)),
                        rng, 5)
            # 虚 — petals turned away cool toward 花青 and lose chroma, which
            # is what pushes the near petals forward.
            dp = jitter(saturate(hsv_mix(deep, P("huaqing"), 0.05 + 0.20 * back),
                                 1.0 - 0.22 * back), rng, 6)
            Blade(cx=px, cy=py, ang=ang,
                  L=L, W=L * pr["wr"] * bell(rng, 0.86, 1.14),
                  k1=pr["k1"], e1=pr["e1"], k2=pr["k2"], e2=pr["e2"],
                  edge=pr["edge"], eamp=pr["eamp"] * rng.uniform(0.6, 1.4),
                  efreq=pr["efreq"] * rng.uniform(0.8, 1.3),
                  ephase=rng.uniform(0, 6.28),
                  bend=curl * bell(rng, 0.55, 1.45),
                  notch=pr["notch"] * rng.uniform(0.4, 1.4),
                  light=lt, deep=dp, ink=line_c,
                  lit=0.52 * (1.0 - 0.45 * back), wash=wash, bow=0.16,
                  line=rng.uniform(0.50, 0.72), alpha=depth_alpha,
                  w_base=sh.get("w_base", 0.0), p_base=sh.get("p_base", 2.0),
                  w_tip=sh.get("w_tip", 0.0), p_tip=sh.get("p_tip", 2.0),
                  w_rim=sh.get("w_rim", 0.0), p_rim=sh.get("p_rim", 3.0),
                  hl=sh.get("hl", 0.0), hl_tip=sh.get("hl_tip", 0.0),
                  rim_c=rim, w_rimc=sh.get("w_rimc", 0.0) * rng.uniform(0.8, 1.2),
                  p_rimc=sh.get("p_rimc", 4.0),
                  veins=sh.get("veins", 0.0) * rng.uniform(0.7, 1.2),
                  vein_n=sh.get("vein_n", 7.0),
                  vein_fan=sh.get("vein_fan", 1.5),
                  vein_ink=mix(deep, P("mo"), 0.25),
                  ).render(cv)

    paint_center(cv, rng, spec, cx, cy, R, deep, tilt, depth_alpha)


def paint_center(cv, rng, spec, cx, cy, R, deep, tilt, alpha=1.0):
    c = spec["center"]
    kind = c.get("kind", "none")
    if kind == "none":
        return
    col = P(c.get("color", "tenghuang"))
    hot = mix(col, P("zhuhong"), 0.35)
    r = R * c.get("r", 0.2)

    if kind == "pod":                      # 莲蓬 — lotus receptacle
        pod = mix(P("caolv"), P("baifen"), 0.30)
        disc(cv, cx, cy, r, pod, alpha, squash=tilt * 1.1,
             edge=mix(P("shilv"), P("mo"), 0.3), edge_k=1.6)
        for _ in range(int(r * r * 0.09) + 6):
            a = rng.uniform(0, math.tau)
            d = rng.uniform(0, 0.8) * r
            disc(cv, cx + math.cos(a) * d, cy + math.sin(a) * d * tilt,
                 max(1.2, r * 0.11), mix(P("shilv"), P("mo"), 0.45), alpha * 0.9)
    if kind in ("tuft", "pod", "long"):
        n = c.get("n", 24)
        reach = r * (3.0 if kind == "long" else 1.7)
        for _ in range(n):
            a = rng.uniform(0, math.tau)
            d = reach * rng.uniform(0.35, 1.0)
            ex = cx + math.cos(a) * d
            ey = cy + math.sin(a) * d * tilt
            mid = (cx + (ex - cx) * 0.55 + rng.uniform(-r, r) * 0.3,
                   cy + (ey - cy) * 0.55 + rng.uniform(-r, r) * 0.3)
            brush_path(cv, [(cx, cy), mid, (ex, ey)], 1.5, 0.8,
                       mix(col, P("zheshi"), 0.35), alpha * 0.85)
            disc(cv, ex, ey, rng.uniform(1.6, 3.0) * (1 + R * 0.004),
                 hot if rng.random() < 0.35 else col, alpha,
                 edge=P("zheshi"), edge_k=1.2)
        if kind == "tuft":
            disc(cv, cx, cy, r * 0.5, mix(col, P("tenghuang"), 0.5),
                 alpha * 0.8, squash=tilt)


def paint_bud(cv, rng, spec, cx, cy, R, colors, ang, alpha=1.0):
    light, deep, rim = colors
    stage = rng.uniform(0.25, 0.75)        # how far it has opened
    green_l = mix(P("caolv"), P("baifen"), 0.32)
    green_d = mix(mix(P("shilv"), P("huaqing"), 0.35), P("mo"), 0.40)
    # 花苞 — a few wrapped petals, still greener the tighter they are
    for i in range(3):
        a = ang + rng.uniform(-0.55, 0.55)
        L = R * (0.85 + 0.25 * stage) * rng.uniform(0.85, 1.1)
        Blade(cx=cx, cy=cy, ang=a, L=L, W=L * 0.42 * rng.uniform(0.85, 1.15),
              k1=2.2, e1=0.6, k2=5.5, e2=0.45,
              bend=rng.uniform(-0.25, 0.25),
              light=jitter(mix(light, green_l, 0.20 * (1 - stage)), rng, 5),
              deep=jitter(mix(deep, green_d, 0.28 * (1 - stage)), rng, 6),
              ink=mix(deep, P("mo"), 0.45),
              line=0.55, alpha=alpha,
              w_base=0.95, p_base=1.5, w_rim=0.26, p_rim=2.8, hl_tip=0.34,
              rim_c=rim, w_rimc=0.35 if rim else 0.0, p_rimc=3.2,
              ).render(cv)
    # 萼片 — sepals
    for _ in range(rng.randint(3, 5)):
        a = ang + math.pi + rng.uniform(-1.1, 1.1)
        L = R * rng.uniform(0.5, 0.85)
        Blade(cx=cx, cy=cy, ang=a, L=L, W=L * 0.26,
              k1=2.0, e1=0.6, k2=2.2, e2=0.8, tip=0.3,
              bend=rng.uniform(-0.4, 0.4),
              light=jitter(green_l, rng, 7), deep=jitter(green_d, rng, 7),
              ink=mix(green_d, P("mo"), 0.5),
              line=0.5, alpha=alpha, w_base=0.7, p_base=1.6,
              w_rim=0.28, p_rim=3.0, midrib=0.18, veins=0.35, vein_n=0.0,
              vein_ink=green_d).render(cv)


# --------------------------------------------------------------------------
# 叶 — leaves
# --------------------------------------------------------------------------

def leaf_palette(rng, far=False, back=False):
    """石绿 over 花青 for a leaf face, 草绿 + 白粉 for its underside, 赭石 on
    an old rim. Returns (light, deep, rim, vein_ink).

    Values are kept deliberately high because foliage is laid down with a
    multiplying wash — the silk darkens it on the way in.
    """
    if far:                                # 远叶: cool, diluted, nearly flat
        g_l = hsv_mix(hsv_mix(P("shilv"), P("huaqing"), 0.26), P("baifen"), 0.52)
        g_d = hsv_mix(hsv_mix(P("shilv"), P("huaqing"), 0.40), P("baifen"), 0.30)
        rim = None
    elif back:                             # 叶背: paler, yellower, matte
        g_l = hsv_mix(P("caolv"), P("baifen"), 0.50)
        g_d = hsv_mix(P("caolv"), P("shilv"), 0.55)
        rim = mix(P("zheshi"), P("caolv"), 0.45)
    else:                                  # 叶面: deep blue-green
        g_l = saturate(hsv_mix(P("caolv"), P("baifen"), 0.26), 1.10)
        g_d = saturate(hsv_mix(hsv_mix(P("shilv"), P("huaqing"), 0.34),
                               P("mo"), 0.16), 1.12)
        rim = mix(P("zheshi"), P("mo"), 0.20)
    if not far and rng.random() < 0.20:    # 老叶 — autumn, sun-burnt
        g_l = hsv_mix(g_l, P("zheshi"), 0.38)
        g_d = hsv_mix(g_d, P("zheshi"), 0.34)
        rim = mix(P("zhuhong"), P("zheshi"), 0.5)
    # 叶筋 — veins are reserved *lighter* than the face, not inked darker:
    # on a deep malachite leaf a dark vein simply disappears.
    vink = hsv_mix(g_l, P("baifen"), 0.30 if not far else 0.10)
    return jitter(g_l, rng, 8), jitter(g_d, rng, 8), rim, vink


def _leaflet(cv, rng, lf, x, y, ang, L, pal, alpha, far, glossy=False):
    g_l, g_d, rim, vink = pal
    rnd = lf.get("form") == "round"
    W = L * lf["wr"] * bell(rng, 0.84, 1.16)
    Blade(
        cx=x, cy=y, ang=ang, L=L, W=W,
        # a 荷叶 is a disc. k=1 with e=1/2 at both ends makes
        # hw = sqrt(u(1-u)) — an exact ellipse, a circle when W == L.
        # Holding the width out to the rim (large k2) chops the end square,
        # which is what made the old lotus leaves read as moth wings.
        k1=1.0 if rnd else 2.1, e1=0.52 if rnd else 0.58,
        k2=1.1 if rnd else 2.2, e2=0.50 if rnd else 0.80,
        # 远叶 keep the species silhouette — it is contrast that falls away
        # with distance, not shape. Flat ovals back there read as smudges.
        tip=0.0 if rnd else lf.get("tip", 0.0),
        # 荷叶 is the one leaf whose ends are *supposed* to be blunt: the
        # extremity of a circle has a vertical tangent, and tucking it to a
        # point would turn the disc into a lens.
        tuck=0.0 if rnd else 1.0,
        edge=lf.get("edge", "plain"),
        eamp=lf.get("eamp", 0.0) * rng.uniform(0.75, 1.25) * (0.7 if far else 1.0),
        efreq=lf.get("efreq", 6.0) * rng.uniform(0.85, 1.2),
        ephase=rng.uniform(0, 6.28),
        bend=rng.uniform(-0.42, 0.42),
        light=g_l, deep=g_d, ink=mix(g_d, P("mo"), 0.55),
        line=0.18 if far else rng.uniform(0.44, 0.62), alpha=alpha,
        w_base=0.46, p_base=1.9, w_rim=0.34, p_rim=2.4,
        w_mid=0.10, p_mid=3.0,
        hl=(0.34 if glossy else 0.18) if not far else 0.08,
        soft=max(1.2, L * 0.022) if far else 0.0,
        # a leaf is a trough, not a plate: light one half, shade the other
        lit=0.18 if far else (0.52 if glossy else 0.40),
        bow=0.08, blend="mul",
        rim_c=rim, w_rimc=0.0 if far else 0.30, p_rimc=5.0,
        veins=0.55 if not far else 0.18,
        vein_n=rng.uniform(5.0, 8.0) if not rnd else rng.uniform(5.0, 7.5),
        vein_fan=1.6 if not rnd else -1.0,     # -1 = radial 荷叶 ribs
        vein_ink=vink, midrib=0.0 if rnd else 0.07,
    ).render(cv)
    return W


def _petiole(cv, rng, x, y, ang, pl, w0, w1, color, alpha, soft=0.0, over=0.0):
    """叶柄 — the stalk from the joint on the wood out to the base of the
    blade. Returns where the blade should start and which way it should go.

    This was a straight `brush_seg`, and it was the one genuinely *ruled*
    mark in the painting: fifteen pixels of hairline at constant width,
    constant colour and zero curvature, which on bare silk reads as a pencil
    construction line rather than as a stalk. On a 远叶 it was worse — the
    stalk carried nearly twice the blade's alpha, an overcorrection for "a
    one-pixel line needs more alpha than a forty-pixel wash", so the stick
    came out *darker than the leaf hanging off it* and the eye went straight
    to it. Both faults are fixed here. The stalk is a stem Blade with a
    little `bend`, so it leaves the wood on a curve and is drawn by the same
    analytic edge as everything else; and the extra weight a thin mark needs
    is taken in width rather than in alpha. The blade then starts at the
    stalk's *bent* tip and continues its tangent, so nothing kinks at the
    joint — which is also why this has to be drawn before the blade, not
    after: the base of the leaf should cover the end of its own stalk.

    `soft` spreads the edge into a wash, for a 远叶 stalk that has to sit in
    the same plane as its leaf. Without it the width that buys a thin mark its
    weight buys it something else as well: at a couple of pixels wide, fully
    opaque, dark-rimmed and square at the tip, the stalk stopped reading as a
    line at all and became a little domino lying next to a pale green cloud.

    `over` runs the stalk a few pixels *past* the point it hands the blade,
    because the blade now tucks its own base to a cusp: the first seven pixels
    of the leaf are pinched to nothing, so a stalk that stopped exactly at the
    anchor stopped in mid-air, with bare paper between its blunt end and the
    first of the green. The overlap is drawn under the blade and only shows
    where the blade is too thin to cover it, which is precisely where a stalk
    should still be visible.
    """
    ben = rng.uniform(-0.13, 0.13)
    ca, sa = math.cos(ang), math.sin(ang)
    off = ben * pl
    nx, ny = x + ca * pl - sa * off, y + sa * pl + ca * off
    if pl > 2.4 and w0 > 0.52:
        Blade(cx=x, cy=y, ang=ang, L=pl + over, W=w0, mode="stem",
              stem_taper=max(0.0, min(0.95, 1.0 - w1 / w0)), bend=ben,
              light=color,
              # A pale stalk takes no modelling: a cylinder rim on something
              # two pixels across is all rim, and the "shading" is just a
              # darker stick.
              deep=mix(color, P("mo"), 0.30 if soft <= 0.0 else 0.06),
              ink=P("mo"), soft=soft,
              w_base=0.0, w_rim=0.50 if soft <= 0.0 else 0.0,
              p_rim=1.6, w_mid=0.0,
              lit=0.55 if soft <= 0.0 else 0.30, bow=0.10,
              line=0.0, alpha=alpha,
              blend="mul").render(cv)
    return nx, ny, ang + math.atan(2.0 * ben)


def paint_leaf(cv, rng, spec, x, y, ang, L, alpha=1.0, far=False, depth=0.0):
    """One leaf as the species actually grows it, petiole included.

    (x, y) is where the leaf *joins* — the point on the stem's edge the
    petiole springs from, not the base of the blade. Everything is built
    outward from there along `ang`, so the stalk always reads as leaving the
    wood and the blade always hangs off the end of the stalk.

    `depth` says this leaf lies behind one already painted. Two leaves of the
    same value, overlapping, do not read as two leaves — they merge into one
    ragged green mass, which is what a shingle of 菊花 leaves had become.
    Pushing the hind one back half a step — a little less alpha, and a much
    better chance of being turned to show its pale underside — is what a
    painter does instead: 层次, the overlap itself becomes the depth cue.
    """
    lf = spec["leaf"]
    form = lf.get("form", "simple")
    pal = leaf_palette(rng, far=far,
                       back=(not far and rng.random() < 0.28 + 0.34 * depth))
    alpha = clamp01(alpha * (1.0 - 0.26 * depth))
    glossy = lf.get("glossy", False)

    # Nominal blade width, so the stalk can be sized before the blade is
    # drawn — the stalk has to go down first for its tip to be covered.
    Wn = L * lf["wr"]

    if form == "trifoliate" and not far:   # 牡丹叶 — three leaflets per stalk
        pl = L * 0.30
        nx, ny, bang = _petiole(cv, rng, x, y, ang, pl,
                                max(1.0, Wn * 0.090), max(1.0, Wn * 0.070),
                                mix(pal[1], P("zheshi"), 0.35), alpha * 0.95,
                                over=min(7.0, 0.11 * L) + 1.0)
        spread = rng.uniform(0.52, 0.80)
        for da, sc in ((0.0, 1.0), (-spread, rng.uniform(0.66, 0.84)),
                       (spread, rng.uniform(0.66, 0.84))):
            ox = nx - math.cos(bang) * (0.0 if da == 0.0 else pl * 0.35)
            oy = ny - math.sin(bang) * (0.0 if da == 0.0 else pl * 0.35)
            _leaflet(cv, rng, lf, ox, oy, bang + da, L * sc,
                     pal, alpha, far, glossy)
        return

    # 叶柄 — petiole, running *out* from the joint to the base of the blade.
    # Drawn the other way, back from the blade, it sat on the wrong side of
    # the attachment point: short of the stem it vanished inside the bark and
    # the blade appeared to sprout from the middle of the wood, and longer
    # than the stem was thick it came out the back as a stray green stub. The
    # trifoliate branch above already builds outward; this is the same order.
    # Thicker at the stem than at the blade, and thinner overall than the wood
    # it springs from. The taper used to run the other way, so the stalk
    # widened into a green lump where it met the blade and the joint at the
    # stem end tapered away to nothing — the leaf looked stuck on with a blob
    # of paint rather than hinged to the branch. A 远叶 stalk gets a little
    # extra weight — it is the only evidence the leaf is attached to anything,
    # and one pixel of line against forty of blade needs the help — but only a
    # little, and taken in width rather than in alpha. Pushed further it stops
    # being a thin mark that needs help and becomes a dark bar that needs
    # removing; and because it lies in the same misty plane as its leaf it is
    # washed out to the same soft edge. See _petiole.
    pl = L * 0.18
    nx, ny, bang = _petiole(cv, rng, x, y, ang, pl,
                            max(0.82, Wn * 0.040) if far else max(0.9, Wn * 0.048),
                            max(0.70, Wn * 0.029) if far else max(0.8, Wn * 0.034),
                            mix(pal[1], P("zheshi"), 0.30) if not far
                            else mix(mix(pal[1], P("zheshi"), 0.30),
                                     P("baifen"), 0.26),
                            clamp01(alpha * (1.05 if far else 0.9)),
                            soft=0.75 if far else 0.0,
                            over=min(7.0, 0.11 * L) + 1.0)
    _leaflet(cv, rng, lf, nx, ny, bang, L, pal, alpha, far, glossy)


def leaf_sites(rng, hosts, n, Lmin, Lmax, span=(0.45, 1.35),
               sep=0.80, grade=0.26, pair=0.34, room=-0.30, sheet=None):
    """叶序 — settle where every leaf goes *before* any of them is painted.

    Each leaf used to choose a host, a node and a side independently and
    uniformly. Nothing then stopped two of them from landing on the same node
    on the same side at the same angle, and with eight or ten leaves over
    thirty nodes that happened constantly: a shingle of five identical blades
    all hanging the same way, interlocking lobes, no bark visible between
    them — one ragged green mass instead of five leaves. Worse, the pile left
    the rest of the branch bare, so the sheet lost its balance as well as its
    foliage.

    So the leaves are laid out as a set rather than drawn one at a time. Each
    is chosen by farthest-point sampling: a few dozen candidate nodes are
    offered, each is scored on how far its *disc* clears every leaf already
    settled, and the roomiest wins. That is blue noise — even coverage with no
    clumps, and it costs nothing at these counts.

    Three things keep it from becoming a mechanical row, which would be its
    own kind of ugly:

    - 对生. One leaf in three is placed deliberately as the partner of the
      last, at the same node on the *other* side. Two leaves at one node
      pointing opposite ways read as a node, not as a clump, and that is how
      most of these species actually grow.
    - 参差. Size is graded down toward the tip of each host and jittered, so
      no two neighbours match. Equal leaves at equal spacing is a fence.
    - 疏密. Sampling stops as soon as the roomiest candidate left is worse
      than `room`, so a short twig simply carries fewer leaves instead of
      being packed to the quota. The 留白 is part of the picture.

    `sheet` is the size of the paper. The edge counts as a crowded neighbour:
    of two equally roomy nodes the sampler then takes the one pointing in at
    the picture, which is how the half-leaves hanging off the bottom corner
    stopped happening. It is a preference, not a rule — foliage running off
    the edge of a 图卷 is ordinary, a leaf whose middle is past it is not.

    Returns [(x, y, ang, L, depth)] ordered tip-first, so the basal leaves —
    the big near ones — are painted last and lap *over* the small distant
    ones. `depth` is how much of the leaf is buried under one already placed;
    `paint_leaf` spends it on value separation.
    """
    cand = []
    for h, (path, hw0, hw1) in enumerate(hosts):
        npt = len(path)
        for i in range(max(1, int(npt * 0.15)), npt):
            cand.append((h, i, i / float(npt - 1)))
    if not cand:
        return []

    def realize(h, i, t, sgn, L):
        """A candidate as actual geometry: where the petiole leaves the bark,
        which way the leaf hangs, and the disc it will cover."""
        path, hw0, hw1 = hosts[h]
        tg = tangent(path, i)
        r = hw0 + (hw1 - hw0) * t
        x = path[i][0] + math.cos(tg + sgn * 1.5708) * r
        y = path[i][1] + math.sin(tg + sgn * 1.5708) * r
        a = tg + sgn * rng.uniform(*span)
        return (x, y, a, L, x + math.cos(a) * L * 0.5,
                y + math.sin(a) * L * 0.5, L * 0.46)

    out = []                                 # (h, i, t, sgn, geometry...)

    def room_for(g):
        """Clearance of a candidate disc, in pixels: how far it is from
        touching its nearest settled neighbour, or from falling off the
        sheet, whichever is tighter."""
        s = min((math.hypot(g[4] - o[8], g[5] - o[9]) - sep * (g[6] + o[10])
                 for o in out), default=1e9)
        if sheet is not None:
            s = min(s, min(g[4], g[5], sheet[0] - g[4],
                           sheet[1] - g[5]) - 0.45 * g[6])
        return s

    for k in range(n):
        best, best_score = None, None
        # 对生 — try the partner of the last leaf first, on the far side.
        if out and rng.random() < pair:
            h, i, t, sgn = out[-1][:4]
            g = realize(h, i, t, -sgn, out[-1][7] * rng.uniform(0.74, 0.98))
            s = room_for(g)
            if s > room * 1.7 * g[3]:        # a partner may crowd a little
                best, best_score = (h, i, t, -sgn) + g, s
        if best is None:
            for _ in range(22):
                h, i, t = cand[rng.randrange(len(cand))]
                sgn = rng.choice((1, -1))
                L = rng.uniform(Lmin, Lmax) * (1.0 - grade * t)
                g = realize(h, i, t, sgn, L)
                s = room_for(g)
                if best_score is None or s > best_score:
                    best, best_score = (h, i, t, sgn) + g, s
        if best is None or best_score < room * best[7]:
            break                            # no room left — leave the 留白
        out.append(best)

    out.sort(key=lambda o: -o[2])             # tip-first, so the base laps over
    sites = []
    for j, o in enumerate(out):
        bur = 0.0
        for p in out[:j]:                     # how buried by what goes down first
            d = math.hypot(o[8] - p[8], o[9] - p[9])
            bur = max(bur, clamp01((o[10] + p[10] - d) / max(o[10], 1e-6) - 0.25))
        sites.append((o[4], o[5], o[6], o[7], bur))
    return sites


# --------------------------------------------------------------------------
# 蝴蝶 — a butterfly, the classic gongbi companion
# --------------------------------------------------------------------------

def paint_butterfly(cv, rng, x, y, s, ang):
    body = mix(P("mo"), P("zheshi"), 0.25)
    schemes = [(rgb("#F2C24A"), rgb("#8A3B1E")), (rgb("#EFE6D2"), rgb("#4A4A52")),
               (rgb("#CFE0EC"), rgb("#2D5574")), (rgb("#F6D9E2"), rgb("#8E3358"))]
    light, deep = schemes[rng.randrange(len(schemes))]
    for side in (1, -1):
        for fore in (1, 0):
            a = ang + side * (0.88 if fore else 1.90) * rng.uniform(0.9, 1.1)
            L = s * (1.08 if fore else 0.66)
            # 前翅 draws out to a point, 后翅 stays round — two round blades of
            # the same size read as a moth blob rather than a butterfly.
            Blade(cx=x, cy=y, ang=a, L=L, W=L * (0.52 if fore else 0.76),
                  k1=1.8, e1=0.7,
                  k2=2.3 if fore else 4.2, e2=0.62 if fore else 0.48,
                  tip=0.30 if fore else 0.0,
                  bend=side * rng.uniform(0.05, 0.3),
                  light=light, deep=deep, line=0.75,
                  w_base=0.30, p_base=2.2, w_tip=0.55, p_tip=2.4,
                  w_rim=0.45, p_rim=2.2, hl=0.25,
                  lit=0.35, bow=0.14,
                  rim_c=mix(deep, P("mo"), 0.45), w_rimc=0.5, p_rimc=7.0,
                  veins=0.40, vein_n=5.0, vein_fan=1.0,
                  vein_ink=mix(deep, P("mo"), 0.5)).render(cv)
            # 点彩 — a scatter of small wing spots, not two big eyes
            for _ in range(rng.randint(3, 7)):
                t = rng.uniform(0.42, 0.9)
                o = rng.uniform(-0.5, 0.5) * L * 0.55 * (1.0 - t * 0.5)
                sx = x + math.cos(a) * L * t - math.sin(a) * o
                sy = y + math.sin(a) * L * t + math.cos(a) * o
                disc(cv, sx, sy, s * rng.uniform(0.022, 0.05),
                     P("baifen") if rng.random() < 0.45 else P("mo"),
                     rng.uniform(0.55, 0.85))
    for i in range(5):
        t = i / 4
        brush_seg(cv, x + math.cos(ang) * s * 0.3 * (1 - t) * -1,
                  y + math.sin(ang) * s * 0.3 * (1 - t) * -1,
                  x - math.cos(ang) * s * 0.42 * t,
                  y - math.sin(ang) * s * 0.42 * t,
                  s * 0.085 * (1 - t * 0.6), s * 0.07 * (1 - t * 0.6),
                  body, 0.95)
    disc(cv, x, y, s * 0.1, body, 1.0)
    for side in (1, -1):
        ax = ang + side * 0.5
        brush_path(cv, [(x, y),
                        (x + math.cos(ax) * s * 0.3, y + math.sin(ax) * s * 0.3),
                        (x + math.cos(ax + side * 0.6) * s * 0.52,
                         y + math.sin(ax + side * 0.6) * s * 0.52)],
                   1.1, 0.6, body, 0.9)


# --------------------------------------------------------------------------
# 蜂 — a bee, the other classic companion
# --------------------------------------------------------------------------

def paint_bee(cv, rng, x, y, s, ang):
    """One hovering bee. (x, y) is the thorax, `ang` the way it faces, and `s`
    is the body length from head to sting.

    A bee is not a small butterfly, and the difference is entirely in *what
    gets the paint*. A butterfly is its wings: four broad patterned blades,
    with a body that is a stub to hold them together. A bee is the other way
    round — the striped body carries the whole picture and the wings are a
    smear of almost nothing. So they are narrow, swept back, pale, soft-edged
    and drawn *under* the body. Given a butterfly's wing weight the same marks
    read as a moth every time, which is what the first draft of this was.

    The parts are laid down back to front, the way they occlude: wings, then
    the legs (so the abdomen covers the roots of them, as a body does), then
    the abdomen, its bands, the furry thorax, and the head last.
    """
    ca, sa = math.cos(ang), math.sin(ang)

    def pt(f, o=0.0):
        """A point `f` forward of the thorax and `o` to its left."""
        return x + ca * f - sa * o, y + sa * f + ca * o

    ink = mix(P("mo"), P("zheshi"), 0.16)

    # 翅 — two pairs, swept back. 花青 diluted almost to nothing: a wing over
    # silk is a film, so it multiplies, and `soft` spreads its edge into a
    # wash. A crisp one-pixel edge here would make it a cut-out of a wing.
    #
    # The venation is the detail that decides whether these read as wings at
    # all. A `Blade`'s default veins are a midrib with a fan off it — which is
    # a *leaf*, and with the wing's pointed tip the first draft of this grew
    # two small leaves out of the thorax. So: `tip` raised to a round end, the
    # width cut to a quarter of the length, and `vein_fan` dropped to almost
    # nothing, which is what turns a fanned leaf midrib into the two long
    # near-parallel veins an insect wing actually has.
    for side in (1, -1):
        for fore in (1, 0):
            wx, wy = pt(s * 0.19, side * s * 0.05)
            a = ang + side * (2.34 if fore else 2.66) * rng.uniform(0.96, 1.04)
            L = s * (0.60 if fore else 0.40)
            Blade(cx=wx, cy=wy, ang=a, L=L, W=L * (0.25 if fore else 0.28),
                  k1=2.2, e1=0.52, k2=2.6, e2=0.62, tip=0.10,
                  bend=side * rng.uniform(0.04, 0.14),
                  light=mix(P("baifen"), P("huaqing"), 0.04),
                  deep=mix(P("huaqing"), P("baifen"), 0.72),
                  ink=P("huaqing"), line=0.34,
                  w_base=0.26, p_base=1.6, w_rim=0.30, p_rim=2.4,
                  veins=0.09, vein_n=2.0, vein_fan=0.15,
                  vein_ink=mix(P("huaqing"), P("baifen"), 0.40),
                  lit=0.18, soft=0.7, bow=0.06,
                  alpha=rng.uniform(0.26, 0.36) * (1.0 if fore else 0.86),
                  blend="mul").render(cv)

    # 足 — six legs, all trailing: a bee in the air lets them hang, and that
    # backward rake is most of what says "hovering" rather than "pinned".
    #
    # They are drawn first, so the abdomen covers the roots of them as a body
    # does — but the first version put the roots on the axis and made the legs
    # a fifth of the body long, so the abdomen covered *all* of them and six
    # legs painted nothing at all. The two numbers that fixed it: root them out
    # to the side, and make them long enough that the knee clears the abdomen's
    # `0.155 s` half-width. A thin line that escapes the body is a leg; a thick
    # one that does not is a smudge under the paint.
    #
    # `0.16 s` out, specifically, which is just past the thorax disc's `0.15 s`
    # radius. Rooted any closer and the roots show *through* the thorax: the
    # disc is drawn with `soft`, which fades its coverage toward the rim, so a
    # dark line under the edge of it stays a quarter visible and the bee grows
    # three stripes across its back.
    for side in (1, -1):
        for k, swp in enumerate((2.15, 2.40, 2.62)):
            a = ang + side * swp * rng.uniform(0.94, 1.06)
            ln = s * (0.25 + 0.045 * k) * rng.uniform(0.88, 1.12)
            bx, by = pt(s * (0.21 - 0.08 * k), side * s * 0.16)
            kx, ky = bx + math.cos(a) * ln * 0.58, by + math.sin(a) * ln * 0.58
            a2 = a + side * rng.uniform(0.45, 0.85)
            # Darker than they look like they should be. At the size a bee is
            # actually painted the line is already under a pixel wide, so its
            # coverage does the dimming for free; at a gentle alpha on top of
            # that, six legs and two pairs of wings turn the back half of the
            # insect into one grey fan and it reads as a spider.
            brush_path(cv, [(bx, by), (kx, ky),
                            (kx + math.cos(a2) * ln * 0.42,
                             ky + math.sin(a2) * ln * 0.42)],
                       max(0.5, s * 0.012), max(0.35, s * 0.006),
                       ink, rng.uniform(0.72, 0.90))

    # 腹 — the abdomen, drawn backwards off the thorax: blunt where it joins,
    # tapering to the sting. 藤黄 over 赭石, which is the colour a bee
    # actually is once the stripes are on it; a lemon-yellow one reads as a
    # wasp.
    abd = Blade(cx=pt(s * 0.07)[0], cy=pt(s * 0.07)[1], ang=ang + math.pi,
                # Blunt at the sting, and blunt the hard way: `tip` is an
                # acuminate *extension*, so asking for a rounder end by
                # raising it grew a pale needle out of the back of the bee
                # instead. The end is shaped by `k2` large and `e2` small —
                # full width until the last twentieth, then gone.
                L=s * 0.58, W=s * 0.155,
                k1=2.8, e1=0.44, k2=3.2, e2=0.34, tip=0.0,
                bend=rng.uniform(-0.10, 0.10),
                light=mix(P("tenghuang"), P("baifen"), 0.26),
                deep=mix(P("shihuang"), P("zheshi"), 0.42),
                ink=mix(P("zheshi"), P("mo"), 0.45), line=0.60,
                w_base=0.34, p_base=1.5, w_tip=0.72, p_tip=1.5,
                w_rim=0.46, p_rim=2.0, hl=0.22, hl_tip=0.0,
                lit=0.50, bow=0.12)
    abd.render(cv)

    # 环纹 — the bands. Measured off the abdomen's own half-width, so they end
    # where the silhouette does however the blade was shaped; and pulled in to
    # 0.78 of it, because `brush_seg` is round-capped and a band drawn to the
    # full width bulges out past the edge of the body it is painted on.
    #
    # Four thin ones, not three fat ones. `brush_seg`'s `w` is a radius, so the
    # first draft's `s * 0.034` was a bar `0.068 s` across on an abdomen of
    # `0.58 s` — black and yellow in equal measure, which is a wasp. On a bee
    # the yellow wins and the bands are the narrower of the two.
    bang = ang + math.pi
    bca, bsa = math.cos(bang), math.sin(bang)
    for u in (0.20, 0.39, 0.58, 0.76):
        hw = abd._hw(u) * 0.78
        if hw < 0.6:
            continue
        fx = abd.cx + bca * u * abd.L
        fy = abd.cy + bsa * u * abd.L
        bw = max(0.55, s * 0.019)
        brush_seg(cv, fx - bsa * -hw, fy + bca * -hw,
                  fx - bsa * hw, fy + bca * hw, bw, bw,
                  mix(P("mo"), P("zheshi"), 0.22), rng.uniform(0.78, 0.92))

    # 胸 — the thorax, which is the furry part. `soft` on the disc fades the
    # rim, so it reads as hair rather than as a bead of paint.
    tx, ty = pt(s * 0.17)
    disc(cv, tx, ty, s * 0.150, mix(P("zheshi"), P("tenghuang"), 0.42),
         1.0, soft=0.32, edge=ink, edge_k=1.1)
    # 点彩 — a fleck or two of fuzz. Faint: at 0.5 alpha these were three white
    # dots on a dark disc and the thorax grew a face.
    for _ in range(rng.randint(2, 4)):
        a = rng.uniform(0, math.tau)
        d = s * rng.uniform(0.02, 0.10)
        disc(cv, tx + math.cos(a) * d, ty + math.sin(a) * d,
             s * rng.uniform(0.014, 0.026), P("baifen"), rng.uniform(0.16, 0.30))

    # 头 and 触角 — the head, and two elbowed antennae reaching forward. Kept
    # short and finer than the legs: at a quarter of the body each, in leg
    # weight, they stood off the head like a cricket's and the insect changed
    # species. A bee's antenna is about a head and a half, and it bends.
    hx, hy = pt(s * 0.38)
    disc(cv, hx, hy, s * 0.100, ink, 0.95, soft=0.2)
    for side in (1, -1):
        a = ang + side * rng.uniform(0.42, 0.68)
        ln = s * rng.uniform(0.13, 0.18)
        ex, ey = pt(s * 0.43, side * s * 0.05)
        kx, ky = ex + math.cos(a) * ln * 0.55, ey + math.sin(a) * ln * 0.55
        a2 = a + side * rng.uniform(0.30, 0.70)
        brush_path(cv, [(ex, ey), (kx, ky),
                        (kx + math.cos(a2) * ln * 0.5,
                         ky + math.sin(a2) * ln * 0.5)],
                   max(0.45, s * 0.013), max(0.35, s * 0.007), ink, 0.85)


# --------------------------------------------------------------------------
# 题款与印章 — inscription column and seals (procedural, no font needed)
# --------------------------------------------------------------------------

class BrushFont:
    """Decoder for the baked brush-calligraphy outlines in gongbi_font.py.

    Contours arrive delta-encoded as zig-zag varints on a 1000-unit em square
    with the baseline at y = 0 and y pointing up, as in the source font.
    """

    def __init__(self, blob, em=1000):
        raw = zlib.decompress(base64.b85decode(blob))
        self.em = float(em)
        self.glyphs = {}
        self._bbox = {}
        p = 0

        def rd():
            nonlocal p
            v, sft = 0, 0
            while True:
                b = raw[p]
                p += 1
                v |= (b & 0x7F) << sft
                if not b & 0x80:
                    return v
                sft += 7

        def rds():
            v = rd()
            return (v >> 1) if not v & 1 else -((v + 1) >> 1)

        n = rd()
        for _ in range(n):
            ch = chr(rd())
            adv = rd()
            polys = []
            for _ in range(rd()):
                m = rd()
                x = y = 0
                pts = []
                for _ in range(m):
                    x += rds()
                    y += rds()
                    pts.append((x, y))
                polys.append(pts)
            self.glyphs[ch] = (adv, polys)

    def has(self, ch):
        return ch in self.glyphs

    def bbox(self, ch):
        if ch not in self._bbox:
            _, polys = self.glyphs[ch]
            xs = [p[0] for c in polys for p in c]
            ys = [p[1] for c in polys for p in c]
            self._bbox[ch] = (min(xs), min(ys), max(xs), max(ys))
        return self._bbox[ch]

    def outline(self, ch, x, y, s, rot=0.0, fill_cell=False):
        """Device-space contours for `ch` drawn in an s-sized cell at (x, y)."""
        if ch not in self.glyphs:
            return []
        _, polys = self.glyphs[ch]
        if fill_cell:                       # pack the glyph into the box (印章)
            bx0, by0, bx1, by1 = self.bbox(ch)
            gw = max(1.0, bx1 - bx0)
            gh = max(1.0, by1 - by0)
            kx, ky = s * 0.96 / gw, s * 0.96 / gh
            ox, oy = -bx0 * kx + s * 0.02, by1 * ky + s * 0.02
        else:
            kx = ky = s / self.em
            ox, oy = 0.0, s * 0.88          # baseline inside the cell
        ca, sa = math.cos(rot), math.sin(rot)
        out = []
        for c in polys:
            dc = []
            for gx, gy in c:
                lx = gx * kx + ox
                ly = oy - gy * ky
                dc.append((x + lx * ca - ly * sa, y + lx * sa + ly * ca))
            out.append(dc)
        return out


def fill_contours(cv, polys, color, alpha=1.0, mottle=None):
    """Scanline fill, non-zero winding, 4x vertical AA. `color` may be a
    callable (px, py) -> rgb, which the seals use to carve 白文."""
    if not polys:
        return
    xs = [p[0] for c in polys for p in c]
    ys = [p[1] for c in polys for p in c]
    x0, y0, x1, y1 = cv.clip(min(xs) - 1, min(ys) - 1, max(xs) + 1, max(ys) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    edges = []
    for c in polys:
        for i in range(len(c)):
            ax, ay = c[i]
            bx, by = c[(i + 1) % len(c)]
            if ay != by:
                edges.append((min(ay, by), max(ay, by), ax, ay, bx, by))
    if not edges:
        return
    rows = {}
    for e in edges:                          # bucket edges by scanline
        for py in range(max(y0, int(e[0])), min(y1, int(e[1]) + 1)):
            rows.setdefault(py, []).append(e)

    SS = 4
    inv = 1.0 / SS
    width = x1 - x0
    callable_color = callable(color)
    for py in range(y0, y1):
        bucket = rows.get(py)
        if not bucket:
            continue
        cov = [0.0] * width
        for sub in range(SS):
            sy = py + (sub + 0.5) * inv
            hits = []
            for ymin, ymax, ax, ay, bx, by in bucket:
                if ay <= sy < by or by <= sy < ay:
                    hits.append((ax + (bx - ax) * (sy - ay) / (by - ay),
                                 1 if by > ay else -1))
            if len(hits) < 2:
                continue
            hits.sort()
            wind = 0
            start = 0.0
            for hx, d in hits:
                prev = wind
                wind += d
                if prev == 0 and wind != 0:
                    start = hx
                elif prev != 0 and wind == 0:
                    xa, xb = start, hx
                    if xb <= x0 or xa >= x1:
                        continue
                    if xa < x0:
                        xa = x0
                    if xb > x1:
                        xb = x1
                    ia, ib = int(xa), int(xb)
                    if ia == ib:
                        cov[ia - x0] += (xb - xa) * inv
                    else:
                        cov[ia - x0] += (ia + 1 - xa) * inv
                        for k in range(ia + 1 - x0, ib - x0):
                            cov[k] += inv
                        if ib < x1:
                            cov[ib - x0] += (xb - ib) * inv
        for k in range(width):
            c = cov[k]
            if c <= 0.002:
                continue
            if c > 1.0:
                c = 1.0
            px = x0 + k
            if mottle is not None:           # dry-brush unevenness of real ink
                c *= 0.82 + 0.18 * mottle.at((px % 97) / 97.0, (py % 89) / 89.0)
            cv.blend(px, py, color(px, py) if callable_color else color, c * alpha)


def paint_column(cv, rng, font, text, x, y, size, ink=None, lead=1.22,
                 alpha=(0.74, 0.95)):
    """One vertical line of writing, top to bottom. Returns the ending y."""
    ink = ink or mix(P("mo"), P("danmo"), rng.uniform(0.0, 0.22))
    mottle = Noise(rng, 12, 12)
    for i, ch in enumerate(text):
        cy = y + i * size * lead
        if font is None or not font.has(ch):
            continue
        fill_contours(cv, font.outline(ch, x + rng.uniform(-0.035, 0.035) * size,
                                       cy, size, rot=rng.uniform(-0.028, 0.028)),
                      jitter(ink, rng, 7), rng.uniform(*alpha), mottle)
    return y + len(text) * size * lead


def paint_seal(cv, rng, font, text, x, y, s, relief=None):
    """Vermilion seal: 白文 carves the characters out, 朱文 prints them."""
    if relief is None:
        relief = rng.random() < 0.4
    red = jitter(rgb("#B23221"), rng, 9)
    x0, y0, x1, y1 = cv.clip(x, y, x + s, y + s)
    if x0 >= x1 or y0 >= y1:
        return
    # remember the silk underneath so 白文 can be carved back to it
    keep = {}
    for py in range(y0, y1):
        for px in range(x0, x1):
            i = (py * cv.w + px) * 3
            keep[(px, py)] = (cv.buf[i], cv.buf[i + 1], cv.buf[i + 2])
    grain = Noise(rng, 8, 8)
    pad = s * 0.1
    if not relief:                            # 白文: solid ground of ink-red
        for py in range(y0, y1):
            for px in range(x0, x1):
                n = grain.at((px - x) / s, (py - y) / s)
                if n < 0.14:                  # worn patches of 印泥
                    continue
                cv.blend(px, py, red, 0.70 + 0.30 * n)
    cell = (s - 2 * pad) / max(1, len(text))
    polys = []
    for i, ch in enumerate(text):
        if font is None or not font.has(ch):
            continue
        polys += font.outline(ch, x + pad, y + pad + i * cell, cell,
                              fill_cell=True)
    if relief:
        fill_contours(cv, polys, red, 0.92, grain)
    else:
        fill_contours(cv, polys, lambda px, py: keep.get((px, py), red), 0.95)
    bw = max(1.1, s * 0.045)
    for a, b in (((x, y), (x + s, y)), ((x + s, y), (x + s, y + s)),
                 ((x + s, y + s), (x, y + s)), ((x, y + s), (x, y))):
        brush_seg(cv, a[0], a[1], b[0], b[1], bw * 0.5, bw * 0.5, red,
                  0.9 if relief else 0.8)


# --------------------------------------------------------------------------
# 章法 — composition
# --------------------------------------------------------------------------

TITLE_A = ["春风", "晓露", "秋光", "寒香", "烟雨", "幽谷", "清秋", "朝阳",
           "月下", "东篱", "疏影", "露华", "金风", "碧池", "小园", "新晴",
           "晚凉", "山窗"]
TITLE_B = ["图", "小景", "清趣", "图卷", "写生", "小品", "册页"]
TIANGAN = "甲乙丙丁戊己庚辛壬癸"
DIZHI = "子丑寅卯辰巳午未申酉戌亥"
ACTION = ["写生", "戏作", "写意", "并记", "南窗写生", "客窗戏墨", "拟古"]
SEAL_WORDS = ["清赏", "墨缘", "闲心", "长乐", "知春", "石友", "听雨", "秋声",
              "写心", "心迹", "一枝"]


class Occupancy:
    """Where paint already sits, so the 题款 can be given real 留白.

    Every element registers a rough bounding disc; candidate text blocks are
    then scored against them. Cheap, and good enough for a few dozen shapes.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.items = []

    def add(self, x, y, r):
        if r > 0.5:
            self.items.append((x, y, r))

    def add_path(self, pts, w0, w1, step=2):
        n = max(1, len(pts) - 1)
        for i in range(0, len(pts), step):
            self.add(pts[i][0], pts[i][1], (w0 + (w1 - w0) * i / n) * 1.3)

    def overlap(self, x0, y0, x1, y1, margin=0.0):
        """Squared penetration of the rect into painted matter; 0 == clear."""
        tot = 0.0
        for cx, cy, r in self.items:
            dx = x0 - cx if cx < x0 else (cx - x1 if cx > x1 else 0.0)
            dy = y0 - cy if cy < y0 else (cy - y1 if cy > y1 else 0.0)
            rr = r + margin
            d2 = dx * dx + dy * dy
            if d2 < rr * rr:
                tot += (rr - math.sqrt(d2)) ** 2
        return tot

    def place(self, bw, bh, margin, corner_bias=1.0):
        """Best top-left corner for a bw x bh block: clear of paint, and
        tucked toward an edge the way an inscription actually sits."""
        W, H, pad = self.w, self.h, margin
        xs = [pad + i * max(0.0, W - 2 * pad - bw) / 8.0 for i in range(9)]
        ys = [pad + i * max(0.0, H - 2 * pad - bh) / 10.0 for i in range(11)]
        best, best_score = (pad, pad), None
        for y in ys:
            for x in xs:
                ov = self.overlap(x, y, x + bw, y + bh, margin * 0.8)
                cx, cy = x + bw * 0.5, y + bh * 0.5
                corner = min(math.hypot(cx - ex, cy - ey)
                             for ex in (0.0, W) for ey in (0.0, H))
                score = ov * 90.0 + corner * corner_bias + cy * 0.15
                if best_score is None or score < best_score:
                    best, best_score = (x, y), score
        return best

    def clear_spot(self, rng, anchors, dist, r, tries=40):
        """A free point out among the flowers — for the butterfly."""
        W, H = self.w, self.h
        for _ in range(tries):
            ax, ay, ar = anchors[rng.randrange(len(anchors))]
            a = rng.uniform(0, math.tau)
            d = ar + dist * rng.uniform(0.6, 1.5)
            x = ax + math.cos(a) * d
            y = ay + math.sin(a) * d
            if not (r < x < W - r and r < y < H - r):
                continue
            if self.overlap(x - r, y - r, x + r, y + r, r * 0.12) == 0.0:
                return x, y
        return None


def vocabulary():
    """Every character the painter can write — the extractor bakes just these."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [s["cn"] for s in SPECIES.values()]
             + [s["season"] for s in SPECIES.values()]
             + [TIANGAN, DIZHI, "日月年写于图"])
    return sorted(set("".join(words)))


def _load_font():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import gongbi_font as gf
    except ImportError:
        return None
    try:
        return BrushFont(gf.DATA, gf.EM)
    except Exception as exc:                 # corrupt blob: paint without 题款
        print("  (font blob unreadable: %s)" % exc, file=sys.stderr)
        return None


FONT = _load_font()


def compose(cv, rng, key, spec):
    W, H = cv.w, cv.h
    S = min(W, H)
    set_light(rng)                           # one light for the whole scene
    ground = paint_ground(cv, rng)

    flip = rng.choice((1, -1))               # entry side of the branch
    style = spec["branch"]["style"]
    bw0, bw1 = spec["branch"]["w"]
    scale = S / 1000.0
    bw0 *= scale
    bw1 *= scale

    def mx(fx):                              # mirror helper
        return fx if flip > 0 else 1.0 - fx

    # --- skeleton of the main branch ---------------------------------
    if style == "reed":                      # lotus: vertical stems from water
        p0 = (W * mx(rng.uniform(0.30, 0.46)), H * 1.02)
        p3 = (W * mx(rng.uniform(0.40, 0.62)), H * rng.uniform(0.22, 0.38))
        p1 = (p0[0] + W * 0.04 * flip, H * 0.75)
        p2 = (p3[0] - W * 0.06 * flip, H * 0.48)
    else:
        p0 = (W * mx(-0.04), H * rng.uniform(0.74, 0.98))
        p3 = (W * mx(rng.uniform(0.62, 0.86)), H * rng.uniform(0.14, 0.34))
        p1 = (W * mx(rng.uniform(0.18, 0.34)), H * rng.uniform(0.62, 0.82))
        p2 = (W * mx(rng.uniform(0.34, 0.56)), H * rng.uniform(0.30, 0.56))
    main = bezier(p0, p1, p2, p3, 30)
    if style == "zigzag":                    # 梅枝 — angular, brittle joints
        main, _ = knuckle(rng, main, k=rng.randint(3, 5), amp=0.30)
    elif style == "reed":                    # 荷茎 — nearly straight, slight sway
        main, _ = knuckle(rng, main, k=1, amp=0.07)
    else:
        main, _ = knuckle(rng, main, k=rng.randint(2, 3), amp=0.16)

    occ = Occupancy(W, H)
    mist = Noise(rng, 4, 5)                  # 虚实 — where the air thickens

    # --- 衬枝 — a pale sprig standing behind the subject --------------
    # The 远景 foliage used to be scattered at a random offset from a point on
    # the main branch — up to a tenth of the sheet away, each leaf with its
    # petiole pointing off into bare paper. At 远景 alpha those leaves are not
    # faint enough to read as atmosphere, so they read as mistakes; and for the
    # species that carry only one or two leaves on the branch itself (山茶,
    # 玉兰, 梅) they are *most* of the foliage in the picture. Hence "many of
    # the leaves are not attached to the stems" — they weren't, by
    # construction. In 工笔 the pale leaves of the 远景 belong to a 衬枝, a
    # second sprig standing behind the subject; give them one and every leaf on
    # the paper grows out of wood. Drawn here, ahead of the twigs and the main
    # branch, so the near wood laps over it.
    wmid = bw0 * 0.55 + bw1 * 0.45
    back = []
    for _ in range(rng.randint(1, 2)):
        j = rng.randrange(int(len(main) * 0.15), int(len(main) * 0.80))
        ax, ay = main[j]
        a0 = tangent(main, j) + rng.choice((1, -1)) * rng.uniform(0.40, 1.15)
        ln = S * rng.uniform(0.30, 0.55)
        q1 = (ax + math.cos(a0) * ln * 0.35, ay + math.sin(a0) * ln * 0.35)
        q3 = (ax + math.cos(a0) * ln, ay + math.sin(a0) * ln)
        q2 = (q3[0] - math.cos(a0 + rng.uniform(0.6, 1.2)) * ln * 0.30,
              q3[1] - math.sin(a0 + rng.uniform(0.6, 1.2)) * ln * 0.30)
        bp, _ = knuckle(rng, bezier((ax, ay), q1, q2, q3, 16),
                        k=rng.randint(1, 2), amp=0.20, start=0.20)
        # Thinner than the near wood and washed toward the paper, so it sits
        # back without the eye having to be told — but sized off the *mid*
        # width of the main branch, like a twig. Taken off the branch's tip
        # width it came out a hairline carrying leaves thirty times its own
        # width: attached in arithmetic, floating to the eye.
        hw = max(1.6, wmid * rng.uniform(0.52, 0.74))
        haze = 0.55 + 0.75 * mist.at(clamp01(q3[0] / W), clamp01(q3[1] / H))
        # and carried well above the alpha of its own leaves. A three-pixel
        # line covers a fortieth of the paper a blade does, so equal alpha is
        # nothing like equal weight; at the leaves' own alpha the sprig sank
        # into the silk and they were back to floating, only now in rows.
        pale = clamp01(0.74 * haze)
        brush_path(cv, bp, hw, hw * 0.50,
                   mix(P("zheshi"), P("baifen"), 0.30), pale)
        back.append((bp, hw, hw * 0.50))

    # --- 远景 — far foliage on the 衬枝, thin and cool for depth (虚) --
    # Spaced wider than the near foliage: these are big pale shapes, and two
    # of them half-overlapping make a cloud with a crease in it rather than
    # two leaves. 对生 is switched off back here for the same reason — a pair
    # at one node is a detail, and the 远景 carries none. 荷叶 need the most
    # room of all: a round leaf fills its disc, so two of them overlapping
    # cross in two hard crescents with nothing to disguise them, where a
    # pointed leaf only ever meets its neighbour obliquely.
    rnd_leaf = spec["leaf"].get("form") == "round"
    for lx, ly, a, L, bur in leaf_sites(rng, back, rng.randint(5, 11),
                                        S * 0.10, S * 0.20, span=(0.5, 1.5),
                                        sep=1.06 if rnd_leaf else 0.92,
                                        grade=0.20, pair=0.0,
                                        room=-0.05 if rnd_leaf else -0.16,
                                        sheet=(W, H)):
        haze = 0.55 + 0.75 * mist.at(clamp01(lx / W), clamp01(ly / H))
        paint_leaf(cv, rng, spec, lx, ly, a, L,
                   alpha=clamp01(rng.uniform(0.30, 0.48) * haze), far=True,
                   depth=bur)
        occ.add(lx + math.cos(a) * L * 0.5, ly + math.sin(a) * L * 0.5, L * 0.5)

    # --- twigs -------------------------------------------------------
    twigs = []
    for _ in range(rng.randint(2, 4)):
        i = rng.randrange(int(len(main) * 0.25), int(len(main) * 0.9))
        bx, by = main[i]
        a = tangent(main, i) + rng.choice((1, -1)) * rng.uniform(0.35, 0.95)
        ln = S * rng.uniform(0.14, 0.34)
        q3 = (bx + math.cos(a) * ln, by + math.sin(a) * ln)
        q1 = (bx + math.cos(a) * ln * 0.35, by + math.sin(a) * ln * 0.35)
        q2 = (q3[0] - math.cos(a + 0.9) * ln * 0.25,
              q3[1] - math.sin(a + 0.9) * ln * 0.25)
        tw, _ = knuckle(rng, bezier((bx, by), q1, q2, q3, 12),
                        k=rng.randint(1, 2), amp=0.22, start=0.25)
        twigs.append(tw)

    for tw in twigs:
        draw_branch(cv, rng, tw, wmid * 0.8, bw1 * 0.7, spec)
        occ.add_path(tw, wmid * 0.8, bw1 * 0.7)
    draw_branch(cv, rng, main, bw0, bw1, spec)
    occ.add_path(main, bw0, bw1)

    # --- 叶 — leaves along the branches -------------------------------
    lf = spec["leaf"]
    nmin, nmax = lf["n"]
    # A leaf joins the stem at its *edge*, and the width profile has to be
    # carried along for that: `path[i]` is a point on the axis, and anchoring
    # the petiole there put its base half a stem-width inside the wood. Since
    # the leaves go down after the branch, the petiole was then painted across
    # the bark and out the far side — every leaf looked pinned through the
    # stem rather than growing off it. `leaf_sites` does that arithmetic and
    # the 叶序 both; a trifoliate leaf is three blades off one stalk, so it
    # needs more clearance than a single blade of the same nominal length.
    hosts = [(main, bw0, bw1)] + [(tw, wmid * 0.8, bw1 * 0.7) for tw in twigs]
    for bx, by, a, L, bur in leaf_sites(
            rng, hosts, rng.randint(nmin, nmax),
            S * lf["L"][0] * 0.22, S * lf["L"][1] * 0.22,
            sep={"trifoliate": 0.98, "round": 1.06}.get(lf.get("form"), 0.80),
            sheet=(W, H)):
        paint_leaf(cv, rng, spec, bx, by, a, L,
                   alpha=rng.uniform(0.9, 1.0), depth=bur)
        occ.add(bx + math.cos(a) * L * 0.5, by + math.sin(a) * L * 0.5, L * 0.55)

    # --- 花头 — blossoms at the branch ends and nodes ------------------
    colors = glaze_palette(rng, spec["colors"][rng.randrange(len(spec["colors"]))])
    base_R = S * 0.16 * spec["size"]
    sites = []
    for path, _hw0, _hw1 in hosts:
        sites.append(path[-1])
        if len(path) > 8 and rng.random() < 0.6:
            sites.append(path[int(len(path) * rng.uniform(0.55, 0.85))])
    rng.shuffle(sites)
    nb = rng.randint(*spec["blossoms"])
    nbud = rng.randint(*spec["buds"])

    flowers = []
    for sx, sy in sites:
        if len(flowers) >= nb:
            break
        k = len(flowers)
        R = base_R * (1.0 if k == 0 else rng.uniform(0.58, 0.86))
        sx += rng.uniform(-0.05, 0.05) * W
        sy += rng.uniform(-0.05, 0.05) * H
        # keep the heads off the edge — a cropped blossom reads as a mistake.
        # Petals reach about 1.4 R past the centre (ring offset + petal length).
        pad = R * 1.45
        sx = min(W - pad, max(pad, sx))
        sy = min(H - pad, max(pad, sy))
        # 聚散 — two heads that touch merge into one unreadable mass, so keep
        # them at least their combined reach apart (plum carries many small
        # blossoms and is allowed to crowd).
        sep = 0.80 if spec["size"] < 0.6 else 1.08
        if any(math.hypot(sx - fx, sy - fy) < (R + fr) * sep
               for fx, fy, fr, _ in flowers):
            continue
        flowers.append((sx, sy, R, 1.0 if k == 0 else rng.uniform(0.82, 1.0)))

    buds = []
    for k in range(nbud):
        path, _hw0, _hw1 = hosts[rng.randrange(len(hosts))]
        i = rng.randrange(max(1, int(len(path) * 0.4)), len(path))
        bx, by = path[i]
        a = tangent(path, i) + rng.uniform(-0.8, 0.8)
        buds.append((bx, by, base_R * rng.uniform(0.28, 0.46), a))

    for bx, by, R, a in buds:
        paint_bud(cv, rng, spec, bx, by, R, colors, a,
                  alpha=rng.uniform(0.92, 1.0))
        occ.add(bx + math.cos(a) * R * 0.5, by + math.sin(a) * R * 0.5, R * 1.2)
    flowers.sort(key=lambda f: f[2])         # small/far first
    for sx, sy, R, op in flowers:
        # 同株异色 — nudge each head a few degrees around the hue circle so a
        # cluster looks grown rather than stamped, without breaking the glaze.
        dh = rng.uniform(-7.0, 7.0)
        sk = rng.uniform(0.94, 1.08)
        cols = (hue_rotate(colors[0], dh, sk), hue_rotate(colors[1], dh, sk),
                hue_rotate(colors[2], dh, sk) if colors[2] else None)
        paint_blossom(cv, rng, spec, sx, sy, R, cols, openness=op)
        occ.add(sx, sy, R * 1.12)

    # --- 草虫 — one companion insect, 蝶 or 蜂 -------------------------
    # 工笔 gives a flower sheet an insect about half the time, and it is one
    # or the other: a butterfly and a bee in the same picture are two things
    # competing for the one gap the composition had to spare.
    #
    # They are placed differently, because they behave differently. A
    # butterfly *passes* — it hovers in open air at some distance, facing
    # anywhere. A bee *visits*: it sits close in to a blossom and points at
    # it. That one detail does more to make a bee read as a bee than any of
    # the painting of it.
    comp = rng.random()
    anchors = [(f[0], f[1], f[2]) for f in flowers]
    if flowers and comp < 0.30:
        bs = S * rng.uniform(0.055, 0.085)
        spot = occ.clear_spot(rng, anchors, bs * 1.4, bs * 1.05)
        if spot:
            paint_butterfly(cv, rng, spot[0], spot[1], bs,
                            rng.uniform(0, math.tau))
            occ.add(spot[0], spot[1], bs * 1.05)
    elif flowers and comp < 0.58:
        # 蜂多成三 — one, or a pair, or three working the same head. They are
        # settled one at a time so each is kept clear of the last.
        for _ in range(rng.choice((1, 1, 1, 2, 2))):
            bs = S * rng.uniform(0.026, 0.038)
            # Two radii, and they are not the same kind of number.
            #
            # `reach` is the bee's own: 0.78 of its body length, because the
            # antennae put its front at 0.72 and the swept wings its back at
            # 0.64. Asked for less, `clear_spot` happily settles a bee twenty
            # pixels from the top border and the sheet gets half a bee.
            #
            # `stand` is how far outside the blossom to look, and it is not a
            # taste number — it is forced. `clear_spot` proposes
            # `d = ar + stand · U(0.6, 1.5)` out from the flower's *edge*, but
            # then tests a box of half-width `reach`, with a `0.12 reach`
            # margin, against that blossom's registered disc, which is
            # `1.12 ar`. So a proposal survives its own anchor only if
            # `stand · U ≥ 0.12 ar + 1.12 reach`. Guessing a stand-off scaled
            # to the *bee* leaves the whole ring under that bound for a large
            # blossom — every try refused before it has even looked at the rest
            # of the painting, which is how a 28 % companion came out at 17 %.
            #
            # And it has to be solved against the blossom this bee is actually
            # visiting, which is why the head is drawn *first* and `clear_spot`
            # is given just the one anchor. Solving it against the biggest head
            # on the sheet also raises the acceptance rate, but it stands a bee
            # off a small flower by a big flower's width, and a bee hovering in
            # open paper facing a blossom it is nowhere near is not a bee
            # visiting a flower. It is a fly.
            host = anchors[rng.randrange(len(anchors))]
            reach = bs * 0.78
            stand = (0.14 * host[2] + 1.25 * reach) / 0.6
            spot = occ.clear_spot(rng, [host], stand, reach, tries=60)
            if not spot:
                break
            paint_bee(cv, rng, spot[0], spot[1], bs,
                      math.atan2(host[1] - spot[1], host[0] - spot[0])
                      + rng.gauss(0.0, 0.42))
            # 1.5 reach, not reach: `reach` is the body's half-length, but the
            # legs and the swept wings put a bee's *visible* width at better
            # than its length, and two of them settled a bare body-length apart
            # merge into one dark smudge.
            occ.add(spot[0], spot[1], reach * 1.5)

    # --- 题款 — title, signature and seals, written into the 留白 -------
    title = (TITLE_A[rng.randrange(len(TITLE_A))] + spec["cn"]
             + TITLE_B[rng.randrange(len(TITLE_B))])
    sign = (TIANGAN[rng.randrange(10)] + DIZHI[rng.randrange(12)]
            + spec["season"] + rng.choice(("日", "月"))
            + ACTION[rng.randrange(len(ACTION))])
    if FONT is not None:
        # Measure the whole block (title column, signature column, seal),
        # then look for somewhere it can sit without touching the painting.
        gs = S * rng.uniform(0.034, 0.044)
        ss = gs * 0.62
        lead, lead_s = 1.22, 1.18
        gap = gs * 0.42
        sign_dy = gs * rng.uniform(1.2, 2.2)
        seal_s = gs * rng.uniform(0.82, 1.0)
        title_h = len(title) * gs * lead
        sign_h = sign_dy + len(sign) * ss * lead_s + gs * 0.25 + seal_s
        bw = gs + gap + ss
        bh = max(title_h, sign_h)
        margin = S * 0.035
        if bw < W - 2 * margin and bh < H - 2 * margin:
            bx, by = occ.place(bw, bh, margin)
            x_title = bx + gap + ss        # columns read right to left
            x_sign = bx
            paint_column(cv, rng, FONT, title, x_title, by, gs, lead=lead)
            end_s = paint_column(cv, rng, FONT, sign, x_sign, by + sign_dy, ss,
                                 lead=lead_s, alpha=(0.66, 0.86))
            paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                       x_sign + (ss - seal_s) * 0.5, end_s + gs * 0.2, seal_s)
            occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
        if rng.random() < 0.5:               # 压角章 in a corner, if it fits
            s2 = S * rng.uniform(0.038, 0.052)
            spot = None
            for cx, cy in ((W * 0.08, H * 0.88), (W * 0.88 - s2, H * 0.88),
                           (W * 0.08, H * 0.08), (W * 0.88 - s2, H * 0.08)):
                if occ.overlap(cx, cy, cx + s2, cy + s2, s2 * 0.5) == 0.0:
                    spot = (cx, cy)
                    break
            if spot:
                paint_seal(cv, rng, FONT,
                           SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                           spot[0], spot[1], s2)
    return title, ground


def age(cv, rng):
    """Final unifying glaze: specks, a faint ink-dust veil (罩染 over all)."""
    W, H = cv.w, cv.h
    for _ in range(int(W * H * 0.00012)):
        disc(cv, rng.uniform(0, W), rng.uniform(0, H),
             rng.uniform(0.6, 1.8), P("danmo"), rng.uniform(0.08, 0.3))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def paint(path, size=(1000, 1360), seed=None, species=None, quiet=False):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    key = species or sorted(SPECIES)[rng.randrange(len(SPECIES))]
    if key not in SPECIES:
        raise SystemExit("unknown species %r; try --list" % key)
    spec = SPECIES[key]

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, key, spec)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-28s %s  %s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, spec["cn"], ground, seed,
                 size[0], size[1], time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random vivid Chinese Gongbi-style flower paintings.")
    ap.add_argument("-o", "--out", default="gongbi.png", help="output PNG path")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--species", help="peony|lotus|plum|chrysanthemum|magnolia|camellia")
    ap.add_argument("--size", default="1000x1360", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("花卉 repertoire:")
        for k, v in sorted(SPECIES.items()):
            print("  %-15s %s  (%s)  %d 层花瓣" % (k, v["cn"], v["season"],
                                                 len(v["rings"])))
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 1000x1360")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("工笔花卉 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.species)
    return 0


if __name__ == "__main__":
    sys.exit(main())
