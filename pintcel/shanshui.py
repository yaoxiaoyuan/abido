#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shanshui.py — 随机生成中国山水画 / Random Chinese shanshui landscape paintings.

山水 ("mountains and water") is landscape painting as the Song and Yuan
masters practised it, and like 工笔 it is a *procedure* rather than a look.
This generator is built as that procedure:

    纸本   paper ground, and a snapshot of it kept for 留白
    布局   三远 — the composition is chosen as 高远 / 深远 / 平远
    勾     the ridge line of every rock mass, drawn in ink
    皴     the texture strokes that say what the rock is made of
           (披麻 / 斧劈 / 雨点 / 折带 / 卷云 — one school per painting)
    擦     dry scrubbing over the 皴
    染     graded ink washes: dark under the crest, fading at the feet
    点     苔点 moss, 点叶 foliage, 远树 distant trees
    树石   trees drawn 干-枝-叶, rocks, 坡岸 banks
    云水   mist bands and water, both of them 留白 rather than paint
    屋舟   a pavilion, a hut, a bridge, a boat — the human scale
    题款   inscription column and seals in the empty quarter

The idea carried over from gongbi.py is that **a shape is a field, not a
polygon**. There a petal was a signed field in its own frame; here a mountain
is a *height field* `y = top(x)`, which hands you the silhouette, exact
anti-aliasing, slope-direction lighting and a coordinate system for the 皴 all
in a single column scan. See docs/HOW-IT-WORKS.md §2-§3 for that argument.

Infrastructure (PNG writer, value noise, paper ground, brush strokes, the
baked calligraphy outlines, seals, the occupancy map) is shared with
gongbi.py rather than copied — this module imports it.

Usage
-----
    python3 shanshui.py                          # one random landscape
    python3 shanshui.py -n 6                     # six of them
    python3 shanshui.py --scene 高远 --cun 斧劈
    python3 shanshui.py --style 青绿 --seed 7
    python3 shanshui.py --size 1000x1500 -o scroll.png
    python3 shanshui.py --list                   # repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Noise, P, rgb, mix, jitter, hsv_mix, saturate,
                    bell, disc, brush_seg, brush_path, bezier, knuckle,
                    paint_column, paint_seal, Occupancy, BrushFont,
                    TIANGAN, DIZHI)

# --------------------------------------------------------------------------
# 墨色 — the five tones of ink, and the three colour schools of landscape
# --------------------------------------------------------------------------

INK = {
    "jiao": "#1B1917",   # 焦墨  burnt — the darkest accents only
    "nong": "#2C2825",   # 浓墨  thick
    "zhong": "#4A4337",  # 重墨  heavy
    "dan":  "#807868",   # 淡墨  diluted
    "qing": "#AAA496",   # 清墨  clear — a breath of grey
}


def I(name):
    return rgb(INK[name])


# Each style gives the two stops of the rock ramp (pale wash -> deep pigment)
# for near and far masses, plus the ink the trees are drawn with. 浅绛 and
# 青绿 are the two historical colour schools; 水墨 is ink alone.
STYLES = {
    "水墨": dict(
        cn="水墨", en="ink only",
        near=("#C2BCAB", "#8A8579", "#262320"),
        far=("#CAC5B6", "#A8A396", "#6E6A5E"),
        tree="#2A2724", leaf="#3A362F", bow=0.03,
    ),
    "浅绛": dict(                       # ink drawing warmed with 赭石
        cn="浅绛", en="ochre-tinted ink",
        near=("#D2B38B", "#A3663C", "#2B241C"),
        far=("#C6BBA2", "#9A9080", "#6A665A"),
        tree="#312A22", leaf="#4A4328", bow=0.16,
    ),
    "青绿": dict(                       # 石绿 and 石青 over ink structure
        cn="青绿", en="azurite and malachite",
        # 石绿 is a mineral ground in water, not a dye: on silk it sits pale
        # and slightly grey, and the ink drawing reads straight through it.
        # Mixed at full chroma the mid stop swamped the 皴 and a wide flank
        # came out as a lawn — a green hill with no rock in it. Pulled back
        # toward 花青 and lightened, the same wash lets the strokes show.
        near=("#C6D1BC", "#6F9881", "#222C2E"),
        far=("#BCC9C4", "#8AA2A6", "#546A6E"),
        tree="#2B2A24", leaf="#415C3E", bow=0.14,
    ),
}

# 皴法 — the texture-stroke grammars. Each is one school's answer to "what is
# this rock made of", and each is a different cheap expression in the inner
# loop of Ridge.render (mode 0..4).
CUN = {
    "披麻": dict(mode=0, cn="披麻皴", who="董源·黄公望",
                 per=(7.0, 13.0), amp=(0.30, 0.46), scrub=0.10),
    "斧劈": dict(mode=1, cn="斧劈皴", who="李唐·马远",
                 per=(10.0, 18.0), amp=(0.36, 0.54), scrub=0.12),
    "雨点": dict(mode=2, cn="雨点皴", who="范宽",
                 per=(1.0, 1.0), amp=(0.30, 0.44), scrub=0.14),
    "折带": dict(mode=3, cn="折带皴", who="倪瓒",
                 per=(9.0, 16.0), amp=(0.26, 0.40), scrub=0.08),
    "卷云": dict(mode=4, cn="卷云皴", who="郭熙",
                 per=(14.0, 24.0), amp=(0.30, 0.46), scrub=0.12),
}

# 光 — one light for the whole painting (see gongbi.py §3.2 for why this
# matters). In a landscape it only has to say which flank of each ridge is
# turned away; it is never a cast shadow.
LIGHT = (-0.62, -0.78)


def set_light(rng):
    global LIGHT
    a = rng.uniform(-2.45, -0.70)        # pointing up-left .. up-right
    LIGHT = (math.cos(a), math.sin(a))
    return a


# --------------------------------------------------------------------------
# 擦 — one tileable grain, baked to pixel resolution and read by index
# --------------------------------------------------------------------------

class Grain:
    """Tileable value noise, interpolated once into a flat array.

    The 皴 loop wants noise at every pixel of every mountain. A bilinear
    lattice lookup there (gongbi's `Noise.at`) would cost more than the rest
    of the shading put together, so the smoothing is paid once at build time
    and the loop does one multiply-add and one list index.

    One tile serves many frequencies, but only in one direction. Reading it
    at `(x << 2, y << 2)` quadruples the frequency and stays smooth, because
    every pixel still lands on a different entry of the baked array. Reading
    it at `(x >> 2, y >> 2)` quarters the frequency and does *not*: sixteen
    neighbouring pixels share one entry, so a broad tonal wash taken that way
    comes out as a visible grid of hard-edged 16 px rectangles. That grid was
    plainly readable in the 擦 and 积墨 passes.

    So: bake at the *coarsest* scale you need and shift up. Two tiles —
    8 px/cell for fibre-scale noise, 128 px/cell for facet-scale washes —
    cover 2 px, 8 px, 32 px and 128 px with no blocking anywhere.
    """

    __slots__ = ("n", "m", "g", "bn", "bm", "b")

    def __init__(self, rng, n=256, cells=32, bn=512, bcells=4):
        self.n, self.m, self.g = n, n - 1, self._bake(rng, n, cells)
        self.bn, self.bm, self.b = bn, bn - 1, self._bake(rng, bn, bcells)

    @staticmethod
    def _bake(rng, n, cells):
        lat = [[rng.random() for _ in range(cells + 1)] for _ in range(cells + 1)]
        for i in range(cells + 1):           # wrap, so the tile is seamless
            lat[i][cells] = lat[i][0]
        for j in range(cells + 1):
            lat[cells][j] = lat[0][j]
        s = n / cells
        inv = 1.0 / s
        g = [0.0] * (n * n)
        for y in range(n):
            fy = y * inv
            jy = int(fy)
            ty = fy - jy
            ty = ty * ty * (3.0 - 2.0 * ty)
            r0, r1 = lat[jy], lat[jy + 1]
            row = y * n
            for x in range(n):
                fx = x * inv
                jx = int(fx)
                tx = fx - jx
                tx = tx * tx * (3.0 - 2.0 * tx)
                a = r0[jx] + (r0[jx + 1] - r0[jx]) * tx
                b = r1[jx] + (r1[jx + 1] - r1[jx]) * tx
                g[row + x] = a + (b - a) * ty
        return g

    def at(self, x, y):
        m = self.m
        return self.g[((y & m) * self.n) + (x & m)]

    def broad(self, x, y):
        m = self.bm
        return self.b[((y & m) * self.bn) + (x & m)]


# --------------------------------------------------------------------------
# 山形 — generating a ridge line
# --------------------------------------------------------------------------

def _octave(rng, n, k):
    """One octave of 1-D value noise: 2^k random knots, smoothstepped to
    n+1 columns."""
    m = 1 << k
    v = [rng.uniform(-1.0, 1.0) for _ in range(m + 1)]
    out = [0.0] * (n + 1)
    sc = m / float(n)
    for i in range(n + 1):
        t = i * sc
        j = int(t)
        if j >= m:
            j = m - 1
        f = t - j
        f = f * f * (3.0 - 2.0 * f)
        out[i] = v[j] + (v[j + 1] - v[j]) * f
    return out


def fbm(rng, n, rough=0.62, amp=1.0, lo=2, hi=7):
    """Sum of octaves, each twice the frequency and `rough` the amplitude.

    The obvious thing — midpoint displacement — concentrates almost all its
    amplitude in the lowest frequency, so a mountain generated with it comes
    out a clean triangle with a slight wobble. That was the first thing wrong
    with these ridges. Summing explicit octaves keeps real amplitude at the
    frequency where a ridge line actually crumples: three or four shoulders
    per peak, each with its own smaller ones.
    """
    out = [0.0] * (n + 1)
    a, s = 1.0, 0.0
    for k in range(lo, hi):
        o = _octave(rng, n, k)
        for i in range(n + 1):
            out[i] += o[i] * a
        s += a
        a *= rough
    inv = amp / s
    return [v * inv for v in out]


def summits(rng, n, k, spire=(0.55, 1.5), wmin=0.16, hw=0.0):
    """The large shape: k asymmetric humps, combined with **max** rather than
    a sum, because mountains occlude one another instead of adding up. The
    exponent is the whole character of the range: below 1 the flanks bulge out
    (土山, 卷云皴 country), above 1 they fall away concave into a spire."""
    def exp_(lo, hi):
        # Exponent 1 is a straight flank — a cone. Step over the band around
        # it so every hump is either concave (a spire) or convex (a 土山).
        e = rng.uniform(lo, hi)
        if 0.88 < e < 1.14:
            e = 0.80 if rng.random() < 0.5 else 1.22
        return e

    hs = []
    for i in range(k):
        c = rng.uniform(0.10, 0.90) if i else rng.uniform(0.28, 0.72)
        h = 1.0 if i == 0 else rng.uniform(0.34, 0.86)
        # Each hump needs its own aspect floor, not just the mass. `wmin`
        # keeps a narrow *mass* from being a needle, but inside a wide span a
        # single tall hump can still draw one: that is what the middle
        # distance's row of pointed cones was. A hump is at least about as
        # wide as it is tall. (`hw` is the mass's height over its span, so
        # `h * hw` is this hump's height measured in spans.)
        need = h * hw * 0.75
        wl = rng.uniform(wmin, wmin + 0.36)
        wr = rng.uniform(wmin, wmin + 0.36)
        hs.append((c, h,
                   need if wl < need else wl,
                   need if wr < need else wr,
                   exp_(*spire), exp_(*spire)))
    env = [0.0] * (n + 1)
    for i in range(n + 1):
        u = i / float(n)
        best = 0.0
        for c, h, wl, wr, el, er in hs:
            if u <= c:
                z = (u - c + wl) / wl
                v = h * z ** el if z > 0.0 else 0.0
            else:
                z = (c + wr - u) / wr
                v = h * z ** er if z > 0.0 else 0.0
            if v > best:
                best = v
        env[i] = best
    return env


def cliffs(rng, top, k, amp):
    """断崖 — a few vertical breaks in the silhouette.

    A ridge that is pure fBm reads as a heap of gravel; real mountains have
    sheer faces. Each break drops the line at one column and lets it climb
    back over a short run, which is a cliff seen edge-on. (Same idea as
    knuckle() on a gongbi branch: keep the gesture, lose the smoothness.)
    """
    n = len(top) - 1
    for _ in range(k):
        j = rng.randrange(int(n * 0.12), int(n * 0.88) + 1)
        w = max(3, int(n * rng.uniform(0.03, 0.13)))
        d = amp * rng.uniform(0.35, 1.0) * rng.choice((1.0, -1.0))
        # The riser is steep, not infinite. Dropped in one column, a 20 px
        # break gives the 勾 line a single aliased vertical bar, and two or
        # three of those along a crest read as a row of needles rather than
        # as cliffs. Over three or four columns the same break is still a
        # sheer face and the contour stays a drawn line.
        r = rng.randint(2, 5)
        for i in range(j, min(n + 1, j + r)):
            t = (i - j + 1) / float(r)
            top[i] += d * t * t * (3.0 - 2.0 * t)
        for i in range(j + r, min(n + 1, j + r + w)):
            top[i] += d * (1.0 - (i - j - r) / float(w))
    return top


def ridge(rng, span, height, peaks=2, rough=0.55, detail=0.26, breaks=2,
          spire=(0.55, 1.5), flat=0.0):
    """One ridge line, `span` columns wide, rising `height` px at the summit.

    Returns a list of *offsets above the base line*, zero at both feet, so a
    mass never has a vertical edge unless the canvas cuts it off.
    """
    n = max(4, int(span))
    # A hump's width is a fraction of the span, so on a tall narrow mass the
    # narrowest draw is a needle. Raise the floor with the aspect ratio: a
    # peak may be steep, but a mountain is never a spear.
    hw = height / max(1.0, float(span))
    env = summits(rng, n, peaks, spire,
                  wmin=0.16 + 0.24 * min(1.0, hw), hw=hw)
    det = fbm(rng, n, rough, 1.0)
    # index.html's crest is `cos(x) * noise(x)` — a *product*, not a sum, and
    # that is most of why its mountains have no straight edge anywhere: every
    # flank is the envelope bent by a slow wave. Added detail cannot do this
    # job, because detail is scaled by the envelope and so dies exactly where
    # the long straight runs are; a multiplier bends the whole flank. Without
    # it a hump whose exponent sits near 1 draws a 200 px ruled diagonal, and
    # one ruled diagonal is enough to make the whole sheet look computed.
    bend = fbm(rng, n, 0.55, 1.0, lo=1, hi=5)
    # ...and the last thing the silhouette needs is *incident*. Measured
    # against a chord, a flank built from an envelope plus fBm already bows
    # by a sixth of its own height — and still reads as drafted, because the
    # bow is smooth and a drawn line never is. What the eye looks for is the
    # small stuff: a kink every dozen pixels, a notch, a step. index.html
    # gets it for free, since its crest is a 50-point polyline with noise at
    # every point; here it is one more band of fBm at the top of the
    # spectrum, where `detail` has almost no amplitude left to give.
    nib = fbm(rng, n, 0.62, 1.0, lo=5, hi=8)
    edge = max(1.2, height * 0.018)
    top = [0.0] * (n + 1)
    for i in range(n + 1):
        e = env[i] * (1.0 + 0.32 * bend[i])
        if e < 0.0:
            e = 0.0
        # detail is scaled by the envelope, so the crumple dies away at the
        # feet instead of making the foot of the mountain look chewed
        top[i] = (height * (e + detail * det[i] * (e ** 0.8))
                  + edge * nib[i] * (e ** 0.4))
    if breaks:
        cliffs(rng, top, breaks, height * 0.075)
    if flat > 0.0:                      # 平台 — shear the top off a mesa
        # Cutting at a fixed fraction of the height shears a *spire* into a
        # tower: the cut is only a few columns wide and the sides stay
        # vertical, which is Monument Valley and not 倪瓒. Choose the cap by
        # rank instead, so the plateau is a guaranteed share of the span and
        # the mass always reads as a low flat bank.
        srt = sorted(top, reverse=True)
        frac = flat * rng.uniform(0.25, 0.55)
        cap = srt[min(len(srt) - 1, int(n * frac))]
        if cap < height * 0.45:
            cap = height * 0.45
        # ...and the plateau itself must not be a ruled line, so the clipped
        # run gets its own slow crumple
        pw = fbm(rng, n, 0.5, height * 0.07, lo=1, hi=4)
        top = [min(v, cap + pw[i]) for i, v in enumerate(top)]
    # A convex hump (exponent below 1) is near-vertical at its foot, which is
    # exactly what the near-plane 坡岸 wants — but when such a foot lands
    # inside the frame instead of off the edge, the mass terminates in a
    # sheer vertical cut a hundred pixels tall. The composition then trims
    # the ridge to its support and the cut becomes the silhouette.
    # A wedge clamp at each end fixes it without touching anything else: it
    # only ever lowers columns that stand higher than the wedge allows, so a
    # gradual foot is left alone and a cliff foot gets a steep taper.
    ramp = max(5.0, min(n * 0.13, height * 0.38))
    lim = min(4.2, height * 0.9 / ramp)         # ~77° is as steep as a foot
    if lim * ramp < height * 0.25:              # ...but never clip the mass
        lim = height * 0.25 / ramp              # down to a stump
    rise = lim * ramp
    # The wedge must not be a *straight* ramp, and no amount of noise on a
    # straight ramp fixes it. Measured off the paper, a flank clamped by a
    # constant slope came out ruled to within a third of a pixel over 150
    # rows: `bend` and `detail` both have their amplitude at wavelengths of
    # a hundred columns or more, so across a 50-column wedge they are
    # locally linear and the clamp inherits a perfect edge. What a rock foot
    # actually does is *step* — two or three terraces at different angles —
    # so the wedge is a three-segment profile with random knees, and only
    # then is it bent and roughened.
    prof = []
    for _ in range(2):
        us = sorted((rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82)))
        vs = sorted((rng.uniform(0.10, 0.90), rng.uniform(0.10, 0.90)))
        prof.append(((0.0, 0.0), (us[0], vs[0]), (us[1], vs[1]), (1.0, 1.0)))
    for i in range(n + 1):
        if top[i] < 0.0:
            top[i] = 0.0
        e = i if i < n - i else n - i
        if e < ramp:
            u = e / ramp
            p = prof[0] if i < n - i else prof[1]
            for q in range(3):
                ax, ay = p[q]
                bx, by = p[q + 1]
                if u <= bx or q == 2:
                    w = (ay if bx <= ax else
                         ay + (by - ay) * (u - ax) / (bx - ax)) * rise
                    break
            w *= 1.0 + 0.22 * bend[i]
            if top[i] > w:
                top[i] = w + edge * nib[i] * 1.2
                if top[i] < 0.0:
                    top[i] = 0.0
    top[0] = top[n] = 0.0
    return top


# --------------------------------------------------------------------------
# 山石 — THE primitive. One rock mass as a height field over x.
#
# Silhouette, anti-aliasing, lighting, the ink wash and the 皴 are all
# functions of (column, depth below the crest), evaluated in one column scan.
# Everything in the painting that is rock — 主峰, 远山, 坡岸, 石 — is this
# class with different parameters, exactly as every petal and leaf in
# gongbi.py is one Blade.
# --------------------------------------------------------------------------

NS = 48                                  # resolution of the ink ramp


class Ridge:
    __slots__ = ("x0", "top", "base", "light", "mid", "deep", "tone", "a0",
                 "a1", "w_light", "w_cur", "w_line", "line", "w_crest",
                 "crest", "w_foot", "foot", "cun", "per", "amp", "scrub",
                 "bow", "fade", "grain")

    def __init__(self, x0, top, base, light, mid, deep,
                 tone=0.30, a0=0.10, a1=0.80,
                 w_light=0.22, w_cur=0.10, w_line=0.55, line=2.2,
                 w_crest=0.26, crest=0.30, w_foot=0.30, foot=0.75,
                 cun=-1, per=10.0, amp=0.36, scrub=0.10, bow=0.08,
                 fade=0.0, grain=None):
        self.x0 = int(x0)
        self.top = top                   # absolute y of the crest per column
        self.base = base                 # absolute y the mass is filled to
        self.light, self.mid, self.deep = light, mid, deep
        self.tone = tone                 # base ink density of this mass
        self.a0, self.a1 = a0, a1        # alpha floor / slope: paper breathes
        self.w_light = w_light           # how hard the light reads on a flank
        self.w_cur = w_cur               # 积墨 in the gullies
        self.w_line, self.line = w_line, line      # 勾 — the crest line
        self.w_crest, self.crest = w_crest, crest  # 染 under the crest
        self.w_foot, self.foot = w_foot, foot      # feet lighten into mist
        self.cun, self.per, self.amp = cun, per, amp
        self.scrub, self.bow = scrub, bow
        self.fade = fade                 # px of alpha fade above the base
        self.grain = grain

    # -- geometry queries used by the composition ------------------------

    def crest_y(self, x):
        i = int(x) - self.x0
        if i < 0 or i >= len(self.top):
            return None
        return self.top[i]

    def span(self):
        return self.x0, self.x0 + len(self.top) - 1

    def sample(self, rng, k, lo=0.08, hi=0.92, flatish=False):
        """k points along the crest. `flatish` prefers level ground, which is
        where a painter puts a hut, a pavilion or a boat."""
        n = len(self.top) - 1
        out = []
        for _ in range(k):
            best = None
            for _ in range(6 if flatish else 1):
                i = rng.randrange(int(n * lo), max(int(n * lo) + 1, int(n * hi)))
                if not flatish:
                    best = i
                    break
                g = abs(self.top[min(n, i + 2)] - self.top[max(0, i - 2)])
                if best is None or g < best[0]:
                    best = (g, i)
            i = best if not flatish else best[1]
            out.append((self.x0 + i, self.top[i]))
        return out

    def register(self, occ, step=12):
        n = len(self.top) - 1
        for i in range(0, n + 1, step):
            y = self.top[i]
            b = self.base if isinstance(self.base, float) else self.base[i]
            if b - y < 3.0:
                continue
            occ.add(self.x0 + i, (y + b) * 0.5, max(6.0, (b - y) * 0.42))

    # -- the renderer ----------------------------------------------------

    def render(self, cv):
        W, H, buf = cv.w, cv.h, cv.buf
        top = self.top
        n = len(top) - 1
        if n < 3:
            return
        x0 = self.x0

        # ---- the ink ramp, interpolated in HSV ------------------------
        # Three stops, not two: a pale wash, the body pigment, and ink. That
        # is the actual order of work in 青绿山水 — the whole structure is
        # drawn in ink first and the mineral colour laid between the lines —
        # and a two-stop ramp from pale green to dark green loses it, which
        # is why the first landscapes came out looking like boiled sweets.
        # Straight RGB interpolation would also drag the 石绿 toward grey
        # exactly where the 染 is working; see gongbi.py §4.1.
        half = NS >> 1
        bw = self.bow * 0.6
        ramp = [hsv_mix(self.light, self.mid, j / float(half), bow=bw)
                for j in range(half)]
        ramp += [hsv_mix(self.mid, self.deep, (j - half) / float(NS - half),
                         bow=bw) for j in range(half, NS + 1)]

        # ---- per-column tables: light, curvature, slope of the 皴 -----
        # Everything in this block is read off a *smoothed* crest, never off
        # `top` itself. The ridge line carries fine jitter by design — that
        # is what makes it look drawn — but none of the jitter is part of the
        # form. Take the slope straight off `top` and every kink in the
        # silhouette becomes a stripe of light running the whole height of
        # the mountain: the mass comes out combed, which is precisely what
        # adding the crest incident did to it at first.
        def sm(a):
            return ([a[0]] + [(a[i - 1] + 2.0 * a[i] + a[i + 1]) * 0.25
                              for i in range(1, n)] + [a[n]])

        tmid = top                      # the form, with the jitter taken off
        for _ in range(10):
            tmid = sm(tmid)
        # ...and the 皴 phase cannot be measured from the crest at all, fine
        # or not: depth-below-crest is shifted by a different amount in every
        # column, so the stroke band is cut and re-offset at each one, which
        # is what reduced every stroke to a stack of dashes. The strokes hang
        # off a crest smoothed much harder still.
        tsm = tmid
        for _ in range(14):
            tsm = sm(tsm)

        sl = [0.0] * (n + 1)
        cu = [0.0] * (n + 1)
        lx, ly = LIGHT
        for i in range(n + 1):
            a = tmid[i + 1] if i < n else tmid[n]
            b = tmid[i - 1] if i > 0 else tmid[0]
            g = (a - b) * 0.5                       # dy/dx, y pointing down
            inv = 1.0 / math.sqrt(1.0 + g * g)
            # outward normal of the slope; negative dot means "facing the
            # light", which has to *remove* ink rather than add it
            sl[i] = -(lx * (g * inv) + ly * (-inv))
            # >0 in a gully. Bounded, because at a cliff step the raw second
            # difference is tens of pixels and used neat it burns one column
            # to solid black — a bright or dark thread running the whole
            # height of the mountain.
            c = (a + b - 2.0 * tmid[i]) * 0.5
            cu[i] = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)
        # The lean of the 皴 — the direction the strokes run — has to be a
        # *slowly varying* field. Taken raw per column it changes under every
        # stroke, the phase comb no longer lines up with its neighbour, and a
        # long hemp-fibre stroke comes out as a column of 3 px dashes. It is
        # the derivative of the hard-smoothed crest, which costs nothing now
        # that the crest is already there.
        sk = [0.0] * (n + 1)
        for i in range(n + 1):
            v = ((tsm[i + 1] if i < n else tsm[n])
                 - (tsm[i - 1] if i > 0 else tsm[0])) * 0.5
            sk[i] = 1.6 if v > 1.6 else (-1.6 if v < -1.6 else v)
        for _ in range(6):
            sl = sm(sl)
            cu = sm(cu)

        base = self.base
        scalar_base = not isinstance(base, list)
        hmax = 8
        for i in range(n + 1):
            b = base if scalar_base else base[i]
            d = int(b - top[i]) + 3
            if d > hmax:
                hmax = d
        if hmax > H + 4:
            hmax = H + 4

        # ---- the wash, as a function of depth below the crest ---------
        # Two tables, because the 勾 line and the 染 wash are modulated
        # differently: the wash is the same all along the ridge, the line
        # thickens and breaks from column to column the way a real brush does.
        lw = self.line
        cd = max(2.0, self.crest * hmax)
        prw = [0.0] * (hmax + 1)
        amt = [0.0] * (hmax + 1)
        dec = [0.0] * (hmax + 1)
        amp0 = self.amp
        for dy in range(hmax + 1):
            z = dy / cd
            if z < 1.0:                             # 染 — dark under the crest
                prw[dy] = self.w_crest * (1.0 - z) ** 1.6
            # 皴 dies away toward the foot, and it also starts *below* the
            # crest: the ridge line itself is clean ink, and strokes run up
            # against it rather than over it. Stacking 勾, 染 and 皴 all at
            # dy=0 gave every mountain a furry edge.
            z = dy / float(hmax)
            r0 = dy / (cd * 0.55)
            amt[dy] = amp0 * (1.0 - 0.72 * z * z) * (0.30 + 0.70 * r0
                                                     if r0 < 1.0 else 1.0)
            # How far down the slope light still means anything. The slope is
            # measured from the ridge line, so it describes the rock just
            # under the crest and nothing else; applied at full strength all
            # the way down it paints a vertical stripe per column and the
            # mountain comes out looking combed.
            z = dy / (cd * 2.4)
            dec[dy] = 0.18 + 0.82 * (1.0 - z) ** 1.4 if z < 1.0 else 0.18

        # The foot is measured *up from the base*, never down from the crest.
        # The crest drops to the base at the skirts of every mass, so a foot
        # depth counted from the ridge never arrives there and each mass got
        # a hard line along its base wherever it ran out — flat-bottomed
        # trapezoids standing in the water. Counting up from the base the
        # fade always completes, and the band is also clamped to a fraction
        # of each column so a short skirt fades over itself instead of
        # going transparent all the way to the ridge.
        w_foot = self.w_foot
        foot_d = max(2.0, self.foot * hmax)

        mode = self.cun
        gr = self.grain
        gg, gm, gn = gr.g, gr.m, gr.n
        bg, bm, bn = gr.b, gr.bm, gr.bn
        inv_per = 1.0 / self.per
        scrub = self.scrub * 2.2
        blot = self.scrub * 3.2
        # 勾 weight per column: a contour of constant darkness is the single
        # loudest tell that a shape was filled rather than drawn
        row0 = ((self.x0 * 37 + n) & gm) * gn
        ln = [0.50 + 1.15 * gg[row0 + ((i >> 1) & gm)] for i in range(n + 1)]
        # ...and a brush line varies in *width* as well as in darkness — that
        # is most of what separates a drawn contour from a traced one. It is
        # the same thing index.html gets by offsetting its outline polygon by
        # `wid * noise(i)`; held as one width per column it costs nothing.
        # The low end of the draw goes under a pixel, so the ridge line
        # thins to nothing here and there and the eye closes the gap itself.
        lwd = [lw * (0.22 + 1.42 * gg[row0 + ((i >> 2) & gm)])
               for i in range(n + 1)]

        # 皴 is never laid evenly over a mass. Two things decide where the
        # brush went, and both are properties of a *column*, not of a pixel:
        #
        #   — the turn of the form. Texture belongs where the rock rolls
        #     away from you; a face that lies open to the light is left as
        #     bare paper and reads as light precisely because it is bare.
        #   — the painter's economy. Whole patches of a flank are simply not
        #     worked at all, and the bare paper between them is what makes
        #     the worked rock read as rock. index.html does this with its
        #     `dis()`, which throws every stroke into the outer third of the
        #     mass and leaves a white belly down the middle.
        #
        # The turn is a column field, because it is a property of the ridge
        # line. The economy must *not* be: held per column it paints the mass
        # in vertical bands, worked stripe beside bare stripe, which is a
        # curtain and not a mountain. It is read per pixel off the broad
        # tile instead, a few dozen pixels to a patch — see the loop below.
        tx = [0.0] * (n + 1)
        for i in range(n + 1):
            g = sk[i]
            v = 0.34 + 0.84 * (g if g > 0.0 else -g)
            tx[i] = 1.12 if v > 1.12 else v

        # Light and turn, collapsed into one weight per column — and then,
        # in the loop, *sheared*. This is the last of the vertical-streak
        # problems and the stubborn one: a mass here is a height field
        # y = top(x), so anything derived from the ridge is constant down a
        # column by construction, and slope shading therefore paints the
        # mountain in vertical bands however smooth the slope field is. No
        # amount of filtering fixes that, because the stripes are not noise.
        # What fixes it is reading the weight at the column the rock
        # *came from*: a point partway down a flank belongs to the crest
        # upslope of it, so the shading is sampled along the fall line, with
        # a slow 128 px wobble so the bands curve instead of ruling. The
        # light now runs down the flanks the way the 皴 does.
        cvl = [sl[i] * self.w_light + cu[i] * self.w_cur
               for i in range(n + 1)]

        # 卷云皴 — the curl has to be a function of *screen height*, never of
        # depth below the crest. Depth moves a pixel per pixel across a 45°
        # flank, so a warp keyed to it swings the comb sideways as fast as x
        # advances and the spacing aliases into a fine mesh — exactly the
        # failure the slope compensation below undoes for the lean. Keyed to
        # y, with only a heavily smoothed phase drifting from column to
        # column, the whole comb rolls together and the spacing is untouched.
        wA = wB = wC = wD = None
        s1 = c1 = s2 = c2 = None
        boff = None
        if mode == 3:
            # 折带 strata are horizontal *on the paper*. Index the bands by
            # depth below the crest instead and they run parallel to the
            # silhouette — chevrons over a chevron peak, rings over a dome:
            # a contour map, not a rock. Index them by screen y, with a
            # smoothed per-column offset so the strata lift and dip across
            # the mass rather than ruling it.
            row1 = ((self.x0 * 11 + 7) & gm) * gn
            # Enough swing that the strata lift and dip across the mass. Held
            # to a fraction of the spacing they stay level, every band runs
            # the full width at the same height, and the rock reads as a
            # layer cake instead of as tilted and broken ground.
            boff = [gg[row1 + ((i >> 3) & gm)] * self.per * 3.0
                    for i in range(n + 1)]
            for _ in range(18):
                boff = ([boff[0]]
                        + [(boff[i - 1] + 2.0 * boff[i] + boff[i + 1]) * 0.25
                           for i in range(1, n)] + [boff[n]])
        if mode == 4:
            s1 = [math.sin(y * 0.080) for y in range(H + 1)]
            c1 = [math.cos(y * 0.080) for y in range(H + 1)]
            s2 = [math.sin(y * 0.185) for y in range(H + 1)]
            c2 = [math.cos(y * 0.185) for y in range(H + 1)]
            row1 = ((self.x0 * 11 + 7) & gm) * gn
            cph = [gg[row1 + ((i >> 4) & gm)] * 6.283 for i in range(n + 1)]
            for _ in range(40):
                cph = ([cph[0]]
                       + [(cph[i - 1] + 2.0 * cph[i] + cph[i + 1]) * 0.25
                          for i in range(1, n)] + [cph[n]])
            # peak swing 1.42 * wamp, i.e. ~0.37 of a stroke spacing: enough
            # to see every fibre roll, little enough that neighbours never
            # cross and interleave back into a straight screen
            wamp = self.per * 0.26
            wA = [wamp * math.cos(p) for p in cph]
            wB = [wamp * math.sin(p) for p in cph]
            wC = [wamp * 0.42 * math.cos(p * 2.1) for p in cph]
            wD = [wamp * 0.42 * math.sin(p * 2.1) for p in cph]

        tone, a0, a1 = self.tone, self.a0, self.a1
        w_light, w_cur = self.w_light, self.w_cur
        w_line = self.w_line
        fade = self.fade

        for i in range(n + 1):
            x = x0 + i
            if x < 0 or x >= W:
                continue
            ty = top[i]
            by = base if scalar_base else base[i]
            if by - ty < 0.75:
                continue
            # slope of the 皴: strokes run down the fall line, so the
            # across-stroke coordinate is x shifted by the slope
            sxk = sk[i]
            tys = tsm[i]
            # The comb is spaced in x, but the lean term `sxk * (y - tys)`
            # also moves with x, which squeezes the screen period by
            # (1 + sxk^2/2) — on a steep flank that is better than 2x, and a
            # 3 px stroke with a 6 px gap collapses into a 1 px mesh. One
            # division per column puts the spacing back where it was asked
            # for, whatever the slope.
            ip = inv_per / (1.0 + 0.5 * sxk * sxk)
            if mode == 4:
                wa, wb, wc, wd = wA[i], wB[i], wC[i], wD[i]
            elif mode == 3:
                yoff = boff[i]
            lnm = ln[i]
            txi = tx[i]
            lwi = lwd[i]
            inv_lwi = 1.0 / lwi if lwi > 0.35 else 0.0
            colh = by - ty
            ftd = foot_d if foot_d < colh * 0.80 else colh * 0.80
            inv_ftd = 1.0 / ftd
            # a tall mass fades over `fade` pixels; a short skirt column
            # fades over most of itself, so nothing ends in a straight edge
            fdi = fade if fade < colh * 0.80 else colh * 0.80
            inv_fade = 1.0 / fdi if fdi > 0.0 else 0.0
            iy = int(ty)
            if iy < 0:
                iy = 0
            iyb = int(by) + 1
            if iyb > H:
                iyb = H
            for y in range(iy, iyb):
                dy = y - ty
                cov = 1.0
                if dy < 0.0:
                    cov = dy + 1.0
                    if cov <= 0.0:
                        continue
                r = by - y
                if r < 1.0:
                    cov *= r
                    if cov <= 0.0:
                        continue
                if fdi > 0.0 and r < fdi:
                    cov *= r * inv_fade
                dyp = y - tys          # depth below the *smoothed* crest
                idy = int(dy)
                if idy < 0:
                    idy = 0
                elif idy > hmax:
                    idy = hmax
                # The wash is keyed to the smoothed crest, not to this
                # column's own crest — and this is the whole reason large
                # masses used to come out *combed*. The 染 band runs a good
                # quarter of the height; measured from `top[i]`, a notch
                # 30 px deep carries its band 30 px down with it, so every
                # notch in the ridge drags a full-height stripe of lighter
                # ink behind it. The band belongs to the ridge as a form, so
                # it hangs off the form's crest and crosses the notches
                # level. (In a notch `dyp` starts negative and clamps to 0,
                # which leaves the notch dark — which is correct: a notch is
                # recessed.)
                idp = int(dyp)
                if idp < 0:
                    idp = 0
                elif idp > hmax:
                    idp = hmax
                # the fall-line sample: upslope by the lean, wobbled. The
                # travel is capped, and the cap matters: let it run the full
                # depth and a steep flank sweeps the sample right across the
                # mass, so the bands close into concentric loops and the rock
                # comes out as marble or melted wax. Thirty pixels is enough
                # to break the vertical and little enough that the shading
                # still describes the form it sits on.
                off = sxk * dyp
                if off > 30.0:
                    off = 30.0
                elif off < -30.0:
                    off = -30.0
                j = i - int(off
                            + (bg[((y & bm) * bn) + (x & bm)] - 0.5) * 22.0)
                if j < 0:
                    j = 0
                elif j > n:
                    j = n
                k = tone + cvl[j] * dec[idp] + prw[idp]
                if inv_lwi and dy < lwi:        # 勾 — the crest line
                    z = 1.0 - dy * inv_lwi
                    if z > 1.0:
                        z = 1.0
                    k += w_line * z * z * lnm
                if r < ftd:
                    k -= w_foot * (1.0 - r * inv_ftd)

                # ---- 皴 ------------------------------------------------
                if mode >= 0:
                    amp = amt[idp] * txi
                    # 擦 and 积墨 — two frequencies of broad tonal patch, 32 px
                    # and 128 px, both off the broad tile so neither lands on
                    # a block boundary. Without them a mass is a smooth
                    # gradient under a stroke pattern, which reads as
                    # airbrush; a painted rock has whole facets left pale and
                    # whole facets built up dark.
                    gc = bg[(((y << 2) & bm) * bn) + ((x << 2) & bm)]
                    k += (scrub * (gc - 0.45)
                          + blot * (bg[((y & bm) * bn) + (x & bm)] - 0.48))
                    # 留 — the unworked patch. 64 px to a blob, so a flank
                    # carries worked rock next to bare paper in areas, not in
                    # the vertical stripes a per-column gate would give.
                    e = (bg[(((y << 1) & bm) * bn)
                            + ((x << 1) & bm)] - 0.26) * 2.3
                    if e <= 0.0:
                        amp = 0.0
                    else:
                        amp *= 1.3 if e > 1.3 else e
                    if mode == 0:                   # 披麻 — long hemp fibres
                        # The phase comb says *where* the strokes are; the
                        # gate says which of them the brush actually laid
                        # down, and how far. That gate has to be read in the
                        # stroke's own frame — one value per stroke index,
                        # changing slowly down its length. Read it off the
                        # screen instead and it flickers on and off every few
                        # pixels as the slanted stroke crosses the noise, and
                        # every fibre breaks into a column of 3 px dashes.
                        gw = gg[(((y >> 5) & gm) * gn) + ((x >> 3) & gm)]
                        q = (x - sxk * dyp * 0.5) * ip + gw * 1.3
                        iq = int(q)
                        fr = q - iq
                        if fr < 0.0:
                            fr += 1.0
                            iq -= 1
                        if fr < 0.34:
                            gl = gg[(((idy >> 2) & gm) * gn)
                                    + ((iq * 5) & gm)]
                            if gl > 0.33:
                                k += (amp * (0.34 - fr) * 3.2
                                      * (gl - 0.33) * 2.0 * (0.5 + gc))
                    elif mode == 1:                 # 斧劈 — axe-cut slabs
                        # 擦 picks which facets are worked at all; the comb
                        # leans down the fall line, because letting depth
                        # drive the phase draws contour lines on a survey
                        # map; and the gate cuts the slabs to length, since
                        # an un-gated comb is a hatching screen and not a
                        # brush that was pressed down and lifted.
                        if gc > 0.38:
                            q = (x - dyp * sxk * 0.45) * ip + gc * 0.35
                            iq = int(q)
                            fr = q - iq
                            if fr < 0.0:
                                fr += 1.0
                                iq -= 1
                            if fr < 0.50:
                                # 斧劈 is a short, broad slab — a few pixels
                                # long. Consecutive rows of the grain tile are
                                # correlated over its 8 px cell, so gating on
                                # `idy >> 1` alone holds one value for ~30 px
                                # of depth and every axe-cut came out as a
                                # long vertical drip. The stride has to clear
                                # the cell, not just the row: at 5 rows per
                                # 2 px the gate was still landing inside the
                                # same 8-row cell and the drips survived.
                                gx = gg[((((idy >> 1) * 11) & gm) * gn)
                                        + ((iq * 7) & gm)]
                                if gx > 0.34:
                                    # a flat-ended slab, not a tapered hair:
                                    # the brush is laid on its side, pressed,
                                    # and dragged off
                                    e = fr * 8.0
                                    if e > 1.0:
                                        e = 1.0
                                    k += (amp * 1.7 * e * (gc - 0.38) * 2.2
                                          * (gx - 0.34) * 1.6)
                    elif mode == 2:                 # 雨点 — raindrop stipple
                        gx = gg[(((y << 2) & gm) * gn) + ((x << 2) & gm)]
                        if gx > 0.56:
                            k += amp * (gx - 0.56) * 2.3 * (0.45 + gc)
                    elif mode == 3:                 # 折带 — folded bands
                        # 倪瓒's strata run horizontally, so here the band
                        # index is the depth and the length runs along x.
                        q = (y + yoff) * inv_per + gc * 0.5
                        iq = int(q)
                        fr = q - iq
                        if fr < 0.0:
                            fr += 1.0
                            iq -= 1
                        if fr < 0.26:
                            gl = gg[(((iq * 7) & gm) * gn) + ((x >> 3) & gm)]
                            if gl > 0.34:
                                k += (amp * (0.26 - fr) * 4.2
                                      * (gl - 0.34) * 2.2)
                        elif fr > 0.86:
                            gx = gg[((y & gm) * gn) + (x & gm)]
                            if gx > 0.62:           # the vertical tick that
                                k += amp * 0.7      # ends each band
                    else:                           # 卷云 — rolling cloud
                        gw = gg[(((y >> 5) & gm) * gn) + ((x >> 3) & gm)]
                        q = ((x - sxk * dyp * 0.5
                              + s1[y] * wa + c1[y] * wb
                              + s2[y] * wc + c2[y] * wd) * ip
                             + gw * 0.7)
                        iq = int(q)
                        fr = q - iq
                        if fr < 0.0:
                            fr += 1.0
                            iq -= 1
                        if fr < 0.36:
                            gl = gg[(((idy >> 2) & gm) * gn)
                                    + ((iq * 5) & gm)]
                            if gl > 0.30:
                                k += (amp * (0.36 - fr) * 2.9
                                      * (gl - 0.30) * 1.8 * (0.45 + gc))
                else:
                    # 远山 carry no 皴 at all, but a mass of perfectly even
                    # tone reads as cut paper rather than as a hill seen
                    # through air. index.html gives each facet of its
                    # distant ranges its own grey; two broad bands of noise
                    # do the same without a seam between facets.
                    k += (0.050 * (bg[(((y << 1) & bm) * bn)
                                      + ((x << 1) & bm)] - 0.48)
                          + 0.075 * (bg[((y & bm) * bn) + (x & bm)] - 0.50))

                if k <= 0.004:
                    continue
                if k > 1.0:
                    k = 1.0
                # the opacity floor fades in over the first sliver of density,
                # so the edge of a wash is a soft boundary and not a step
                a = ((a0 * (k * 9.0) if k < 0.111 else a0) + a1 * k) * cov
                if a <= 0.004:
                    continue
                if a > 1.0:
                    a = 1.0
                cr, cg, cb = ramp[int(k * NS)]
                i3 = (y * W + x) * 3
                o = buf[i3]
                buf[i3] = int(o + (cr - o) * a)
                i3 += 1
                o = buf[i3]
                buf[i3] = int(o + (cg - o) * a)
                i3 += 1
                o = buf[i3]
                buf[i3] = int(o + (cb - o) * a)

    # -- 皴 — the strokes that are actually drawn, over the field ---------

    def cun_strokes(self, cv, rng, k, scale=1.0, ink=None, alpha=1.0):
        """The 皴 the viewer reads as brushwork, laid on top of the field.

        The per-pixel texture in `render` is right about *grain*: it gives the
        rock its fibre, its fold, its worked and unworked patches, and it does
        so at a scale of a few pixels, which is the scale of the hairs of a
        brush. But a painter does not work at the scale of a hair. The marks
        the eye actually reads are strokes ten to forty pixels long, laid one
        at a time down the fall line, each with a head where the brush landed
        and a tail where it was dragged off — and no amount of per-pixel noise
        adds up to one of those, because a stroke is a decision and noise is
        not. This is the lesson of index.html: its mountains have no shading
        model at all, only a few hundred explicit strokes, and they read as
        painted where a smooth field reads as rendered.

        So: pick a point on the rock, find the fall line, and drag. The point
        has to land where the field left the rock worked — the same broad
        tile, read the same way — or the strokes and the grain will describe
        two different mountains.
        """
        mode = self.cun
        if mode < 0 or k <= 0:
            return
        top, x0 = self.top, self.x0
        n = len(top) - 1
        if n < 12:
            return
        base = self.base
        scalar = not isinstance(base, list)
        gr = self.grain
        col = ink if ink is not None else self.deep
        # The wash does not reach the nominal base: `fade` dissolves the last
        # stretch of every column into the paper. A stroke measured against
        # the nominal base therefore gets drawn on bare paper below the rock
        # it belongs to — dashes of dry ink floating in the water, which is
        # exactly what the first version put there.
        fd = self.fade * 0.75
        for _ in range(k):
            i = rng.randrange(2, n - 1)
            b = (base if scalar else base[i]) - fd
            colh = b - top[i]
            if colh < 11.0:
                continue
            # never on the ridge line itself: 勾 is clean ink and the 皴 runs
            # up against it from below
            sy = top[i] + colh * rng.uniform(0.05, 0.86)
            sx = float(x0 + i)
            if gr.broad(int(sx) << 1, int(sy) << 1) < 0.30:
                continue                      # an unworked patch — leave it
            # the fall line, over a window wide enough not to follow the
            # crumple of the crest: a stroke describes the flank, not the
            # silhouette
            g = ((top[min(n, i + 9)] - top[max(0, i - 9)]) / 18.0)
            if g > 1.7:
                g = 1.7
            elif g < -1.7:
                g = -1.7
            # The stroke's length comes from the *brush*, not from the
            # mountain: `per` is the spacing the 皴 was asked for, so a slab
            # is a couple of spacings long and a hemp fibre is half a dozen.
            # Scaled off the column height instead — the first thing I tried —
            # a 300 px mass gets 60 px strokes and it rains meteors.
            per = self.per
            if mode == 3:                     # 折带 — the strata lie flat
                dx, dy = rng.choice((1.0, -1.0)), rng.uniform(-0.10, 0.10)
                L = per * rng.uniform(1.6, 4.6)
                w0, w1 = 1.4 * scale, 0.4 * scale
                cur = rng.uniform(-0.10, 0.10)
            else:
                inv = 1.0 / math.sqrt(1.0 + g * g)
                dx, dy = g * inv, inv
                if mode == 1:                 # 斧劈 — short broad slabs
                    L = per * rng.uniform(0.9, 2.1)
                    w0, w1 = 2.1 * scale, 0.5 * scale
                    cur = rng.uniform(-0.06, 0.06)
                elif mode == 2:               # 雨点 — barely a drag at all
                    L = 2.0 + 3.0 * rng.random()
                    w0, w1 = 1.8 * scale, 1.0 * scale
                    cur = 0.0
                elif mode == 4:               # 卷云 — the stroke itself curls
                    # 郭熙's clouds of rock are a hand's width of curl, not a
                    # blackboard circle: at four spacings long and a 0.6 bow
                    # these came out as chalk loops laid over the mountain.
                    L = per * rng.uniform(0.7, 2.0)
                    w0, w1 = 1.3 * scale, 0.35 * scale
                    cur = rng.uniform(0.20, 0.52) * rng.choice((1.0, -1.0))
                else:                         # 披麻 — long hemp fibres
                    L = per * rng.uniform(2.4, 7.0)
                    w0, w1 = 1.2 * scale, 0.3 * scale
                    cur = rng.uniform(-0.16, 0.16)
            if L > colh * 0.75:
                L = colh * 0.75
            # 皴 comes in bundles. One stroke on its own is a scratch; five
            # or six laid side by side, staggered along their own length, are
            # a facet of rock — and a painter works facet by facet, which is
            # why a bundle shares one direction and one bow and only the
            # offsets differ.
            px, py = -dy, dx                  # across-stroke unit vector
            gap = per * rng.uniform(0.55, 1.15)
            for q in range(rng.randint(2, 6)):
                o = (q - 0.5) * gap + rng.uniform(-0.3, 0.3) * gap
                ax = sx + px * o + dx * rng.uniform(-0.35, 0.35) * L
                ay = sy + py * o + dy * rng.uniform(-0.35, 0.35) * L
                ll = L * rng.uniform(0.6, 1.15)
                # Walk the stroke out and stop at the first step that leaves
                # the rock, so nothing hangs over the silhouette.
                pts = []
                steps = 6
                for s in range(steps + 1):
                    t = s / float(steps)
                    bow = cur * ll * t * (1.0 - t) * 2.0
                    qx = ax + dx * ll * t + px * bow
                    qy = ay + dy * ll * t + py * bow
                    ii = int(qx) - x0
                    if ii < 0 or ii > n:
                        break
                    bb = (base if scalar else base[ii]) - fd
                    if qy < top[ii] + 0.8 or qy > bb - 0.8:
                        break
                    pts.append((qx, qy))
                if len(pts) < 2:
                    continue
                brush_path(cv, pts, w0 * rng.uniform(0.7, 1.25), w1, col,
                           alpha * rng.uniform(0.07, 0.24))

    # -- 点 — moss dots along the crest ----------------------------------

    def moss(self, cv, rng, k, ink, scale=1.0):
        """点 — a 点 is not a dot. The brush lands, presses and lifts, so the
        mark has a head, a tail and a direction; a disc has none of the
        three. Drawn as little tapered dabs leaning off the vertical, the
        same number of marks stops looking like spattered ink and starts
        looking like scrub growing out of the rock."""
        n = len(self.top) - 1
        for _ in range(k):
            i = rng.randrange(1, n) if n > 2 else 1
            x = self.x0 + i + rng.uniform(-1.5, 1.5)
            y = self.top[i] + rng.uniform(0.0, 4.0) * scale
            lean = rng.uniform(-0.55, 0.55)          # one cluster, one lean
            for _ in range(rng.randint(2, 5)):
                px = x + rng.uniform(-2.4, 2.4) * scale
                py = y + rng.uniform(-1.2, 3.2) * scale
                a = -1.571 + lean + rng.uniform(-0.3, 0.3)
                L = rng.uniform(1.6, 3.6) * scale
                brush_seg(cv, px, py,
                          px + math.cos(a) * L, py + math.sin(a) * L,
                          rng.uniform(0.8, 1.7) * scale, 0.3 * scale,
                          ink, rng.uniform(0.35, 0.8))


# --------------------------------------------------------------------------
# 留白 — mist and water surfaces are *not painted*. They are the paper.
#
# A painter leaves them bare; a generator that draws mountains first has to
# take the paper back, so a snapshot of the bare ground is kept and blended
# in again. Same result, opposite direction.
# --------------------------------------------------------------------------

def foot_lobes(cv, rng, under, R, ink, scale=1.0, strength=0.95, k=None):
    """山脚 — the skirt of a mass, taken back out of what lay behind it.

    A height field has to stop somewhere, and the place it stops is the one
    place the whole construction shows: the wash runs down to the base and
    ends, so every mass sits on a line. No painter draws that line. The foot
    of a mountain is a row of overlapping rounded tongues, each one a lobe of
    ground pushing forward, and the lowest 皴 vanishes behind them.

    index.html lays its feet in as white polygons over the finished texture.
    Here they are the layer *underneath* coming back — the same move as
    `mist`, but against a snapshot taken immediately before this mass was
    rendered rather than against the bare paper. That distinction matters:
    the foot of a shoulder has its parent mountain behind it, not paper, and
    erasing to paper would punch a hole through the middle of the mountain.
    """
    W, H, buf = cv.w, cv.h, cv.buf
    top, x0 = R.top, R.x0
    n = len(top) - 1
    if n < 8:
        return
    base = R.base
    scalar = not isinstance(base, list)
    fd = R.fade * 0.75                   # the wash stops short of the base
    if k is None:
        k = max(2, int(n / (34.0 * scale)))
    hi = 0.0
    for i in range(n + 1):
        b = (base if scalar else base[i]) - fd
        if b - top[i] > hi:
            hi = b - top[i]
    if hi < 10.0:
        return
    for _ in range(k):
        cx = rng.uniform(-0.04, 1.04) * n
        w = n * rng.uniform(0.045, 0.17) + 6.0
        # the lobes are shallow next to the mass they belong to: a toe, not
        # a bite. Taller than about a fifth of the mass and the foot starts
        # eating the structure it is supposed to be hiding.
        h = min(hi * rng.uniform(0.07, 0.20), w * rng.uniform(0.35, 0.8))
        if h < 2.5:
            continue
        inv_w = 1.0 / w
        edge = []
        i0 = int(cx - w)
        i1 = int(cx + w) + 1
        for i in range(max(0, i0), min(n, i1) + 1):
            x = x0 + i
            if x < 0 or x >= W:
                continue
            u = (i - cx) * inv_w
            if u < -1.0 or u > 1.0:
                continue
            # a flat-topped lobe with shoulders that fall away fast — the
            # profile of a tongue of silt, not of a circle
            d = h * (1.0 - u * u) ** 0.62
            b = (base if scalar else base[i]) - fd
            # Where the mass is too shallow to carry the tongue, *shorten*
            # the tongue. Skipping the column instead ends the lobe on a
            # vertical cut, and a lobe clipped at both ends is a hard-edged
            # white slot sitting in the middle of the rock — which is what
            # the first version put into every terraced shoulder.
            dm = (b - top[i]) * 0.55
            if d > dm:
                d = dm
            if d < 1.0:
                continue
            yt = b - d
            edge.append((x, yt))
            iy = int(yt)
            iyb = min(H, int(b) + 2)
            if iy < 0:
                iy = 0
            for y in range(iy, iyb):
                a = strength
                f = y - yt                       # soften the upper boundary
                if f < 1.6:
                    if f < 0.0:
                        continue
                    a *= f * 0.625
                j = (y * W + x) * 3
                o = buf[j]
                buf[j] = int(o + (under[j] - o) * a)
                j += 1
                o = buf[j]
                buf[j] = int(o + (under[j] - o) * a)
                j += 1
                o = buf[j]
                buf[j] = int(o + (under[j] - o) * a)
        # the tongue is then *drawn*, faintly: a dry line along its back.
        # Without it the lobes are only an absence and the foot looks
        # bleached rather than described.
        if len(edge) > 3 and rng.random() < 0.85:
            a = rng.randrange(0, max(1, len(edge) - 3))
            seg = edge[a:a + rng.randint(4, max(5, len(edge) - a))]
            if len(seg) > 2:
                brush_path(cv, seg, rng.uniform(0.6, 1.1), 0.5, ink,
                           rng.uniform(0.09, 0.22))


def mist(cv, rng, paper, y, h, strength=0.9, cells=7):
    """云气 — a lenticular band of cloud that eats the feet of a mountain."""
    W, H, buf = cv.w, cv.h, cv.buf
    n = Noise(rng, cells, 3)
    y0 = max(0, int(y - h))
    y1 = min(H, int(y + h))
    if y1 <= y0:
        return
    inv = 1.0 / h
    for py in range(y0, y1):
        t = (py - y) * inv
        v = 1.0 - t * t
        if v <= 0.0:
            continue
        fy = (py - y0) / float(y1 - y0)
        row = py * W
        for px in range(W):
            a = strength * v * (0.12 + 1.15 * n.at(px / float(W), fy))
            if a <= 0.01:
                continue
            if a > 1.0:
                a = 1.0
            i = (row + px) * 3
            o = buf[i]
            buf[i] = int(o + (paper[i] - o) * a)
            i += 1
            o = buf[i]
            buf[i] = int(o + (paper[i] - o) * a)
            i += 1
            o = buf[i]
            buf[i] = int(o + (paper[i] - o) * a)


def waterfall(cv, rng, paper, x, y0, y1, w, ink):
    """瀑布 — carved out of the rock as 留白, then given a few 水纹 lines.

    A fall that runs the whole drop in one unbroken white band is a scratch
    on the paper, however carefully its edges are feathered: nothing in the
    picture says *water* rather than *gap*. What says it is the breaks. Real
    falls come down in two or three drops over ledges, each drop stepped
    sideways from the last and each landing in a cloud of spray, and the
    spray is the part the eye reads as moving water. So the fall is built as
    a chain of drops with a 水口 under every one of them, not as a slot.
    """
    W, H, buf = cv.w, cv.h, cv.buf
    if y1 <= y0:
        return
    wob = Noise(rng, 2, 9)
    dk = I("zhong")
    span = y1 - y0

    def spray(yy, ww, hh, a0):
        """水口 — paper blown back in a soft cloud where a drop lands."""
        for py in range(max(0, int(yy - hh)), min(H, int(yy + hh)) + 1):
            v = (py - yy) / hh
            row = py * W
            rr = ww * math.sqrt(1.0 - v * v) if v * v < 1.0 else 0.0
            if rr < 0.6:
                continue
            for px in range(max(0, int(x - rr)), min(W, int(x + rr) + 1)):
                d = (px - x) / rr
                a = a0 * (1.0 - d * d) * (1.0 - v * v) * (
                    0.45 + 0.55 * wob.at(0.5 + 0.5 * d,
                                         (py % 97) / 97.0))
                if a <= 0.0:
                    continue
                i = (row + px) * 3
                for _ in range(3):
                    o = buf[i]
                    buf[i] = int(o + (paper[i] - o) * a)
                    i += 1

    def drop(ya, yb, cxo, w0, w1):
        if yb - ya < 3.0:
            return
        sp = yb - ya
        # 夹 — the rock on either side, darkened first. Water carved out of a
        # flat slope is a white post leaning on the hill; what makes it read
        # as a fall is the cleft it comes down, so the walls go in before the
        # 留白 does. They are laid down by column, not stroked: a thick
        # low-alpha brush_path double-composites where consecutive capsules
        # overlap, and the fall came out flanked by two neat rows of dots.
        # The two walls are given different weights — a symmetrical cleft is
        # a machined slot.
        wl = rng.uniform(0.7, 1.25)
        wr = rng.uniform(0.7, 1.25)
        for py in range(max(0, int(ya)), min(H, int(yb)) + 1):
            t = (py - ya) / sp
            hw = w0 + (w1 - w0) * t
            cx = x + cxo + (wob.at(0.2, (py - y0) / span) - 0.5) * w * 0.9
            rad = hw * 1.6
            for side, sw in ((-1.0, wl), (1.0, wr)):
                wx = cx + side * (hw + rad * 0.55)
                for px in range(int(wx - rad), int(wx + rad) + 1):
                    d = (px + 0.5 - wx) / rad
                    # the cleft has to be *dark*. The fall is 留白, so it has
                    # no contrast of its own: all of it is borrowed from the
                    # rock beside it, and on a pale upper flank a faint wall
                    # leaves the fall a ghost.
                    a = ((1.0 - d * d) * 0.40 * sw
                         * (0.55 + 0.45 * wob.at(0.7, (py - y0) / span)))
                    if a > 0.0:
                        cv.blend(px, py, dk, a)
        for py in range(max(0, int(ya)), min(H, int(yb))):
            t = (py - ya) / sp
            hw = w0 + (w1 - w0) * t
            cx = x + cxo + (wob.at(0.2, (py - y0) / span) - 0.5) * w * 0.9
            # the head of the drop comes *out of* the rock, and the tail goes
            # *into* its own spray: with a flat profile both ends are ruled
            # cuts and the whole thing reads as a post
            head = t / 0.18 if t < 0.18 else 1.0
            if t > 0.86:
                head *= (1.0 - t) / 0.14
            a0 = max(0, int(cx - hw))
            a1 = min(W, int(cx + hw) + 1)
            row = py * W
            for px in range(a0, a1):
                d = abs(px - cx) / hw
                a = (1.0 - d * d * 0.55) * head
                if a <= 0.0:
                    continue
                i = (row + px) * 3
                o = buf[i]
                buf[i] = int(o + (paper[i] - o) * a)
                i += 1
                o = buf[i]
                buf[i] = int(o + (paper[i] - o) * a)
                i += 1
                o = buf[i]
                buf[i] = int(o + (paper[i] - o) * a)
        # 石梁 — the lip the water pours over. One dark stroke across the head
        # of the drop, and the fall stops being a slot in the hillside and
        # becomes water leaving an edge. It is also the only piece of real
        # contrast available up here, where the flank is at its palest.
        # Two strokes, one each side, angling in to the lip and stopping at
        # the water — a bar drawn straight across the white would read as a
        # dam holding the fall back rather than as the edge it leaves.
        lx = x + cxo + (wob.at(0.2, (ya - y0) / span) - 0.5) * w * 0.9
        for side in (-1.0, 1.0):
            brush_seg(cv,
                      lx + side * w0 * rng.uniform(2.6, 4.2),
                      ya + rng.uniform(-2.2, -0.4),
                      lx + side * w0 * rng.uniform(0.85, 1.05),
                      ya + rng.uniform(0.0, 1.4),
                      rng.uniform(1.0, 1.7), 0.4, dk, rng.uniform(0.34, 0.58))
        # 水 threads inside the drop — the fibre of the falling sheet
        for _ in range(rng.randint(2, 4)):
            ox = cxo + rng.uniform(-0.5, 0.5) * w
            step = max(4, int(sp / 7))
            pts = [(x + ox
                    + (wob.at(0.2, (py - y0) / span) - 0.5) * w * 0.8, py)
                   for py in range(int(ya + sp * 0.12), int(yb), step)]
            if len(pts) > 1:
                brush_path(cv, pts, 0.6, 0.9, ink, rng.uniform(0.10, 0.22))

    # one drop per ledge, each stepped sideways and each a little wider
    k = 1 if span < 70.0 else rng.randint(2, 3)
    cut = sorted(rng.uniform(0.22, 0.80) for _ in range(k - 1))
    ya, cxo = y0, 0.0
    for d in range(k):
        t1 = cut[d] if d < k - 1 else 1.0
        yb = y0 + span * t1
        w0 = w * (0.55 + 0.60 * (ya - y0) / span)
        w1 = w * (0.55 + 0.60 * (yb - y0) / span)
        drop(ya, yb, cxo, w0, w1)
        spray(yb, w1 * rng.uniform(2.2, 3.6), w1 * rng.uniform(1.1, 2.0),
              0.55 if d == k - 1 else rng.uniform(0.35, 0.60))
        # 水花 — a few flat strokes where the water hits the ledge
        for _ in range(rng.randint(2, 5)):
            yy = yb + rng.uniform(-1.5, 4.0)
            ww = w1 * rng.uniform(1.0, 2.4)
            ox = x + cxo + rng.uniform(-1.0, 1.0) * w
            brush_seg(cv, ox - ww, yy, ox + ww, yy + rng.uniform(-1.0, 1.0),
                      0.6, 0.6, ink, rng.uniform(0.10, 0.22))
        ya = yb + w1 * rng.uniform(0.6, 1.6)
        cxo += rng.uniform(-1.0, 1.0) * w * 1.8
        if ya >= y1:
            break


def water_lines(cv, rng, y0, y1, ink, dens=1.0):
    """水纹 — water is mostly 留白; the lines only exist to tell you the white
    is wet.

    They come in *clusters*. A painter lays three or four strokes in one
    place, lifts the brush, moves along the surface and lays another group —
    so the marks gather, and between the groups there is nothing at all.
    Scattered one at a time over the whole surface, as this did at first,
    the same number of lines reads as an evenly ruled grid; index.html gets
    the grouping the same way, by looping clusters and not lines.
    """
    W = cv.w
    if y1 <= y0 + 2:
        return
    clu = int((y1 - y0) * 0.030 * dens) + 2
    for _ in range(clu):
        t = rng.random() ** 0.7                   # denser toward the far shore
        cy = y0 + (y1 - y0) * t
        cx = rng.uniform(-0.06, 0.90) * W
        L0 = W * rng.uniform(0.10, 0.32)
        ph = rng.uniform(0, 6.28)
        yk = cy
        for _ in range(rng.randint(2, 5)):
            yk += rng.uniform(1.8, 6.0)
            if yk > y1 - 1.0:
                break
            x = cx + rng.uniform(-0.07, 0.07) * W
            L = L0 * rng.uniform(0.55, 1.35)
            amp = rng.uniform(0.5, 1.7)
            # the wave is not a sine: its own amplitude swells and dies along
            # the stroke, which is what keeps a 3 px ripple from looking
            # mechanical
            pts = [(x + L * j / 7.0,
                    yk + math.sin(ph + j * 0.85) * amp
                    * (0.35 + 0.65 * abs(math.sin(ph * 2.0 + j * 0.45))))
                   for j in range(8)]
            brush_path(cv, pts, rng.uniform(0.5, 0.9), rng.uniform(0.4, 0.8),
                       ink, rng.uniform(0.07, 0.19) * (1.15 - t * 0.5))


def shore_rocks(cv, rng, y0, y1, ink, scale=1.0, k=3):
    """汀渚 — small rocks standing in the shallows, with reeds among them.

    Open water here is 留白 and a few ripples, and in 深远 or 平远 — where the
    water band can be a third of the sheet — that leaves a large area with
    nothing in it at all. The traditional filling is not more ripples: it is
    汀渚, a handful of rocks breaking the surface near the banks with 芦苇
    behind them. They cost almost nothing to draw and they give the water a
    near edge and a far edge, which is what makes it read as a surface
    receding rather than as blank paper.
    """
    if y1 <= y0 + 4:
        return
    for _ in range(k):
        t = rng.random() ** 0.8
        cy = y0 + (y1 - y0) * t
        cx = rng.uniform(0.04, 0.96) * cv.w
        # Perspective: a rock near the bottom of the water band is nearer.
        # Sized off the sheet, not off `scale` — at a few pixels across a
        # rock is a speck of dirt on the paper, not a rock.
        s = cv.h * (0.006 + 0.012 * t) * rng.uniform(0.8, 1.25)
        side = 1.0 if rng.random() < 0.5 else -1.0
        if rng.random() < 0.55:                     # 芦苇 — a tuft of reeds
            # Reeds stand *beside* a rock, on the waterline, not out of the
            # top of it, and they go down before the rock so the rock covers
            # their feet. A reed is also a thin arc that bends to one side and
            # tapers out, and the whole tuft bends the same way. Drawn as
            # straight dark segments from the rock's centre at two or three
            # times its height, they came out as a set of antennae and the
            # clump read as an insect. Short, curved, leaning together, pale.
            rx = cx + side * s * rng.uniform(1.6, 3.0)
            hh = min(s * rng.uniform(1.0, 1.8), cv.h * 0.035)
            lean = rng.uniform(-0.5, 0.5)
            for _ in range(rng.randint(3, 6)):
                L = hh * rng.uniform(0.6, 1.1)
                sw = (lean + rng.uniform(-0.22, 0.22)) * L
                x1 = rx + rng.uniform(-0.8, 0.8) * s
                y1 = cy + rng.uniform(-0.2, 0.4) * s
                st = bezier((x1, y1), (x1 + sw * 0.18, y1 - L * 0.45),
                            (x1 + sw * 0.55, y1 - L * 0.82),
                            (x1 + sw, y1 - L), 7)
                # One pass: a reed is thinner than the spacing of its own
                # vertices, so segment-by-segment the overlapping end caps
                # beaded it into a dashed line (see `_taper`).
                _taper(cv, st, max(0.5, s * 0.07), 0.25, ink,
                       rng.uniform(0.16, 0.38))
        # One rock, or one and a smaller one set back on the other side. Three
        # of them at random offsets of up to 2.6s all overlapped, and the
        # stack of discs, contours and ticks fused into one black lump with
        # legs: at this size a 汀渚 is two or three strokes, not a portrait.
        group = ((0.0, 1.0, 1.0),)
        if rng.random() < 0.55:
            group += ((-side * rng.uniform(1.4, 2.4), rng.uniform(0.45, 0.70),
                       0.62),)
        for dx, fs, dep in group:
            ox = cx + dx * s
            oy = cy + rng.uniform(-0.3, 0.3) * s
            sq = s * fs * rng.uniform(0.7, 1.0)
            # 石 — the top line, then the body filled from that line down to
            # the water. Three things had to be got right here, and each one
            # was a different wrong rock:
            #
            #   — the top sits *off centre*, with a short steep side and a
            #     long one. Built symmetrically it is an even little arch, and
            #     a bank of them is a row of tents pitched on the river.
            #   — it is *low*. A rock seen across water is much wider than it
            #     is tall; as high as it was half wide, each one came out a
            #     sharp little peak.
            #   — and it is *solid*. The wash has to go in under the top line
            #     and reach the waterline. Dropped in as an ellipse floating
            #     inside the contour it leaves a pale gap under the top edge,
            #     and the rock reads as an arch with a line beneath it — the
            #     whole shoal came out as a flotilla of canoes.
            ap = rng.uniform(0.28, 0.72)
            xa = ox - sq * rng.uniform(1.15, 1.6)
            xb = ox + sq * rng.uniform(1.15, 1.6)
            hr = sq * rng.uniform(0.52, 0.80)
            px_ = xa + (xb - xa) * ap
            pts = [(xa, oy),
                   (xa + (px_ - xa) * 0.55, oy - hr * rng.uniform(0.45, 0.72)),
                   (px_, oy - hr),
                   (px_ + (xb - px_) * 0.45, oy - hr * rng.uniform(0.52, 0.80)),
                   (xb, oy)]
            # 染 — a column scan under the profile, darkest just below the top
            # line and fading into the water, which is where the rock stops
            # being rock and starts being reflection.
            al = rng.uniform(0.20, 0.34) * dep
            fx0, fy0, fx1, fy1 = cv.clip(xa, oy - hr - 2.0, xb, oy + 2.0)
            for px in range(fx0, fx1):
                fx = px + 0.5
                ty = None
                for q in range(len(pts) - 1):
                    gx, gy = pts[q]
                    hx, hy = pts[q + 1]
                    if hx > gx and gx <= fx <= hx:
                        ty = gy + (hy - gy) * (fx - gx) / (hx - gx)
                        break
                if ty is None:
                    continue
                span = oy - ty
                if span < 0.5:
                    continue
                for py in range(fy0, fy1):
                    fy = py + 0.5
                    cov = fy - ty + 0.5
                    if cov <= 0.0:
                        continue
                    if cov > 1.0:
                        cov = 1.0
                    if fy > oy:
                        cov *= oy + 1.0 - fy
                        if cov <= 0.0:
                            continue
                    cv.blend(px, py, ink,
                             cov * al * (1.0 - 0.58 * (fy - ty) / span))
            # One pass, near-even weight. Tapered hard from end to end the
            # contour grew a long thin prow and the rock turned into a canoe.
            lw0 = max(0.6, sq * 0.19)
            _taper(cv, pts, lw0, lw0 * 0.8, ink, rng.uniform(0.38, 0.68) * dep)
            brush_seg(cv, pts[0][0], oy + rng.uniform(-0.3, 0.3),
                      pts[-1][0], oy + rng.uniform(-0.3, 0.3),
                      max(0.5, sq * 0.14), 0.5, ink,
                      rng.uniform(0.22, 0.44) * dep)
            for _ in range(rng.randint(1, 2)):      # 皴 — a tick or two
                px = ox + rng.uniform(-0.8, 0.8) * sq
                py = oy - rng.uniform(0.1, 0.6) * sq
                brush_seg(cv, px, py, px + rng.uniform(-0.25, 0.25) * sq,
                          py + rng.uniform(0.25, 0.6) * sq,
                          max(0.5, sq * 0.13), 0.4, ink,
                          rng.uniform(0.16, 0.34) * dep)
            # the water running off each side, and the rock is planted. Level,
            # and short: let these slant and lengthen and they stop being
            # ripples and become the legs the clump was standing on.
            for sd in (-1.0, 1.0):
                ry = oy + rng.uniform(-0.1, 0.3)
                brush_seg(cv, ox + sd * sq * 1.1, ry,
                          ox + sd * sq * rng.uniform(1.6, 2.4), ry,
                          0.6, 0.5, ink, rng.uniform(0.06, 0.13))


# --------------------------------------------------------------------------
# 树 — trees, drawn the way a 树谱 teaches them: 干 → 枝 → 叶
# --------------------------------------------------------------------------

def _twig(cv, rng, x, y, a, L, w, ink, alpha, depth, tips, claw=0.0):
    """One branch and its children. Branches leave the parent in threes and
    never in opposite pairs — that one rule is most of what makes a painted
    tree look like a tree."""
    cl = claw * rng.uniform(0.5, 1.5)
    p0 = (x, y)
    p3 = (x + math.cos(a) * L, y + math.sin(a) * L - cl * L * 0.3)
    b = rng.uniform(-0.3, 0.3)
    p1 = (x + math.cos(a + b) * L * 0.4, y + math.sin(a + b) * L * 0.4)
    p2 = (x + math.cos(a - b * 0.6) * L * 0.75,
          y + math.sin(a - b * 0.6) * L * 0.75)
    pts = bezier(p0, p1, p2, p3, 9)
    if depth and rng.random() < 0.7:
        pts, _ = knuckle(rng, pts, k=1, amp=0.30, start=0.3)
    # Wide enough for the seams to show (see `_taper`), lay it in one pass.
    if w >= 2.0:
        _taper(cv, pts, w, w * 0.45, ink, alpha)
    else:
        brush_path(cv, pts, w, w * 0.45, ink, alpha)
    if depth <= 0 or L < 5.0:
        tips.append((pts[-1][0], pts[-1][1], a, L))
        return
    nb = rng.randint(2, 3)
    for j in range(nb):
        t = rng.uniform(0.45, 0.95)
        i = min(len(pts) - 1, int(t * (len(pts) - 1)))
        da = rng.uniform(0.35, 0.95) * (1 if (j % 2) == (depth % 2) else -1)
        _twig(cv, rng, pts[i][0], pts[i][1], a + da,
              L * rng.uniform(0.44, 0.72), max(0.55, w * 0.55), ink,
              alpha, depth - 1, tips, claw)
    tips.append((pts[-1][0], pts[-1][1], a, L))


def _taper(cv, pts, w0, w1, color, alpha):
    """A tapered band laid down in a single pass of blending.

    `brush_path` draws its segments one at a time, and `brush_seg` clamps the
    projection to its own segment, so each one carries a round cap: two
    consecutive segments overlap over a disc of the brush radius centred on
    the vertex they share. That overlap gets blended twice. On a thin line
    nobody notices, but on a *wide* path — a trunk — the double-blended discs
    are as tall as the path is broad and they come at the vertex spacing, so
    the trunk is banded dark-pale-dark down its whole length. The bark trunks
    were coming out as barber poles, and no amount of tuning the alpha moves
    it, because the banding is in the compositing and not in the alpha.

    Here the coverage is taken as the maximum over every segment and written
    once per pixel, which is what a brush actually does: one wet stroke, one
    layer of ink, no seams.
    """
    n = len(pts) - 1
    if n < 1:
        return
    wm = max(w0, w1)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bx0, by0, bx1, by1 = cv.clip(min(xs) - wm - 2.0, min(ys) - wm - 2.0,
                                 max(xs) + wm + 2.0, max(ys) + wm + 2.0)
    segs = []
    for i in range(n):
        ax, ay = pts[i]
        dx, dy = pts[i + 1][0] - ax, pts[i + 1][1] - ay
        q = dx * dx + dy * dy
        if q < 1e-9:
            continue
        segs.append((ax, ay, dx, dy, 1.0 / q,
                     w0 + (w1 - w0) * (i / float(n)),
                     w0 + (w1 - w0) * ((i + 1.0) / n)))
    if not segs:
        return
    for py in range(by0, by1):
        fy = py + 0.5
        for px in range(bx0, bx1):
            fx = px + 0.5
            cov = 0.0
            for ax, ay, dx, dy, inv, wa, wb in segs:
                t = ((fx - ax) * dx + (fy - ay) * dy) * inv
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                ex = fx - (ax + dx * t)
                ey = fy - (ay + dy * t)
                c = (wa + (wb - wa) * t) - math.hypot(ex, ey) + 0.5
                if c > cov:
                    cov = c
                    if cov >= 1.0:
                        cov = 1.0
                        break
            if cov > 0.0:
                cv.blend(px, py, color, cov * alpha)


def _bark(cv, rng, pts, w, ink, alpha):
    """树皮 — the trunk as two broken outlines instead of one filled band.

    A tapered capsule gives a trunk of perfectly even width and perfectly
    smooth edge, and on a near tree that is the loudest machine tell in the
    picture: the mountain beside it is all broken line, and the tree is a
    extruded tube. A brush draws the trunk as two contours, each of which
    swells, thins and *stops* — the bark shows through the gaps. This is
    index.html's `barkify`: offset the centre line both ways, displace the
    offsets with noise, then break them into random runs.
    """
    n = len(pts)
    if n < 3:
        return
    nm = []
    for i in range(n):
        a = pts[i - 1] if i else pts[0]
        b = pts[i + 1] if i < n - 1 else pts[n - 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        nm.append((-dy / d, dx / d))
    # The body still goes in, and it must stay close to full strength. Drawn
    # pale on the theory that the contours carry the drawing, the trunk comes
    # out a hollow tube with the paper weave showing down the middle — two
    # rails and a ladder of rungs where the contour runs break.
    _taper(cv, pts, w * 0.80, w * 0.32, ink, alpha * 0.80)
    wob = Noise(rng, 3, 8)
    for side in (-1.0, 1.0):
        fx = 0.25 if side < 0.0 else 0.75
        edge = []
        for i in range(n):
            t = i / float(n - 1)
            wi = w * (1.0 - 0.52 * t) * (0.78 + 0.46 * wob.at(fx, t))
            edge.append((pts[i][0] + nm[i][0] * wi * side,
                         pts[i][1] + nm[i][1] * wi * side))
        i = 0
        while i < n - 1:
            run = rng.randint(3, 7)
            seg = edge[i:i + run]
            if len(seg) > 1:
                brush_path(cv, seg, rng.uniform(0.7, 1.3), 0.5, ink,
                           alpha * rng.uniform(0.45, 0.95))
            i += run + rng.randint(0, 2)


def tree(cv, rng, x, y, h, kind, ink, leaf, scale=1.0, alpha=1.0):
    """One tree. `h` is the trunk height in pixels; everything scales off it."""
    lean = rng.uniform(-0.22, 0.22)
    w = max(0.8, h * rng.uniform(0.024, 0.038))
    top = (x + lean * h, y - h)
    pts = bezier((x, y),
                 (x + lean * h * 0.2 + rng.uniform(-1, 1) * h * 0.06, y - h * 0.38),
                 (x + lean * h * 0.7 + rng.uniform(-1, 1) * h * 0.06, y - h * 0.72),
                 top, 14)
    pts, _ = knuckle(rng, pts, k=rng.randint(1, 2), amp=0.14, start=0.25)
    # Only a trunk with room for two edges gets them. Under about 6 px across
    # the two contours land on the same pixels and the gaps read as holes.
    if w >= 3.0:
        _bark(cv, rng, pts, w, ink, alpha)
    else:
        _taper(cv, pts, w, w * 0.42, ink, alpha)
    # 皮 — a couple of dry strokes down the trunk for bark
    for _ in range(rng.randint(1, 3)):
        o = rng.uniform(-0.5, 0.5) * w
        j = rng.randrange(0, max(1, len(pts) - 5))
        seg = pts[j:j + rng.randint(4, 7)]
        if len(seg) > 1:
            brush_path(cv, [(p[0] + o, p[1]) for p in seg],
                       0.6, 0.5, ink, alpha * rng.uniform(0.25, 0.5))
    tips = []
    claw = 0.6 if kind == "枯" else 0.0
    # 枯 and 阔 branch twice, a pine once: the crown of a pine is carried by
    # its needles, so a third order of twig only muddies it
    deep = 2 if kind in ("枯", "阔", "柳") else 1
    for j in range(rng.randint(4, 7)):
        t = rng.uniform(0.34, 0.99)
        i = min(len(pts) - 1, int(t * (len(pts) - 1)))
        up = -math.pi * 0.5 + rng.uniform(-0.30, 0.30)
        da = rng.uniform(0.40, 1.10) * (1 if j % 2 else -1)
        _twig(cv, rng, pts[i][0], pts[i][1], up + da,
              h * rng.uniform(0.20, 0.40), max(0.7, w * 0.62), ink, alpha,
              deep, tips, claw)

    leaf2 = mix(leaf, I("jiao"), 0.45)    # 浓淡 — two tones of foliage
    if kind == "松":                      # 松针 — needle fans on every tip
        for tx, ty, a, L in tips:
            # 松针 is a cluster the size of a thumbnail, repeated. Drawn big
            # and sparse — a dozen long rays off one centre — it comes out a
            # palm frond, which was the one tree in the repertoire that
            # always looked wrong.
            for _ in range(rng.randint(3, 6)):
                cx = tx + rng.uniform(-0.55, 0.55) * L
                cy = ty + rng.uniform(-0.55, 0.40) * L
                r = h * rng.uniform(0.042, 0.085)
                a0 = rng.uniform(-2.9, -0.3)       # fans open upward
                col = leaf if rng.random() < 0.6 else leaf2
                for q in range(rng.randint(9, 15)):
                    aa = a0 + (q / 12.0 - 0.5) * 2.1
                    brush_seg(cv, cx, cy, cx + math.cos(aa) * r,
                              cy + math.sin(aa) * r, 0.6, 0.35, col,
                              alpha * rng.uniform(0.45, 0.9))
    elif kind == "阔":                    # 点叶 — 介字/胡椒 dot clusters
        # A leaf dab has a direction. 介字点 is three or five strokes thrown
        # off one point, each pressed at the head and lifted at the tail;
        # 胡椒点 is the round one, and a tree made only of round ones looks
        # like a currant bush. So each cluster picks a lean and lays tapered
        # dabs along it, with a handful of peppercorns mixed in — the same
        # distinction index.html draws between `blob` and a circle.
        for tx, ty, a, L in tips:
            for _ in range(rng.randint(3, 6)):
                cx = tx + rng.uniform(-0.9, 0.9) * L
                cy = ty + rng.uniform(-0.9, 0.5) * L
                col = leaf if rng.random() < 0.6 else leaf2
                lean = rng.uniform(-0.6, 0.6)
                pepper = rng.random() < 0.3
                for _ in range(rng.randint(5, 11)):
                    px = cx + rng.uniform(-4.2, 4.2) * scale
                    py = cy + rng.uniform(-3.4, 3.4) * scale
                    al = alpha * rng.uniform(0.35, 0.85)
                    if pepper:
                        disc(cv, px, py, rng.uniform(1.2, 2.4) * scale, col,
                             al, squash=0.78)
                        continue
                    aa = 1.571 + lean + rng.uniform(-0.45, 0.45)
                    dl = rng.uniform(2.4, 5.2) * scale
                    brush_seg(cv, px, py,
                              px + math.cos(aa) * dl, py + math.sin(aa) * dl,
                              rng.uniform(1.3, 2.4) * scale, 0.3 * scale,
                              col, al)
    elif kind == "柳":                    # 垂柳 — drooping strands
        # Willow strands hang in *curtains*: they come off the whole length
        # of a branch rather than its tip, they all lean the same way because
        # they are all in the same air, they are longest in the middle of the
        # branch, and the sideways sweep is a fraction of the strand's own
        # length. Thrown off the tips in random directions with a fixed 6 px
        # sweep they pile into a mop with no inside and no edge — which is
        # exactly what the willow in this repertoire looked like.
        wind = rng.uniform(-0.9, 0.9)
        for tx, ty, a, L in tips:
            k = rng.randint(3, 6)
            for q in range(k):
                u = (q + 0.5) / k - 0.5          # position along the branch
                d = (h * rng.uniform(0.20, 0.34)
                     * (0.68 + 0.32 * (1.0 - 2.0 * abs(u))))
                sx = tx + math.cos(a) * L * u * 0.9
                sy = ty + math.sin(a) * L * u * 0.9
                sw = wind + rng.uniform(-0.30, 0.30)
                st = bezier((sx, sy),
                            (sx + sw * d * 0.10, sy + d * 0.40),
                            (sx + sw * d * 0.26, sy + d * 0.76),
                            (sx + sw * d * 0.42, sy + d), 9)
                # Light. A curtain is read through, not at: six strands per
                # branch at half strength is a willow, the same six at full
                # strength is a black cloud on a stick.
                brush_path(cv, st, 0.85, 0.3, leaf,
                           alpha * rng.uniform(0.22, 0.46))
                # 柳叶 — a lanceolate leaf is a tapered dab, not a bead
                for _ in range(rng.randint(1, 3)):
                    p = st[rng.randrange(2, len(st))]
                    aa = 1.30 + sw * 0.25 + rng.uniform(-0.45, 0.45)
                    dl = rng.uniform(1.6, 3.4) * scale
                    brush_seg(cv, p[0], p[1],
                              p[0] + math.cos(aa) * dl,
                              p[1] + math.sin(aa) * dl,
                              rng.uniform(0.7, 1.3) * scale, 0.25 * scale,
                              leaf2, alpha * rng.uniform(0.24, 0.52))
    elif kind == "丛":                    # 丛树 — a cloud of foliage, no detail
        for _ in range(rng.randint(8, 18)):
            disc(cv, x + rng.uniform(-0.5, 0.5) * h,
                 y - h * rng.uniform(0.45, 1.15),
                 rng.uniform(1.6, 3.6) * scale, leaf,
                 alpha * rng.uniform(0.3, 0.7), squash=0.7)


def far_trees(cv, rng, pts, h, ink, alpha=0.5):
    """远树如簇 — distant trees are a dash and a dot, never a drawing."""
    for x, y in pts:
        hh = h * rng.uniform(0.7, 1.3)
        brush_seg(cv, x, y, x + rng.uniform(-1.5, 1.5), y - hh,
                  rng.uniform(0.6, 1.1), 0.5, ink, alpha * rng.uniform(0.6, 1.0))
        for _ in range(rng.randint(2, 5)):
            disc(cv, x + rng.uniform(-2.4, 2.4), y - hh * rng.uniform(0.6, 1.25),
                 rng.uniform(0.9, 2.0), ink, alpha * rng.uniform(0.35, 0.75),
                 squash=0.7)


# --------------------------------------------------------------------------
# 屋舟 — the human scale: a pavilion, a hut, a bridge, a boat, a traveller
# --------------------------------------------------------------------------

def pavilion(cv, rng, x, y, s, ink):
    """亭 — four posts and a swept roof.

    Buildings in a 山水 are tiny and the eye still goes straight to them,
    because they are the only straight lines in the picture. That means the
    few marks they get have to be the *right* marks: the platform they stand
    on, the railing between the posts, and eaves that turn up at the tips.
    Without those a pavilion is a table, and a table in the middle distance
    is the one thing that can spoil an otherwise convincing sheet.
    """
    w, hh = s, s * 0.78
    brush_seg(cv, x - w * 0.60, y, x + w * 0.60, y, 0.8, 0.8, ink, 0.75)
    # Corner posts in full ink, inner posts thin and pale — they stand behind.
    # Four posts of equal weight plus a beam across the top plus a rail
    # across the middle is a sash window, not a pavilion.
    for dx, wd, al in ((-w * 0.46, 0.8, 0.85), (-w * 0.16, 0.45, 0.22),
                       (w * 0.16, 0.45, 0.22), (w * 0.46, 0.8, 0.85)):
        brush_seg(cv, x + dx, y, x + dx, y - hh * 0.52, wd, wd, ink, al)
    # 栏杆 — the rail. Low, single, and only between the corner posts.
    ry = y - hh * 0.16
    brush_seg(cv, x - w * 0.46, ry, x + w * 0.46, ry, 0.6, 0.6, ink, 0.6)
    for j in range(3):
        bx = x + (j / 2.0 - 0.5) * w * 0.62
        brush_seg(cv, bx, y - hh * 0.02, bx, ry, 0.45, 0.45, ink, 0.4)
    brush_seg(cv, x - w * 0.46, y - hh * 0.52, x + w * 0.46, y - hh * 0.52,
              0.6, 0.6, ink, 0.40)
    tiers = ((0.0, 1.0),) if rng.random() < 0.65 else ((0.0, 1.0),
                                                       (hh * 0.30, 0.74))
    for dy, k in tiers:                            # 重檐 now and then
        eaves = [(x - w * 0.72 * k, y - hh * 0.58 + dy),
                 (x - w * 0.3 * k, y - hh * 0.88 + dy),
                 (x, y - hh + dy),
                 (x + w * 0.3 * k, y - hh * 0.88 + dy),
                 (x + w * 0.72 * k, y - hh * 0.58 + dy)]
        brush_path(cv, eaves, 1.5 * k, 1.5 * k, ink, 0.95)
        for side in (-1.0, 1.0):                   # 翘角 — the upturned tip
            tx = x + side * w * 0.72 * k
            brush_seg(cv, tx, y - hh * 0.58 + dy,
                      tx + side * w * 0.10, y - hh * 0.70 + dy,
                      1.1 * k, 0.4, ink, 0.85)
    brush_seg(cv, x, y - hh, x, y - hh * 1.14, 1.0, 0.6, ink, 0.8)


def hut(cv, rng, x, y, s, ink):
    """茅屋 — a thatched roof, two walls, and that is all a painter gives it.

    The roof is the whole motif, and it has to have *pitch* and *sag*. Drawn
    with a level ridge the width of the house and straight eaves, which is how
    this started, it came out as a bench standing over a crate: four lines of
    one weight with a flat slab across the top. Thatch is heavy and the bamboo
    under it bends, so each slope dips below its own chord and the eave corner
    hangs; the ridge is short, little more than a crease between the two
    slopes. A dark doorway and one pale wall do the rest — they are what say
    there is an inside.
    """
    w, hh = s, s * 0.62
    sag = hh * rng.uniform(0.12, 0.26)
    rx = w * rng.uniform(0.08, 0.18)               # half the ridge
    ry = y - hh * rng.uniform(1.00, 1.18)
    xl, xr = x - w * rng.uniform(0.60, 0.72), x + w * rng.uniform(0.60, 0.72)
    yl = y - hh * rng.uniform(0.36, 0.48)
    yr = y - hh * rng.uniform(0.36, 0.48)

    def slope(xa, ya, xb, yb):
        mx, my = (xa + xb) * 0.5, (ya + yb) * 0.5 + sag
        return bezier((xa, ya), (mx, my), (mx, my), (xb, yb), 8)

    lf = slope(xl, yl, x - rx, ry)
    rt = slope(x + rx, ry, xr, yr)
    brush_path(cv, lf, 0.9, 1.6, ink, 0.92)
    brush_seg(cv, x - rx, ry, x + rx, ry, 1.6, 1.6, ink, 0.92)
    brush_path(cv, rt, 1.6, 0.9, ink, 0.92)
    for ex, ey, sd in ((xl, yl, -1.0), (xr, yr, 1.0)):   # 滴水 — the eave tip
        brush_seg(cv, ex, ey, ex + sd * w * 0.05, ey + hh * 0.14,
                  1.0, 0.5, ink, 0.7)
    # 茅 — thatch combed *down the pitch*, ridge to eave. Hatched at random
    # angles under the ridge, as it was, it reads as dirt on the roof.
    for path in (lf, rt):
        for _ in range(rng.randint(2, 4)):
            px, py = path[int(rng.uniform(0.25, 0.85) * (len(path) - 1))]
            brush_seg(cv, x + (px - x) * rng.uniform(0.30, 0.60),
                      ry + hh * rng.uniform(0.06, 0.26), px, py,
                      0.5, 0.5, ink, rng.uniform(0.16, 0.36))
    # 墙 — two posts and a sill, the far one pale: four lines of equal weight
    # make a crate, not a house standing in light.
    wl, wr = x - w * 0.40, x + w * 0.40
    brush_seg(cv, wl, yl + hh * 0.04, wl, y, 0.8, 0.8, ink, 0.80)
    brush_seg(cv, wr, yr + hh * 0.04, wr, y, 0.8, 0.8, ink, 0.52)
    brush_seg(cv, wl, y, wr, y, 0.7, 0.7, ink, 0.60)
    if rng.random() < 0.75:                        # 门 — a doorway, cut dark
        dw = max(1.0, w * rng.uniform(0.09, 0.15))
        dx = x + rng.uniform(-0.30, 0.30) * w
        brush_seg(cv, dx, y, dx, y - hh * rng.uniform(0.20, 0.32),
                  dw, dw * 0.9, ink, 0.70)
    if rng.random() < 0.5:                         # 篱 — a low fence alongside
        side = 1.0 if rng.random() < 0.5 else -1.0
        fx = x + side * w * 0.5
        fL = w * rng.uniform(0.5, 1.1)
        fh = hh * rng.uniform(0.18, 0.30)
        brush_seg(cv, fx, y - fh, fx + side * fL, y - fh * rng.uniform(0.6, 1.0),
                  0.5, 0.5, ink, 0.45)
        for j in range(rng.randint(3, 5)):
            px = fx + side * fL * (j + 0.5) / 5.0
            brush_seg(cv, px, y + rng.uniform(-0.5, 0.5), px, y - fh,
                      0.45, 0.45, ink, rng.uniform(0.3, 0.5))


def bridge(cv, rng, x0, y0, x1, y1, ink):
    """桥 — a plank bridge, the one thing that makes a gorge legible."""
    mx, my = (x0 + x1) * 0.5, (y0 + y1) * 0.5 - abs(x1 - x0) * rng.uniform(0.10, 0.22)
    deck = bezier((x0, y0), (mx, my), (mx, my), (x1, y1), 12)
    brush_path(cv, deck, 1.5, 1.5, ink, 0.9)
    rail = [(p[0], p[1] - 4.0) for p in deck]
    brush_path(cv, rail, 0.7, 0.7, ink, 0.6)
    for i in range(0, len(deck), 3):
        brush_seg(cv, deck[i][0], deck[i][1], rail[i][0], rail[i][1],
                  0.5, 0.5, ink, 0.5)


def boat(cv, rng, x, y, s, ink, figure=True):
    """渔舟 — hull, awning, and a figure with a pole."""
    hull = [(x - s * 0.5, y - s * 0.06), (x - s * 0.2, y + s * 0.07),
            (x + s * 0.2, y + s * 0.07), (x + s * 0.52, y - s * 0.08)]
    brush_path(cv, hull, 1.3, 1.3, ink, 0.92)
    brush_seg(cv, x - s * 0.5, y - s * 0.06, x + s * 0.52, y - s * 0.08,
              0.7, 0.7, ink, 0.55)
    if rng.random() < 0.7:                         # 篷
        aw = [(x - s * 0.22, y - s * 0.06), (x - s * 0.1, y - s * 0.24),
              (x + s * 0.12, y - s * 0.24), (x + s * 0.22, y - s * 0.06)]
        brush_path(cv, aw, 1.1, 1.1, ink, 0.85)
    if figure:
        fx = x + s * 0.34
        brush_seg(cv, fx, y - s * 0.08, fx, y - s * 0.26, 1.0, 0.9, ink, 0.9)
        disc(cv, fx, y - s * 0.30, s * 0.035, ink, 0.9)
        brush_seg(cv, fx - s * 0.1, y - s * 0.40, fx + s * 0.16, y + s * 0.02,
                  0.5, 0.5, ink, 0.6)              # 竿


def figure(cv, rng, x, y, s, ink):
    """行人 — two strokes and a dot. Scale is the only thing it has to say."""
    brush_seg(cv, x, y, x + rng.uniform(-0.2, 0.2) * s, y - s * 0.62,
              1.0, 0.8, ink, 0.9)
    disc(cv, x, y - s * 0.72, s * 0.11, ink, 0.9)
    brush_seg(cv, x, y - s * 0.3, x + s * 0.22, y - s * 0.12, 0.6, 0.5, ink, 0.7)


def birds(cv, rng, x, y, s, n, ink):
    """飞鸟 — the 人 character, written small."""
    for _ in range(n):
        bx = x + rng.uniform(-1.0, 1.0) * s * 6.0
        by = y + rng.uniform(-1.0, 1.0) * s * 3.0
        w = s * rng.uniform(0.7, 1.3)
        d = rng.uniform(0.25, 0.6)
        brush_seg(cv, bx - w, by - w * d, bx, by, 0.8, 0.5, ink, 0.75)
        brush_seg(cv, bx, by, bx + w, by - w * d * rng.uniform(0.7, 1.3),
                  0.5, 0.8, ink, 0.75)


# --------------------------------------------------------------------------
# 章法 — composition: 三远, the three distances of Guo Xi
# --------------------------------------------------------------------------

SCENES = {
    "高远": dict(
        cn="高远", en="towering distance — Fan Kuan's wall of rock",
        far=(1, 3), far_y=(0.09, 0.20), far_h=(0.05, 0.11),
        main_h=(0.44, 0.58), main_y=(0.64, 0.76), main_w=(0.88, 1.30),
        # never one lone hump: a single summit on a 500 px mass is a traffic
        # cone, however much fBm you hang on its flanks
        peaks=(2, 3), spire=(0.70, 1.42), mid=(1, 2), near=(1, 2),
        water=(0.03, 0.10), fall=0.8, boat=0.25, bridge=0.35,
        trees=(2, 4), near_trees=(1, 3), mist=0.95, dens=0.7,
    ),
    "深远": dict(
        cn="深远", en="deep distance — ridge behind ridge",
        far=(3, 5), far_y=(0.12, 0.26), far_h=(0.05, 0.12),
        main_h=(0.28, 0.40), main_y=(0.58, 0.70), main_w=(0.65, 1.05),
        peaks=(2, 4), spire=(0.55, 1.3), mid=(1, 3), near=(1, 2),
        water=(0.08, 0.18), fall=0.4, boat=0.4, bridge=0.4,
        trees=(3, 6), near_trees=(1, 3), mist=1.0, dens=0.9,
    ),
    "平远": dict(
        cn="平远", en="level distance — Ni Zan's empty water",
        far=(2, 4), far_y=(0.20, 0.31), far_h=(0.04, 0.09),
        main_h=(0.13, 0.22), main_y=(0.46, 0.56), main_w=(0.55, 1.00),
        peaks=(1, 3), spire=(0.5, 1.0), mid=(0, 1), near=(1, 2),
        water=(0.24, 0.38), fall=0.0, boat=0.65, bridge=0.15,
        trees=(2, 4), near_trees=(2, 4), mist=0.7, dens=1.3,
    ),
}

# Tonal profile per depth plane. The far ridges get almost no 皴, almost no
# light and almost no alpha — 远山无皴, and distance in a landscape is carried
# entirely by contrast, never by perspective.
PROF = {
    "far":  dict(tone=(0.12, 0.22), a0=0.02, a1=0.52, w_light=0.10,
                 w_cur=0.18, w_line=0.20, lw=2.2, w_crest=0.18, crest=0.50,
                 w_foot=0.40, foot=0.55),
    # a0 carries the plane: a near mass whose foot goes transparent lets the
    # distant ridges show straight through it, and the picture loses its
    # depth order. The soft foot is the job of the mist band, not the wash.
    "main": dict(tone=(0.36, 0.44), a0=0.16, a1=0.84, w_light=0.26,
                 w_cur=0.30, w_line=0.62, lw=2.2, w_crest=0.34, crest=0.26,
                 w_foot=0.36, foot=0.30),
    "mid":  dict(tone=(0.32, 0.40), a0=0.18, a1=0.82, w_light=0.24,
                 w_cur=0.26, w_line=0.54, lw=2.0, w_crest=0.30, crest=0.30,
                 w_foot=0.34, foot=0.32),
    "near": dict(tone=(0.42, 0.52), a0=0.26, a1=0.74, w_light=0.28,
                 w_cur=0.34, w_line=0.74, lw=2.8, w_crest=0.36, crest=0.28,
                 w_foot=0.30, foot=0.26),
}

TITLE_A = ["溪山", "烟雨", "秋山", "春山", "寒林", "云岫", "幽涧", "松泉",
           "远岫", "江天", "雨霁", "晴峦", "空山", "层峦", "水村", "渔隐",
           "山居", "碧涧", "暮霭", "万壑", "疏林", "寒江"]
TITLE_B = ["图", "小景", "图卷", "册页", "写意", "清趣", "小品", "卷"]
ACTION = ["写意", "戏墨", "拟古", "并记", "笔意", "南窗写意", "客窗戏墨"]
SEAL_WORDS = ["烟霞", "林泉", "闲云", "卧游", "山静", "听泉", "墨缘", "清赏",
              "得趣", "心迹", "一壑"]


def vocabulary():
    """Every character this painter can write — the extractor bakes just
    these, exactly as for gongbi.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [v["cn"] for v in CUN.values()]
             + list(SCENES) + list(STYLES)
             + [TIANGAN, DIZHI, "日月年写于图山水春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a font baked for this module's vocabulary; fall back to
    gongbi's (the two share most characters), and to no 题款 at all."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("shanshui_font", "gongbi_font"):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        try:
            return BrushFont(m.DATA, m.EM)
        except Exception as exc:
            print("  (font blob %s unreadable: %s)" % (mod, exc), file=sys.stderr)
    return None


FONT = _load_font()


def _ganzhi(rng):
    return TIANGAN[rng.randrange(10)] + DIZHI[rng.randrange(12)]


def inscribe(cv, rng, occ, S, title):
    """题款 — measure the block first, then let 留白 decide where it goes."""
    W, H = cv.w, cv.h
    gs = S * rng.uniform(0.040, 0.055)
    ss = gs * rng.uniform(0.52, 0.66)
    lead, lead_s = rng.uniform(1.12, 1.26), rng.uniform(1.18, 1.34)
    season = "春夏秋冬"[rng.randrange(4)]
    sign = _ganzhi(rng) + season + "日" + ACTION[rng.randrange(len(ACTION))]
    gap = gs * 0.45
    sign_dy = gs * rng.uniform(1.1, 2.0)
    seal_s = gs * rng.uniform(0.80, 1.0)
    bw = gs + gap + ss
    bh = max(len(title) * gs * lead,
             sign_dy + len(sign) * ss * lead_s + gs * 0.25 + seal_s)
    margin = S * 0.035
    # The middle of the paper is not for writing, however empty it looks. A
    # column of 题款 standing in the centre of a landscape reads as a caption
    # over the view; it belongs against one of the two long edges, which is
    # where open water and sky are anyway. One virtual disc says so.
    r0 = min(W, H) * 0.20
    for fy in (0.26, 0.42, 0.58, 0.74):
        occ.add(W * 0.5, H * fy, r0)
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.7)
        paint_column(cv, rng, FONT, title, bx + gap + ss, by, gs, lead=lead)
        end = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, ss,
                           lead=lead_s, alpha=(0.62, 0.84))
        paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                   bx + (ss - seal_s) * 0.5, end + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
    if rng.random() < 0.55:                       # 压角章
        s2 = S * rng.uniform(0.036, 0.050)
        for cx, cy in ((W * 0.07, H * 0.89), (W * 0.89 - s2, H * 0.89),
                       (W * 0.07, H * 0.07), (W * 0.89 - s2, H * 0.07)):
            if occ.overlap(cx, cy, cx + s2, cy + s2, s2 * 0.5) == 0.0:
                paint_seal(cv, rng, FONT,
                           SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                           cx, cy, s2)
                break


def compose(cv, rng, scene_key, style_key, cun_key):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc, st = SCENES[scene_key], STYLES[style_key]
    cun = CUN[cun_key]
    set_light(rng)
    ground = gb.paint_ground(cv, rng)
    paper = bytes(cv.buf)                 # the bare sheet, kept for 留白
    grain = Grain(rng)
    occ = Occupancy(W, H)
    scale = S / 1000.0

    ink_n, ink_d = I("nong"), I("dan")
    tree_ink = jitter(rgb(st["tree"]), rng, 6)
    leaf_ink = jitter(rgb(st["leaf"]), rng, 8)
    near3 = tuple(rgb(c) for c in st["near"])
    far3 = tuple(rgb(c) for c in st["far"])
    bow = st["bow"]
    mode = cun["mode"]
    per = rng.uniform(*cun["per"]) * max(0.7, scale)
    amp = rng.uniform(*cun["amp"])
    scrub = cun["scrub"]
    flip = rng.choice((1, -1))            # which side the composition leans

    def mass(role, span, hgt, base, x0=None, anchor=None, peaks=(1, 3),
             sub=(1, 3), fade=0.0, texture=True, tone=None, spire=(0.5, 1.3)):
        """One rock mass: a dominant ridge plus a few lower shoulders in
        front of it, each carrying its own contour.

        石分三面 — a painter builds even a single boulder out of two or three
        interlocking 勾 contours. One smooth silhouette filled with a wash is
        the single loudest tell that a shape was computed rather than drawn,
        and it was the first thing wrong with this generator.
        """
        p = PROF[role]
        span = max(10, int(span))
        # 倪瓒's rocks are low flat banks, never spires — the 皴 and the
        # silhouette belong to the same hand
        flatten = cun_key == "折带" and role != "far"
        lo, hi = p["tone"]
        t0 = tone if tone is not None else rng.uniform(lo, hi)
        out = []
        for j in range(max(1, rng.randint(*sub))):
            db = 0.0
            if j == 0:
                sw, sh = span, hgt
            else:
                sw = max(8, int(span * rng.uniform(0.16, 0.52)))
                sh = hgt * rng.uniform(0.20, 0.72)
                # A shoulder's height is drawn from the parent's and its
                # width from the parent's span; left independent, a narrow
                # draw of one and a tall draw of the other give a stalagmite.
                # Even clamped to its own width it was a fang — a lobe of
                # rock in front of a peak is *wider* than it is tall, always,
                # and a cluster of them was the last thing in the generator
                # still reading as a row of teeth.
                lim = sw * rng.uniform(0.50, 0.95)
                if flatten:
                    lim *= 0.55     # 折带 country is low banks, not towers
                elif role == "mid":
                    lim *= 0.72     # a bank across the water is a lobe of
                                    # rock; sharpen it and the middle
                                    # distance grows a row of teeth
                if sh > lim:
                    sh = lim
                # a shoulder can sit on a terrace partway up the slope rather
                # than down on the shore — that is what turns a cone into a
                # pile of rock, and it costs one subtraction
                db = hgt * rng.uniform(0.0, 0.38)
            # 台 — a sheared plateau, and only where the mass is wide enough
            # to carry one. The two decisions below have to be made together:
            # the flaring foot (an exponent above 1) only makes sense *under*
            # a shear, because an exponent above 1 also brings the crest to a
            # point, and a point on a tall narrow mass is a needle. Decided
            # separately, as they were, 折带 country in a 高远 composition
            # grew a 300 px spire with two ruled flanks.
            mesa = flatten and sw > sh * 1.7 and rng.random() < 0.75
            top = ridge(rng, sw, sh,
                        peaks=rng.randint(*peaks) if j == 0 else rng.randint(1, 2),
                        rough=rng.uniform(0.48, 0.62),
                        # even a distant hill is not a ruled line: with almost
                        # no detail the long flank of a hump whose exponent is
                        # near 1 draws a 400 px straight edge across the sky
                        detail=0.30 if role == "far" else 0.42,
                        # one or two sheer steps read as 断崖; five turn the
                        # silhouette into a staircase, and the 勾 line draws
                        # every riser
                        breaks=rng.randint(0, 1) if role == "far"
                               else rng.randint(1, 3),
                        # a flat bank wants *flaring* feet (exponent above 1)
                        # under its sheared top; exponents below 1 are steep
                        # at the foot and turn the mesa into a brick
                        # A shoulder is a lobe of rock, not a tooth. Exponents
                        # below 1 are near-vertical at the foot, so a narrow
                        # one came out as a straight-sided spike and a mass
                        # grew a row of shark's teeth around its summit.
                        spire=(spire if j == 0 else (1.10, 1.48))
                              if not flatten
                              else ((1.12, 1.55) if mesa else (0.50, 0.95)),
                        # ...and the shear is only for a mass that is wide
                        # enough to carry a plateau. Cap a narrow one and it
                        # comes out a tombstone: flat top, two vertical
                        # sides, no rock anywhere in it.
                        flat=0.55 if mesa else 0.0)
            # Trim to the ridge's real support. summits() takes the max of a
            # few humps, so outside all of them the envelope is flat zero —
            # and a run of zeroes is a dead-straight crest at the base line,
            # drawn with the full 勾 contour along it. That was the ruled
            # horizontal bar the renders kept growing in odd corners.
            h0 = sh * 0.03
            a = 0
            while a < len(top) - 1 and top[a] <= h0:
                a += 1
            b = len(top) - 1
            while b > a and top[b] <= h0:
                b -= 1
            if b - a < 6:
                continue
            top = top[a:b + 1]
            sw = b - a
            s = 0
            for i in range(1, len(top)):
                if top[i] > top[s]:
                    s = i
            if j == 0:
                if x0 is None:
                    # Place a mass by its *summit*, not by its left foot: the
                    # composition cares where the peak falls, and the feet are
                    # usually off the paper anyway.
                    x0 = int(anchor - s)
                else:
                    x0 += a             # the caller named the untrimmed foot
                sx = x0
                peak = x0 + s
                t = t0
            else:
                # Shoulders belong *around the summit*, not scattered over the
                # whole span: spread across the full width they stop being one
                # mountain and become separate cones standing in the water.
                sx = int(peak + span * rng.uniform(-0.34, 0.34) - sw * 0.5)
                lo_x, hi_x = x0, x0 + span - sw
                sx = lo_x if sx < lo_x else (hi_x if sx > hi_x else sx)
                t = min(0.95, t0 * rng.uniform(1.02, 1.20))
            # the foot is never a ruled line: even a flat shore wanders a
            # couple of pixels, and a horizontal seam across the paper is
            # visible from across the room
            bwob = fbm(rng, sw, 0.6, hgt * 0.012 + 1.5, lo=1, hi=4)
            # A shoulder on a terrace stands on that terrace: its base goes
            # up with its crest. Keeping the base down on the shore gave it
            # a `db`-tall vertical cliff at each end, a straight seam down
            # through the wash.
            R = Ridge(sx, [base - db - v for v in top],
                      [base - db + hgt * 0.03 + v for v in bwob],
                      *(far3 if role == "far" else near3),
                      tone=t, a0=p["a0"], a1=p["a1"],
                      w_light=p["w_light"], w_cur=p["w_cur"],
                      w_line=p["w_line"], line=p["lw"] * max(1.0, scale),
                      w_crest=p["w_crest"], crest=p["crest"],
                      w_foot=p["w_foot"], foot=p["foot"],
                      cun=mode if texture else -1,
                      # The stroke scales with the rock. A 60 px boulder given
                      # the 主峰's 24 px axe-cuts gets two strokes and comes
                      # out a smooth glass pyramid; the brush a painter picks
                      # up is the one that fits what is in front of them.
                      per=per * rng.uniform(0.85, 1.20)
                          * max(0.40, min(1.25, sh / (H * 0.26))),
                      amp=amp,
                      scrub=scrub, bow=bow, grain=grain,
                      fade=fade if j == 0 else max(fade, sh * 0.30))
            # 山脚 — only where the base is actually on the paper. The near
            # plane's base is parked below the bottom edge on purpose, and a
            # mass whose foot the viewer never sees needs no foot.
            under = (bytes(cv.buf)
                     if role != "far" and base - db < H - 2 and sh > H * 0.05
                     else None)
            R.render(cv)
            if under is not None:
                foot_lobes(cv, rng, under, R, ink_d, scale,
                           strength=0.95 if j == 0 else 0.70)
            # the drawn 皴, over the grain. Count scales with the area the
            # brush has to cover, not with the mass's importance.
            if texture:
                R.cun_strokes(cv, rng,
                              int(sw * math.sqrt(sh) * 0.007 * sc["dens"]),
                              max(0.7, scale))
            R.register(occ)
            if j and role != "far" and rng.random() < 0.7:
                R.moss(cv, rng, rng.randint(2, 6), ink_n, scale * 0.8)
            out.append(R)
        return out

    # ---- 布局 — everything is measured off one shore line -------------
    y_shore = H * rng.uniform(*sc["main_y"])     # the far bank: 主峰's foot
    water_h = rng.uniform(*sc["water"]) * H
    y_water = min(H * 0.97, y_shore + water_h)   # the near bank
    main_h = H * rng.uniform(*sc["main_h"])

    # ---- 远山 — pale, untextured, and never sharper than the near rock -
    nfar = rng.randint(*sc["far"])
    feet = y_shore
    for k in range(nfar):
        t = k / float(nfar - 1) if nfar > 1 else 0.0
        hgt = H * rng.uniform(*sc["far_h"]) * (1.0 - 0.18 * t)
        by = H * rng.uniform(*sc["far_y"]) + t * H * 0.055 + hgt * 0.22
        R = mass("far", W * rng.uniform(0.7, 1.4), hgt, by,
                 x0=int(W * rng.uniform(-0.35, 0.22)), peaks=(1, 3),
                 sub=(1, 1), fade=hgt * 0.55, texture=False,
                 tone=0.11 + 0.11 * t)[0]
        feet = max(feet, by)
        if k == nfar - 1 and rng.random() < 0.7:
            far_trees(cv, rng, R.sample(rng, rng.randint(3, 8), 0.1, 0.9),
                      8.0 * scale, ink_d, 0.35)
    if nfar:
        mist(cv, rng, paper, feet - H * 0.01, H * 0.05, 0.60 * sc["mist"])

    # ---- 主峰 — the one mass the whole painting is hung on -------------
    anchor = W * (rng.uniform(0.28, 0.42) if flip > 0 else rng.uniform(0.58, 0.72))
    mains = mass("main", W * rng.uniform(*sc["main_w"]), main_h, y_shore,
                 anchor=anchor, peaks=sc["peaks"], sub=(2, 4),
                 fade=main_h * 0.16, spire=sc["spire"])
    main = mains[0]
    main.moss(cv, rng, int(16 * sc["dens"]), ink_n, scale)

    # 瀑布 — hung in the lowest gully of the main mass, with its 水口 below
    if rng.random() < sc["fall"] and main_h > H * 0.2:
        # A fall hangs in a *notch* — a dip in the crest with rock standing on
        # both sides of it. Taking the lowest crest column instead finds the
        # skirt of the mass almost every time, which is the one place there
        # is no cleft and often not even any canvas: on a mass wider than the
        # sheet it put the fall a hundred pixels off the left edge, so most
        # paintings that asked for one silently got nothing. Score each
        # interior column by how far the crest climbs on *both* sides of it.
        n = len(main.top) - 1
        d = max(4, int(n * 0.07))
        j, gy, best = None, None, 3.0
        for i in range(d, n - d):
            px = main.x0 + i
            if px < W * 0.12 or px > W * 0.88:
                continue
            t = main.top[i]
            a, b = t - main.top[i - d], t - main.top[i + d]
            v = a if a < b else b
            if v > best:
                best, gy, j = v, t, i
        if j is not None and y_shore - gy > H * 0.05:
            fx = main.x0 + j
            fy0 = gy + main_h * rng.uniform(0.10, 0.24)
            fy1 = min(y_shore + 6.0, y_water - 2.0)
            waterfall(cv, rng, paper, fx, fy0, fy1,
                      max(2.2, main_h * 0.020), ink_d)
            occ.add(fx, (fy0 + fy1) * 0.5, (fy1 - fy0) * 0.45)

    far_trees(cv, rng, main.sample(rng, rng.randint(*sc["trees"]), 0.12, 0.88),
              13.0 * scale, ink_d, 0.5)
    mist(cv, rng, paper, y_shore - main_h * rng.uniform(0.02, 0.14),
         H * 0.042, 0.55 * sc["mist"])

    # ---- 中景 — banks across the water, with trees and a roof ---------
    mids = []
    for k in range(rng.randint(*sc["mid"])):
        hgt = H * rng.uniform(0.08, 0.17)
        by = y_shore + water_h * rng.uniform(0.04, 0.46)
        # A mid-ground bank is a rounded rock. Concave exponents give it a
        # pointed summit, which across three or four banks is a row of
        # shark's teeth; convex ones used to stand it on vertical feet, but
        # the wedge clamp in `ridge` now tapers those, so the dome is
        # available again and it is what a bank actually looks like.
        Rs = mass("mid", W * rng.uniform(0.32, 0.78), hgt, by,
                  anchor=W * rng.uniform(0.14, 0.86), peaks=(1, 3),
                  sub=(2, 4), fade=hgt * 0.22, spire=(0.42, 0.86))
        R = Rs[0]
        R.moss(cv, rng, int(7 * sc["dens"]), ink_n, scale * 0.8)
        mids.append(R)
        for x, y in R.sample(rng, rng.randint(1, 3), 0.15, 0.85):
            tree(cv, rng, x, y + 1.0, H * rng.uniform(0.045, 0.080),
                 rng.choice(("松", "阔", "阔", "丛")), tree_ink, leaf_ink,
                 scale * 0.85, alpha=0.82)
        if rng.random() < 0.6:
            bx, by2 = R.sample(rng, 1, 0.2, 0.8, flatish=True)[0]
            s = S * rng.uniform(0.035, 0.055)
            (pavilion if rng.random() < 0.45 else hut)(cv, rng, bx, by2 + 1.0,
                                                       s, ink_n)
            occ.add(bx, by2 - s * 0.4, s * 0.9)

    # ---- 水 — open water is paper, plus a few lines to say it is wet ---
    if water_h > H * 0.04:
        water_lines(cv, rng, y_shore + 5.0, y_water - 2.0, ink_d, sc["dens"])
        # 汀渚 — more of them the wider the water, since a wide band is
        # exactly the case where ripples alone leave the paper empty
        shore_rocks(cv, rng, y_shore + 6.0, y_water - 3.0, ink_n, scale,
                    k=rng.randint(1, 2) + int(water_h / (H * 0.10)))
        if rng.random() < sc["boat"]:
            bx = W * rng.uniform(0.18, 0.82)
            by = y_shore + water_h * rng.uniform(0.45, 0.95)
            s = S * rng.uniform(0.045, 0.075)
            boat(cv, rng, bx, by, s, ink_n)
            occ.add(bx, by, s * 0.7)

    # ---- 近景 — the near bank: darkest ink, biggest trees, human scale -
    nears = []
    base_n = H * 1.04
    for k in range(rng.randint(*sc["near"])):
        # 坡岸, not an island. The near ground enters from a bottom corner
        # and its flank sweeps down across the paper; a mass anchored inside
        # the frame gives a sharp summit standing in open water, which is
        # the one thing the near plane must never look like. So the summit
        # is parked just off the edge and only the slope is in the picture.
        if k == 0:
            hgt = max(H * 0.12, (base_n - y_water) * rng.uniform(0.55, 0.95))
            span = W * rng.uniform(0.95, 1.7)
            ax = W * (-0.04 if flip > 0 else 1.04)
        else:
            hgt = max(H * 0.08, (base_n - y_water) * rng.uniform(0.35, 0.62))
            span = W * rng.uniform(0.45, 0.9)
            ax = W * (1.02 if flip > 0 else -0.02)
        Rs = mass("near", span, hgt, base_n, anchor=ax, peaks=(1, 2),
                  sub=(3, 5) if k == 0 else (2, 3), spire=(0.30, 0.70))
        R = Rs[0]
        R.moss(cv, rng, int(12 * sc["dens"]), I("jiao"), scale)
        nears.append(R)
        for x, y in R.sample(rng, rng.randint(*sc["near_trees"]), 0.08, 0.92):
            kind = rng.choice(("松", "松", "阔", "枯", "柳"))
            # A near tree is the human scale of the picture, not its subject.
            # At a fifth of the sheet, in full ink, it stops being scale and
            # starts being a silhouette pasted over the mountain.
            th = H * rng.uniform(0.085, 0.155)
            tree(cv, rng, x, y + 2.0, th, kind, tree_ink, leaf_ink, scale,
                 alpha=0.92)
            occ.add(x, y - th * 0.6, th * 0.5)
        if k == 0 and rng.random() < 0.4:
            px, py = R.sample(rng, 1, 0.25, 0.75, flatish=True)[0]
            figure(cv, rng, px, py + 1.0, S * 0.028, ink_n)

    # 桥 — it has to land on something. Picking the far end at random leaves
    # a plank hanging over open water attached to one bank, which is the one
    # mistake in the whole painting a viewer spots instantly. Both ends are
    # resolved against real crest lines, and the bridge is simply skipped when
    # no pair of banks is close enough to span.
    if mids and rng.random() < sc["bridge"]:
        a = mids[rng.randrange(len(mids))]
        x0a, x1a = a.span()
        others = [m for m in mids + nears + mains if m is not a]
        span_ok = None
        for _ in range(40):
            bx0 = max(6.0, min(W - 6.0,
                               x0a + (x1a - x0a) * rng.uniform(0.10, 0.90)))
            by0 = a.crest_y(bx0)
            if by0 is None:
                continue
            bx1 = bx0 + W * rng.uniform(0.10, 0.26) * rng.choice((1, -1))
            if bx1 < 6.0 or bx1 > W - 6.0:
                continue
            for o in others:
                by1 = o.crest_y(bx1)
                if by1 is not None and abs(by1 - by0) < H * 0.055:
                    span_ok = (bx0, by0, bx1, by1)
                    break
            if span_ok:
                break
        if span_ok:
            bx0, by0, bx1, by1 = span_ok
            bridge(cv, rng, bx0, by0 + 3.0, bx1, by1 + 3.0, ink_n)

    if rng.random() < 0.5:
        birds(cv, rng, W * rng.uniform(0.25, 0.75), H * rng.uniform(0.10, 0.26),
              4.0 * scale, rng.randint(2, 5), ink_d)

    # ---- 题款 ---------------------------------------------------------
    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))])
    inscribe(cv, rng, occ, S, title)
    return title, ground


def age(cv, rng):
    """A last breath of ink dust over everything, as gongbi does."""
    W, H = cv.w, cv.h
    for _ in range(int(W * H * 0.00010)):
        disc(cv, rng.uniform(0, W), rng.uniform(0, H),
             rng.uniform(0.6, 1.8), I("dan"), rng.uniform(0.06, 0.26))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def paint(path, size=(1000, 1400), seed=None, scene=None, style=None,
          cun=None, quiet=False):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    yk = style or sorted(STYLES)[rng.randrange(len(STYLES))]
    ck = cun or sorted(CUN)[rng.randrange(len(CUN))]
    for key, tbl, what in ((sk, SCENES, "scene"), (yk, STYLES, "style"),
                           (ck, CUN, "cun")):
        if key not in tbl:
            raise SystemExit("unknown %s %r; try --list" % (what, key))

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, yk, ck)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-26s %s  %s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk, CUN[ck]["cn"],
                 STYLES[yk]["cn"], ground, seed, size[0], size[1],
                 time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese shanshui landscape paintings.")
    ap.add_argument("-o", "--out", default="shanshui.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="高远|深远|平远")
    ap.add_argument("--style", help="水墨|浅绛|青绿")
    ap.add_argument("--cun", help="披麻|斧劈|雨点|折带|卷云")
    ap.add_argument("--size", default="1000x1400", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("三远 (composition):")
        for k, v in SCENES.items():
            print("  %-6s %s" % (k, v["en"]))
        print("皴法 (rock texture):")
        for k, v in CUN.items():
            print("  %-6s %-10s %s" % (k, v["cn"], v["who"]))
        print("设色 (colour):")
        for k, v in STYLES.items():
            print("  %-6s %s" % (k, v["en"]))
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 1000x1400")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("山水 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.style, a.cun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
