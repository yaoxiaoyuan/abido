#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
song.py - 松 — pine, drawn the way the ink tradition draws it.

The eleventh painter in this family, and the one whose subject the old
manuals put first: 「画松之法，先立干，次分枝，后点叶」. A pine is three
marks and no more, and all three of them are *different from everything
else in this repertoire*:

  * 干如虬龙 — the trunk twists. Plum wood breaks at hard joints and runs
    straight between them (mei.wood_path); bamboo is rigid between nodes;
    a pine bends continuously and reverses, which is a serpentine, not a
    polyline and not a bezier. `trunk_path` is wood_path's skeleton with a
    slow sinusoid laid over it and its kinks softened — the same code with
    the opposite temperament.
  * 皮如龙鳞 — the bark is *scales*, rows of small arcs staggered like
    masonry, not the streaked 皴 of plum. `scales` draws them across the
    grain, because an arc whose chord lies across a form and whose belly
    bulges along it is the cheapest statement in the language that the
    form is round.
  * 叶如攒针 — and this is the whole picture. Pine needles are never drawn
    as leaves; they are drawn as *bundles* — seven to fifteen straight
    strokes radiating from one point, 扇形 or 轮形 — and bundles are
    gathered into 团, plates of needle that the eye reads as one mass. Two
    things make the difference between a pine and a diagram of wheels:
    every plate hangs off a twig that was actually drawn (gongbi's and
    grape's attachment lesson), and the plates are laid back to front in
    三墨 with the front ones *lifting the paper* out from under themselves
    (mei's Reserve), so a mass in front of a mass reads as in front rather
    than as a darker patch of lattice.

Everything else is the family's: gongbi's canvas, paper, occupancy, font
and seals; bamboo's ink table, washes, moon and snow; orchid's `sweep` and
its 坡石; mei's `Reserve`, `bough`, `blot` and `flyback` for the wood. Zero dependencies outside the standard library.

    python3 song.py -n 6 -o /tmp/song.png
    python3 song.py --size 900x1300 -o pine.png
    python3 song.py --list                      # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Occupancy, BrushFont, clamp01, disc, rgb,
                    paint_column, paint_seal, TIANGAN, DIZHI)
from bamboo import I, dilute, veil, moon, snowfall, cluster_sites
from orchid import stone
from mei import Reserve, bough, blot, flyback, drag

TAU = math.tau


# --------------------------------------------------------------------------
# 设色 — the three schools a pine may be painted in.
#
# `wood` and `needle` are each the two stops of a stroke's ramp — the pale
# flank and the loaded core — and they are kept apart on purpose: the one
# thing every 松 scroll agrees on is that the needles are the darkest, most
# tightly written mark on the sheet and the trunk is the broadest and the
# wettest. Give them the same ink and the tree turns into a shrub.
# --------------------------------------------------------------------------

STYLES = {
    "墨松": dict(cn="墨松", en="plain ink — 水墨松石",
                 wood=("#938B7C", "#332F29"), needle=("#796F62", "#23201D"),
                 rock=("#B6B0A0", "#3A352E"), cone="#2A2522",
                 leaf=("#8E8678", "#2B2724")),
    "浅绛": dict(cn="浅绛", en="light crimson — 赭石 bark, ink needles",
                 wood=("#C09268", "#6A3C22"), needle=("#7C7465", "#262320"),
                 rock=("#C4AE92", "#6A523C"), cone="#4A2A18",
                 leaf=("#9A9078", "#3A3024")),
    "苍翠": dict(cn="苍翠", en="malachite over ink — 青松",
                 wood=("#9A9384", "#38332B"), needle=("#8AA089", "#28422F"),
                 rock=("#ACB2A2", "#3C443A"), cone="#24301F",
                 leaf=("#93A68C", "#2C4430")),
}

# 章法 — the four framings.
#
# `L` is the trunk's length and `w0` its half-width at the root, both as
# fractions of the sheet's short side; `limbs` how many 大枝 leave it;
# `span` the stretch of trunk they may leave from — and it starts high,
# because 松 is 「下不生枝」: a pine that branches from its foot is a bush.
# `plate` is the radius of one needle 团 and `plates` how many the sheet
# carries; `old` drives the bark, the 飞白 and the moss at once.
SCENES = {
    "孤松": dict(cn="孤松", en="one pine standing — 亭亭如盖",
                 n=1, L=(0.78, 0.94), w0=(0.028, 0.038), lean=(0.05, 0.20),
                 limbs=(3, 4), span=(0.34, 0.94), whips=(2, 3), old=0.90,
                 plate=(0.078, 0.102), plates=(14, 19), kink=0.17,
                 crown=1.00, rock=0.22, hang=0.00, turf=1.0),
    "双松": dict(cn="双松", en="two pines, one old one young — 一主一宾",
                 n=2, L=(0.68, 0.84), w0=(0.020, 0.029), lean=(0.04, 0.22),
                 limbs=(2, 3), span=(0.36, 0.95), whips=(2, 3), old=0.72,
                 plate=(0.064, 0.086), plates=(10, 14), kink=0.19,
                 crown=0.92, rock=0.34, hang=0.00, turf=1.0),
    # 偃盖 is a *crown*, not a place. The name is 「偃盖如伞」 — an old pine
    # whose head has stopped climbing and gone out sideways until it lies
    # over the ground like a lowered canopy. The first draft read the word
    # as 悬崖松 and built a cliff for it out of orchid's `ledge`: a grey
    # stepped mass filling one corner of every sheet in the scene, with the
    # pine stuck to its lip. Two things were wrong with that and only one of
    # them was the cliff's drawing. 偃盖 does not need a cliff — the shape
    # has to be won from the tree, which is the harder and the right way —
    # and a borrowed mass that is the biggest thing on the paper makes the
    # painting about the rock. So the cliff is gone and the character is
    # where it belongs: a short trunk that leans early, a growth attractor
    # laid nearly flat so every limb runs out instead of up, and 枝垂 on the
    # outer boughs so the canopy's rim comes down.
    "偃盖": dict(cn="偃盖", en="an old pine gone flat overhead — 偃盖如伞",
                 n=1, L=(0.60, 0.74), w0=(0.026, 0.036), lean=(0.04, 0.20),
                 limbs=(3, 5), span=(0.26, 0.92), whips=(2, 3), old=1.00,
                 plate=(0.072, 0.096), plates=(15, 21), kink=0.22,
                 crown=0.70, rock=0.30, hang=0.34, turf=1.0),
    "松石": dict(cn="松石", en="pine and rock — 松石图",
                 n=1, L=(0.72, 0.88), w0=(0.026, 0.036), lean=(0.06, 0.24),
                 limbs=(3, 4), span=(0.32, 0.93), whips=(2, 3), old=0.95,
                 plate=(0.074, 0.098), plates=(13, 18), kink=0.18,
                 crown=0.95, rock=1.00, hang=0.00, turf=0.6),
}

# 风致 — one air for the whole sheet. `comb` is the wind: how hard every
# needle in the painting is raked toward one quarter, which is the only
# honest way to draw 松涛 — a pine in wind does not wave, its needle plates
# flatten and all point downwind together.
AIR = {
    "清晓": dict(cn="晴松", en="still, at first light",
                 lean=0.00, wet=1.00, dry=0.30, k=1.00, comb=0.00),
    "松风": dict(cn="松涛", en="wind through the needles — 万壑松风",
                 lean=0.26, wet=0.84, dry=0.48, k=1.00, comb=0.85),
    "积雪": dict(cn="雪松", en="snow lying on the plates — 岁寒",
                 lean=0.06, wet=1.10, dry=0.22, k=0.94, comb=0.10,
                 veil=("#8A94A2", 0.09), snow=1.0),
    "月下": dict(cn="月松", en="by moonlight, 云锁松腰",
                 lean=0.10, wet=1.26, dry=0.18, k=0.86, comb=0.05,
                 veil=("#6E7A86", 0.10), moon=1.0, cloud=1.0),
}

TITLE_A = ["古松", "苍松", "听松", "松风", "岁寒", "长青", "虬枝", "云壑",
           "高士", "劲节", "独立", "清荫", "盘龙", "万壑", "石畔", "涧上"]
TITLE_B = ["图", "小景", "写意", "图卷", "册页", "小品", "清趣"]
TITLE_AIR = {
    "清晓": ["晴窗", "晓色", "清晖"],
    "松风": ["松涛", "风入", "天籁"],
    "积雪": ["雪压", "岁寒", "寒柯"],
    "月下": ["月明", "夜深", "云根"],
}
ACTION = ["写生", "戏作", "写意", "拟古", "仿李唐笔意", "南窗写松", "并记"]
SEAL_WORDS = ["长青", "岁寒", "劲节", "听涛", "墨缘", "写心", "老干", "清赏",
              "云根", "龙鳞", "不凋", "贞松", "安之"]


# --------------------------------------------------------------------------
# the small shared helpers — the same four every painter in this family
# keeps a copy of, because they are ten lines each and importing a sibling's
# private names would couple two modules for no gain.
# --------------------------------------------------------------------------

def _perp(pts, i, d):
    """The point `d` pixels to one side of `pts[i]`, along *its own* normal."""
    j = min(len(pts) - 1, i + 1)
    k = max(0, i - 1)
    dx, dy = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
    n = math.hypot(dx, dy) or 1.0
    return pts[i][0] - dy / n * d, pts[i][1] + dx / n * d


def _arc(ctrl, n):
    """A quadratic or cubic through `ctrl`, sampled `n` ways."""
    out = []
    if len(ctrl) == 3:
        (ax, ay), (bx, by), (cx, cy) = ctrl
        for i in range(n + 1):
            t = i / n
            s = 1.0 - t
            out.append((s * s * ax + 2 * s * t * bx + t * t * cx,
                        s * s * ay + 2 * s * t * by + t * t * cy))
    else:
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = ctrl
        for i in range(n + 1):
            t = i / n
            s = 1.0 - t
            a, b = s ** 3, 3 * s * s * t
            c, d = 3 * s * t * t, t ** 3
            out.append((a * ax + b * bx + c * cx + d * dx,
                        a * ay + b * by + c * cy + d * dy))
    return out


def _spread(rng, n, lo, hi, sep):
    """`n` parameters in [lo, hi], no two closer than `sep` — 忌对节."""
    out = []
    for _ in range(n * 8):
        if len(out) >= n:
            break
        t = rng.uniform(lo, hi)
        if all(abs(t - u) > sep for u in out):
            out.append(t)
    return sorted(out)


def _turn(a, b):
    """The signed difference b - a, folded into (-pi, pi]."""
    return (b - a + math.pi) % TAU - math.pi


def _at(pts, t):
    """The point and tangent a fraction `t` along a path."""
    n = len(pts) - 1
    f = clamp01(t) * n
    i = min(n - 1, int(f))
    fr = f - i
    x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * fr
    y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * fr
    return x, y, gb.tangent(pts, min(n, i + 1))


def _arclen(pts):
    """Cumulative length along a path, and its total."""
    cum = [0.0]
    for q in range(len(pts) - 1):
        cum.append(cum[-1] + math.hypot(pts[q + 1][0] - pts[q][0],
                                        pts[q + 1][1] - pts[q][1]))
    return cum, cum[-1]


def _emerge(pts, ppts, phws, w0, lap=0.78):
    """Clip a child's path back to the point where it leaves its parent.

    出枝 — a branch starts at a point on the parent's *centreline*, because
    that is the only place its direction is defined; but everything from
    there to the parent's silhouette is wood *inside* other wood, and under
    a multiply composite that stretch does not disappear, it doubles. It
    was the single ugliest thing on the sheet: the child's two 双勾 contours
    ran straight across the trunk and out the far side, its round base cap
    sat in the middle of the trunk as a dark bead, and the two fills
    multiplied into a hard parallelogram at every fork.

    So the child is cut back to where it emerges. The test is on the
    child's **near edge**, not on its centreline, which is the whole of
    why the first two attempts at this still left a brown lozenge in every
    crotch: a `bough` is at its darkest where it *starts* (mei's 新枝用浓墨
    ramp, `dens += 0.22*(1-t)`) and it starts with a round cap a full
    half-width across, so clipping the *centre* to the silhouette still
    buries half a disc of the child's heaviest ink inside the parent's
    wash.

    `lap` is where that near edge lands, in units of the child's own
    half-width: at 1.0 the child's near edge is exactly tangent to the
    parent's silhouette and the two masses only touch; below that they
    overlap, and the overlap shows up — multiply — as a crescent of double
    ink cut off by the child's round cap. A *hard arc* across a trunk is
    not 墨聚, it is a seam. 0.78 leaves a fifth of a cap's worth of darkening
    at the joint, which is as much as a brush would carry in with it. The
    one child this must never be applied to is the parent's own
    continuation past its tip (出梢): that is the same piece of wood, and
    clipped it comes away from the end cap as a separate sausage.
    """
    n = len(ppts)
    if n < 2:
        return pts, 0
    for q in range(len(pts) - 3):
        x, y = pts[q]
        dm, hm = 1e18, 1.0
        for j in range(n):
            d = (ppts[j][0] - x) ** 2 + (ppts[j][1] - y) ** 2
            if d < dm:
                dm, hm = d, phws[j]
        if dm > (hm + w0 * lap) ** 2:
            return pts[q:], q
    return pts, 0


# --------------------------------------------------------------------------
# 干 — the trunk, and why it is not mei's.
#
# 「画松之干，如老龙之蟠」. The difference between plum wood and pine wood is
# not a parameter, it is a *continuity*: plum changes direction only at a
# joint and is straight in between, pine changes direction everywhere and
# reverses two or three times over its length. Draw a pine with
# mei.wood_path's kinks and it comes out as a lightning bolt; draw it as one
# bezier — which is what the first draft did — and it comes out as a length
# of garden hose with bark on it, exactly the fault bamboo.py's culm comment
# warns about, because a single-signed curvature is the one thing nothing in
# nature has.
#
# So: a slow sinusoid in the *curvature*, with a couple of soft breaks laid
# on top of it where the old wood has been knocked about. The sinusoid is
# what makes it a dragon; the breaks are what keep it from being a spring.
# --------------------------------------------------------------------------

def trunk_path(rng, x, y, a0, L, segs=3, kink=0.18, serp=0.34, waves=1.6,
               grow=None, pull=0.0, cone=1.15, bounds=None, slack=0.07,
               step=None, want=None):
    """A piece of pine. Returns (points, joint indices)."""
    want = want if want is not None else [0]
    step = step or max(2.2, L / 60.0)
    n = max(6, int(L / step))
    want[0] = n
    cuts = set()
    if segs > 1 and n > 10:
        lo, hi = int(n * 0.18), int(n * 0.88)
        if hi - lo > segs:
            cuts = set(rng.sample(range(lo, hi), min(segs - 1, hi - lo)))
    ph = rng.uniform(0, TAU)
    # The serpentine is stated as a *total swing in radians* and then
    # differentiated, rather than as a per-step wobble that happens to add
    # up: a per-step amplitude gives a path whose bendiness depends on how
    # finely it was sampled, which is how the first version came out
    # straight at 380x540 and coiled at 1200x1700.
    dser = serp * waves * TAU / n
    pts = [(x, y)]
    joints = []
    ang = a0
    sgn = rng.choice((1.0, -1.0))
    # 立干出纸 (§5.2) puts a trunk's foot *below* the paper — `roots` spawns
    # it as far down as 1.08 H, which is outside this very margin. A path is
    # therefore allowed to begin outside and walk in; only a path that has
    # actually been inside can be ended by leaving. Without the latch, a root
    # that landed past 1.07 H broke the loop on its first step, returned a
    # single point, and `limb`'s `len(pts) < 4` guard silently abandoned the
    # whole tree: a sheet with a 题款, a seal and no pine on it.
    inside = False
    for i in range(1, n + 1):
        t = i / n
        if i in cuts:
            ang += sgn * kink * rng.uniform(0.55, 1.15)
            sgn = -sgn
            joints.append(len(pts) - 1)
        ang += dser * math.cos(t * waves * TAU + ph) + rng.uniform(-.012, .012)
        if pull and grow is not None:
            ang += _turn(ang, grow) * pull / n
        if cone is not None:
            if ang > a0 + cone:
                ang = a0 + cone
            elif ang < a0 - cone:
                ang = a0 - cone
        x += math.cos(ang) * step
        y += math.sin(ang) * step
        if bounds is not None:
            W, H = bounds
            if (-slack * W < x < (1.0 + slack) * W
                    and -slack * H < y < (1.0 + slack) * H):
                inside = True
            elif inside:
                break
        pts.append((x, y))
    return pts, joints


def scales(cv, rng, pts, hws, air, k=1.0, dens=1.0, old=1.0, lo=0.04, hi=0.96):
    """皮如龙鳞 — the bark, drawn as rows of staggered arcs.

    This is the mark that says "pine" on a sheet where every other mark
    could be some other tree, and it took one correction to get there: the
    first version scattered the arcs at random positions, which reads as
    cracked mud. Bark scales are *courses*, like brickwork — a row across
    the trunk, then the next row half a scale over — and the eye picks up
    the courses long before it picks up any single arc.

    Each arc lies with its chord across the grain and its belly toward the
    root, so it wraps the trunk; the ones near the silhouette are drawn
    shorter and flatter, which is the whole of the foreshortening and costs
    one cosine.
    """
    m = len(pts) - 1
    if m < 3 or dens <= 0.0:
        return
    # A scale has a *size*, and it is not a fraction of the trunk. The first
    # version fitted its courses with `nsc = width / (0.78·hw)`, which is two
    # scales across whatever the trunk measures: on a 40-pixel trunk that is
    # two 20-pixel arcs per course, and a stack of those is not bark, it is
    # the segments of a caterpillar — which is exactly what the first sheets
    # grew up the middle of every pine. 龙鳞 is a few millimetres of tree
    # whatever the tree's girth.
    # ...and that cuts both ways. Sizing the scale across the trunk in
    # pixels while *counting* the courses up it — a fixed `rows` over the
    # whole length — spaces the courses by length/rows, which on a long
    # trunk is three times the scale's own height: the bark comes out as
    # separated hoops, a stack of bands, a bamboo. A course is one scale
    # tall. So the row count is geometry, not a parameter, and what the
    # caller controls is only how much of the tube gets covered.
    unit = max(3.6, min(cv.w, cv.h) * 0.013)
    L = 0.0
    for q in range(m):
        L += math.hypot(pts[q + 1][0] - pts[q][0], pts[q + 1][1] - pts[q][1])
    rows = max(1, min(300, int(L * (hi - lo) / (unit * 0.78))))
    step = (hi - lo) / rows
    phase = 0.0
    for r in range(rows):
        if dens < 1.0 and rng.random() > dens:
            continue
        t = lo + (hi - lo) * (r + 0.5) / rows
        i = max(1, min(m - 1, int(t * m)))
        hw = hws[i]
        if hw < 2.2:
            continue
        ta = gb.tangent(pts, i)
        # how many scales fit across this course, and where they start
        wide = hw * 2.0
        nsc = max(2, min(7, int(wide / unit)))
        # 错缝 — courses stagger; but a perfect half-scale offset every
        # course is roof tiling, and a trunk tiled edge to edge is a
        # knitted sweater. 画鳞不可匀: the courses drift, and most of the
        # scales in any one of them are simply not written. What the eye
        # wants is enough of the pattern to infer the rest.
        phase = 0.5 - phase + rng.uniform(-0.22, 0.22)
        for s in range(nsc):
            u = ((s + 0.5 + phase) / nsc) * 2.0 - 1.0
            if abs(u) > 0.94 or rng.random() < 0.34:
                continue
            # each scale rides a little above or below its course, or the
            # courses read as ruled lines
            i = max(1, min(m - 1, int((t + rng.uniform(-.36, .36) * step) * m)))
            hw = hws[i]
            ta = gb.tangent(pts, i)
            # 近宽远窄: a scale seen near the edge of the cylinder is
            # compressed across and so is its belly.
            fore = math.sqrt(max(0.06, 1.0 - u * u))
            half = hw / nsc * rng.uniform(0.66, 1.30) * (0.45 + 0.75 * fore)
            # 鳞要深 — a scale whose belly is half its chord is a *dash*, and
            # a pale tube with rows of dark dashes on it is a birch, which
            # is what the 册页 came back as. The arc has to be at least as
            # deep as it is wide before the eye reads it as something
            # standing off the trunk.
            belly = half * rng.uniform(0.85, 1.65)
            cx, cy = _perp(pts, i, u * hw)
            nx, ny = -math.sin(ta), math.cos(ta)
            bx, by = -math.cos(ta) * belly, -math.sin(ta) * belly
            a = (cx - nx * half, cy - ny * half)
            b = (cx + bx, cy + by)
            c = (cx + nx * half, cy + ny * half)
            g = rng.random() ** 1.5
            drag(cv, _arc((a, b, c), 6),
                 0.52 + 0.76 * g * fore, 0.34 + 0.44 * g,
                 I("zhong"), I("jiao"), (0.14 + 0.40 * g) * k * (0.5 + 0.5 * old),
                 d0=0.40, d1=0.84, feather=0.70, core=0.18,
                 dry=0.30 + air["dry"], dry_ph=rng.uniform(0, TAU))
    # 擦 — a few long dry rubs with the grain, to keep the courses from
    # reading as a net. Same lesson as mei's §bark: short marks alone leave
    # the unbroken length of the tube intact, and the length is the tube.
    for _ in range(max(2, int(rows * 0.10 * dens))):
        i = rng.randrange(1, max(2, int(m * 0.84)))
        if hws[i] < 3.0:
            continue
        j = min(m, i + rng.randint(int(m * 0.06) + 2, int(m * 0.26) + 4))
        if j - i < 3:
            continue
        d = rng.uniform(-0.78, 0.78)
        seg = [_perp(pts, q, d * hws[q] * rng.uniform(0.94, 1.06))
               for q in range(i, j + 1)]
        drag(cv, seg, rng.uniform(0.58, 1.05), rng.uniform(0.40, 0.72),
             I("zhong"), I("nong"), rng.uniform(0.20, 0.44) * k,
             d0=0.30, d1=0.78, feather=0.80, core=0.20,
             dry=0.60 + air["dry"], dry_k=3.4, dry_ph=rng.uniform(0, TAU))


def moss(cv, rng, pts, hws, k=1.0, n=5):
    """点苔 — clusters of dark dots where the wood turns and where it is
    wettest. Never evenly spread, and never on young wood."""
    m = len(pts) - 1
    if m < 3:
        return
    for _ in range(n):
        i = rng.randrange(1, m)
        hw = hws[i]
        if hw < 1.6:
            continue
        cx, cy = _perp(pts, i, rng.uniform(-0.85, 0.85) * hw)
        for _ in range(rng.randint(2, 6)):
            disc(cv, cx + rng.gauss(0, hw * 0.42), cy + rng.gauss(0, hw * 0.5),
                 rng.uniform(0.7, 1.8), I("jiao"), rng.uniform(0.28, 0.68) * k,
                 soft=0.4)


def claw(cv, rng, pts, w, st, air, k=1.0, up=-math.pi * 0.5):
    """蟹爪 — the hook a pine twig ends in.

    「枝如蟹爪」 is the second half of the manual's description and it is the
    detail that stops a twig being a line that stopped. A pine's last inch
    turns hard, almost always upward, and it is written in one quick 浓墨
    flick that is thinner than the twig it leaves.
    """
    if len(pts) < 3:
        return
    x, y = pts[-1]
    ta = gb.tangent(pts, len(pts) - 1)
    L = max(3.0, w * rng.uniform(3.0, 5.5))
    turn = _turn(ta, up)
    a1 = ta + max(-1.5, min(1.5, turn)) * rng.uniform(0.55, 0.95)
    mid = (x + math.cos(ta) * L * 0.55, y + math.sin(ta) * L * 0.55)
    end = (mid[0] + math.cos(a1) * L * 0.62, mid[1] + math.sin(a1) * L * 0.62)
    drag(cv, _arc(((x, y), mid, end), 7), max(0.5, w * 0.78), 0.34,
         rgb(st["wood"][0]), rgb(st["wood"][1]), 0.92 * k,
         d0=0.78, d1=0.95, feather=0.75, core=0.22, dry=0.25 + air["dry"])


def cones(cv, rng, pts, hws, st, k=1.0, n=2):
    """松球 — one or two cones hanging where a twig forks. Small on purpose:
    like the grass on orchid's cliff, they are there to give the needles a
    scale to be large against."""
    m = len(pts) - 1
    if m < 4:
        return
    ink = rgb(st["cone"])
    for _ in range(n):
        i = rng.randrange(max(1, m // 3), m)
        r = max(1.6, hws[i] * rng.uniform(1.5, 2.4))
        cx, cy = _perp(pts, i, rng.uniform(-1.0, 1.0) * hws[i])
        cy += r * 0.5
        for q in range(9):
            a = q * TAU / 9 + rng.uniform(0, 0.3)
            disc(cv, cx + math.cos(a) * r * 0.52,
                 cy + math.sin(a) * r * 0.78 * 0.52,
                 r * rng.uniform(0.30, 0.46), ink, rng.uniform(0.40, 0.72),
                 soft=0.35)
        disc(cv, cx, cy, r * 0.46, ink, 0.55 * k, soft=0.5, squash=0.74)


# --------------------------------------------------------------------------
# 针 — the needles. 「松叶如针，攒三聚五」.
#
# One bundle is a splay of straight strokes from a single point. The three
# things that decide whether it reads as pine or as a dandelion clock:
#
#   * **the strokes are straight and they are written outward.** Dark and
#     broad where they leave the shoot, thin and pale at the point, because
#     that is what a brush does when it is pushed out and lifted. Draw them
#     the other way and every bundle grows a bead at its centre.
#   * **the tips land on an arc, not on a circle of identical radius.** A
#     bundle whose needles are all exactly `R` long is a cog.
#   * **the splay is less than you think.** 扇形 spans about a radian and a
#     half. Only the 轮形 bundle — the one at the very end of a shoot seen
#     end-on — goes all the way round, and a sheet with nothing but those
#     is the wheel-diagram failure the plates below exist to prevent.
# --------------------------------------------------------------------------

def fan(cv, rng, x, y, a0, span, R, lo, hi, n=11, k=1.0, dry=0.3, wob=0.10,
        alpha=1.0):
    """One 攒针 bundle: `n` needles from (x, y), centred on `a0`."""
    if R < 2.0 or n < 2:
        return
    for j in range(n):
        t = (j + 0.5) / n
        a = a0 + (t - 0.5) * span + rng.gauss(0.0, span * 0.05)
        # 长短参差 — and the variation has to be in the *length*, not in the
        # start: needles share a base and differ at the point.
        r = R * rng.uniform(0.76, 1.06)
        r0 = R * rng.uniform(0.02, 0.10)
        bx, by = x + math.cos(a) * r0, y + math.sin(a) * r0
        ex, ey = x + math.cos(a) * r, y + math.sin(a) * r
        # one consistent bow per needle, alternating, so the bundle splays
        # like hair rather than fanning like a compass
        c = wob * r * (1.0 if j % 2 else -1.0) * rng.uniform(0.4, 1.3)
        mx_ = (bx + ex) * 0.5 - math.sin(a) * c
        my_ = (by + ey) * 0.5 + math.cos(a) * c
        # 针要细 — and the width is in *pixels*, only mildly sloped in the
        # plate's size. Scaled purely off `R` (the first version did) a
        # large plate came out drawn with grass blades and a small one with
        # hairs: a needle is a needle whatever the tree is doing.
        g = 0.74 + 0.5 * rng.random()
        wid = (0.46 + R * 0.016) * g
        drag(cv, _arc(((bx, by), (mx_, my_), (ex, ey)), 4),
             wid, max(0.20, wid * 0.42),
             lo, hi, alpha * (0.62 + 0.42 * rng.random()) * k,
             d0=0.92, d1=0.42, feather=0.68, core=0.24,
             dry=dry, dry_k=3.0, dry_ph=rng.uniform(0, TAU))


def plate(cv, rng, res, cx, cy, R, ang, st, air, k=1.0, tone=1.0, lift=0.0,
          wind=0.0, wind_a=0.0, snow=0.0, wheel=0.30, twig=None):
    """一团松针 — a plate of needle, built out of three to six bundles.

    The 册页 lesson from crab and from mei, applied before it could be
    earned the hard way: *the failure mode of a repeated mark is that every
    instance is the same instance*. A plate therefore varies in how many
    bundles it holds, where their origins sit, how wide each splays, how
    far each reaches and which way the whole thing faces — and, most of
    all, in **tone**, because 浓淡 is what separates one plate from the
    plate behind it.

    `lift` is mei's Reserve put to a use mei had no need for. Under a
    multiply composite a near plate cannot hide a far one: two lattices of
    strokes over each other give a third, denser lattice, and a crown built
    that way is a flat grey cloud with no interior. So a front plate first
    *takes the paper back* in a soft irregular blob under itself. That is
    not a trick; it is 留白 — the painter leaves the gap, and the gap is
    what makes the plate in front sit in front.
    """
    if R < 3.0:
        return
    nlo, nhi = rgb(st["needle"][0]), rgb(st["needle"][1])
    # 三墨: the back plates are 淡, the front ones 浓, and the pale ones are
    # pale in *both* stops — thinning only the core leaves a hard dark rim.
    nlo = dilute(nlo, 0.30 + 0.70 * tone)
    nhi = dilute(nhi, 0.22 + 0.78 * tone)
    if lift > 0.0 and res is not None:
        # ...and it has to stay *inside* the needle, which is the whole
        # difficulty: a lift as wide as the plate shows as a bleached disc
        # wherever the needle happens to be thin. Half the radius, a soft
        # edge, and the same flattening as the bundle field, so the paper
        # comes back only under the body of the 团.
        # 针不离枝 — and the lift is where that rule was being broken. The
        # plate is set on the end of a twig, so the twig's last stretch runs
        # *under* the plate; a lift centred on the plate and round takes the
        # paper back over that stretch as well, and the wood comes out of
        # the branch, stops in clean paper, and the needle mass floats a
        # finger's width beyond it. The blob is therefore cut back on the
        # twig side — θ = π in the lift's own rotated frame, because `rot`
        # is the twig's direction and the plate was pushed out along it —
        # so the paper comes back under the *front* of the 团 only and the
        # wood keeps its ink all the way in.
        # The size matters as much as the shape. At half a radius with a
        # feather of nearly another half, a plate's lift reached a long way
        # past its own rim — and the plates are packed at `sep=0.56`, so
        # what it mostly bleached was the *neighbours'* twigs. That is the
        # whole of why the crown read as needles floating free of the tree:
        # the wood was drawn, and then twenty plates rubbed it out.
        pr = [R * (0.30 + 0.15 * rng.random()) for _ in range(7)]
        ph = rng.uniform(0, TAU)

        def rad(th, _pr=pr, _ph=ph):
            u = ((th + _ph) % TAU) / TAU * 7.0
            i = int(u)
            f = u - i
            r = _pr[i % 7] + (_pr[(i + 1) % 7] - _pr[i % 7]) * f
            return r * (1.0 - 0.52 * max(0.0, -math.cos(th)) ** 0.7)

        res.lift(cv, cx, cy, rad, alpha=lift * 0.46, feather=R * 0.30,
                 squash=rng.uniform(0.62, 0.80), rot=ang)
    if twig is not None:
        # 枝入针中 — a stub of wood run from the twig's end into the body of
        # the 团, after the lift and before the needles. Three pixels of
        # branch is all it takes: the eye wants to see the wood *enter* the
        # mass, and a plate whose twig stops at its rim reads as hung on a
        # nail however close it is.
        tx, ty = twig
        # ...and no further than the middle of it. Run on past that and the
        # stub comes out the other side as a bare brown bar lying across a
        # mass the lift has just paled — the twig stops being the thing the
        # needles hang from and becomes a stick someone pushed through them.
        ex = cx + math.cos(ang) * R * rng.uniform(-0.04, 0.09)
        ey = cy + math.sin(ang) * R * rng.uniform(-0.04, 0.09)
        wd = max(0.7, R * 0.042)
        drag(cv, _arc(((tx, ty), ((tx + ex) * 0.5 + rng.gauss(0, R * 0.03),
                                  (ty + ey) * 0.5 + rng.gauss(0, R * 0.03)),
                      (ex, ey)), 6),
             wd, wd * 0.52, rgb(st["wood"][0]), rgb(st["wood"][1]),
             0.62 * k * (0.6 + 0.4 * tone), d0=0.62, d1=0.26,
             feather=0.70, core=0.26, dry=0.24 + air["dry"] * 0.6,
             dry_k=2.6, dry_ph=rng.uniform(0, TAU))
    # 攒针宜密 — and this is the number that decided whether the module
    # worked. The first version put three to six bundles of seven to
    # thirteen needles into a plate, which is forty strokes spread over a
    # disc eighty pixels across: at that density a bundle is a *star*, the
    # eye counts its arms, and a crown of them is a page of asterisks. The
    # mark is called 攒 — gathered — and the gathering is the mark. Five to
    # nine bundles of sixteen to twenty-six needles, each bundle reaching
    # barely half the plate, puts a hundred and fifty strokes in the same
    # disc; no arm is counted, and what the eye gets is a *mass* with a
    # ragged edge, which is what a pine's 团 is.
    # 团 — and the bundles are scattered *over the plate*, each facing out
    # of it, not strung along the shoot each facing the same way. That was
    # the third draft's fault and it is a subtler one than the density:
    # with every fan aimed at `ang ± a bit`, forty of the hundred and fifty
    # needles lie on top of another needle and the only ones the eye can
    # separate are the ones at the rim — so the plate reads as a thistle,
    # a ring of long spikes round an empty middle, however many strokes
    # went into it. A pine's 团 is a *body* of needle: the bundles sit all
    # over its section and each one points away from the middle, which
    # fills the interior with crossings and leaves the rim soft.
    #
    # `out` is how radial that is. At 1 the plate is a rosette; at 0 it is
    # a comb. The blend is what makes a plate hanging off the end of a
    # shoot different from one sitting astride it.
    # 针短而团大 — the fourth draft's fault, and the last one. Scattering the
    # origins and aiming them outward was right; giving each bundle a reach
    # of *half the plate* was not. A fan that long puts its base in the
    # middle of the 团 and its point outside it, so every bundle is a star
    # drawn over the same centre and the plate is a thistle again however
    # the origins move. The relation that matters is **fan reach : plate
    # radius**, and in a painted pine it is about a quarter: the 团 is a
    # body of many *small* gatherings, the way a cloud is a body of small
    # curls. Twenty-odd fans of a dozen short needles over the whole disc,
    # each with its own tone, and the mass closes.
    # 风不散团 — and the count has to answer the wind. A combed bundle has
    # its splay closed to two-thirds, so it covers two-thirds of the paper
    # it covered in still air; at the same count the 团 stops being a mass
    # and becomes a scatter of darts, and the ones that land near the rim
    # read as needles blown loose with no twig under them — which is
    # exactly how the 松涛 sheets failed. Wind combs a plate flat; it does
    # not thin it.
    # 团要实 — and the count above is a *coverage*, not a count: nb·fr² is
    # held at a fixed multiple of R², so a big plate and a small one are
    # equally dense. The multiple is what took three rounds to find. A fan
    # is not a filled disc: it is a sector of about a quarter turn with a
    # dozen hairlines in it, so it covers something like a quarter of the
    # fr-disc it occupies. At 2.3 the real coverage was therefore around a
    # half — the 团 had paper showing between its bundles, every fan was
    # separately countable, and a fan near the rim with its neighbours a
    # bundle-width away read as a loose asterisk floating in the sky with
    # nothing holding it. 松针 is drawn 攒 — gathered — and a gathering is
    # not a gathering at half cover.
    unit = max(2.6, min(cv.w, cv.h) * 0.006)
    fr = max(unit * 1.6, R * rng.uniform(0.24, 0.33))
    nb = max(8, min(78, int(R * R * 3.40 / (fr * fr) * rng.uniform(.85, 1.15)
                            / max(0.58, 1.0 - 0.42 * wind))))
    sx, sy = math.cos(ang), math.sin(ang)
    for b in range(nb):
        # origins over an ellipse that is longer along the shoot, and out to
        # most of the rim — the rim bundles are what gives the 团 its edge
        pa = rng.uniform(0, TAU)
        # ...and not at even density over the disc. `sqrt` spreads the
        # origins uniformly by area, which puts most of them in the outer
        # ring — a 团 with a thin middle and a spattered edge. A needle
        # mass is heaviest at its body and ragged only at its rim.
        # And the origin field has to be inset by a *whole* fan, not half of
        # one. At 0.55 an origin could sit at 0.85 R, the 1.18 stretch along
        # the shoot carried it to the rim, and its fan then reached a third
        # of a radius beyond that: the outermost needles of a plate landed
        # at 1.4 R, a clear bundle-width outside the body, with the twig
        # stub and the rest of the mass nowhere near them. Those strays are
        # the asterisks floating off the crown. Inset by the fan's own reach
        # and the plate ends where `R` says it ends.
        pd = (R - fr * 0.95) * rng.random() ** 0.80
        lx, ly = math.cos(pa) * pd * 1.06, math.sin(pa) * pd * 0.72
        ox = cx + sx * lx - sy * ly
        oy = cy + sy * lx + sx * ly
        if rng.random() < wheel * 0.5:
            span, ba = TAU * rng.uniform(0.70, 1.0), rng.uniform(0, TAU)
            nn = rng.randint(11, 16)
        else:
            span = rng.uniform(1.10, 2.10)
            rad_a = math.atan2(oy - cy, ox - cx) if pd > R * 0.10 else ang
            out = 0.60 + 0.4 * rng.random()
            ba = ang + _turn(ang, rad_a) * out + rng.gauss(0.0, 0.40)
            nn = rng.randint(8, 13)
        if wind > 0.0:
            # 风 — the plate is combed: every needle swings toward the wind
            # and the splay closes up. This is the one place in the module
            # where the air changes a shape rather than a value.
            ba += _turn(ba, wind_a) * wind * rng.uniform(0.55, 0.95)
            span *= 1.0 - 0.42 * wind
        # 一团之中亦分浓淡 — per-bundle tone, so the interior of the mass has
        # layers in it instead of one even grey
        bt = rng.random() ** 1.5
        fan(cv, rng, ox, oy, ba, span, fr * rng.uniform(0.80, 1.22),
            dilute(nlo, 0.52 + 0.48 * bt), dilute(nhi, 0.44 + 0.56 * bt),
            n=nn, k=k * (0.74 + 0.40 * tone) * (0.70 + 0.50 * bt),
            dry=0.18 + air["dry"] * 0.7,
            wob=rng.uniform(0.05, 0.16))
    if snow > 0.0 and res is not None:
        # 雪压松 — snow does not sit *on* a needle plate, it sits in the
        # dips of its upper surface, and in ink painting it is the paper.
        for _ in range(rng.randint(2, 4)):
            a = -math.pi * 0.5 + rng.uniform(-0.9, 0.9)
            rr = R * rng.uniform(0.26, 0.46)
            res.lift(cv, cx + math.cos(a) * R * rng.uniform(0.3, 0.8),
                     cy + math.sin(a) * R * rng.uniform(0.3, 0.8),
                     lambda th, _r=rr: _r * (0.78 + 0.3 * math.sin(th * 3)),
                     alpha=0.62 * snow, feather=rr * 0.5, squash=0.52)


def _infront(res, cv, pts, hws, grow=1.14, pad=1.1, feather=1.5):
    """Put a snapshot back inside the band swept by `pts` at `hws`.

    `Reserve.lift` restores the paper inside a *polar* outline, which is the
    right shape for a flower, a cloud or a plate of needle and the wrong one
    for a trunk: a pine runs two thirds of the way down the sheet and
    reverses twice on the way, and no star-shaped blob covers that. This is
    the same restore done along a stroke instead — one capsule per segment,
    at the stroke's own half-width grown a little.

    What it is *for* is the one thing a multiply composite cannot do by
    itself. 松石图 means the two masses touch, and where they touch one of
    them is in front; under multiply neither is, and the trunk and the rock
    simply show through each other like two panes of smoked glass. So the
    rock is laid down over the finished wood against a second snapshot, and
    then the wood is lifted back out of it. `grow` and `pad` make the band
    slightly wider than the wood, which leaves a hair of bare paper between
    the trunk's edge and the rock's wash — and that gap is not a cheat to
    hide a seam, it is what the hand does: 石不碍树, the rock's 皴 stops
    short of the trunk and the paper between them is the drawing.

    Restoring with `alpha` 1 is idempotent, so segments may overlap freely
    and the joints cost nothing.
    """
    base, buf, cw, ch = res.base, cv.buf, cv.w, cv.h
    inv_f = 1.0 / max(0.2, feather)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        r0 = hws[i] * grow + pad
        r1 = hws[i + 1] * grow + pad
        rm = max(r0, r1) + feather + 1.0
        px0 = max(0, int(min(ax, bx) - rm))
        px1 = min(cw, int(max(ax, bx) + rm) + 1)
        py0 = max(0, int(min(ay, by) - rm))
        py1 = min(ch, int(max(ay, by) + rm) + 1)
        dx, dy = bx - ax, by - ay
        ll = dx * dx + dy * dy or 1.0
        for py in range(py0, py1):
            fy = py + 0.5 - ay
            row = py * cw
            for px in range(px0, px1):
                fx = px + 0.5 - ax
                t = (fx * dx + fy * dy) / ll
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                ex, ey = fx - dx * t, fy - dy * t
                a = (r0 + (r1 - r0) * t
                     - math.sqrt(ex * ex + ey * ey)) * inv_f + 0.5
                if a <= 0.004:
                    continue
                if a > 1.0:
                    a = 1.0
                j = (row + px) * 3
                o = buf[j]
                buf[j] = int(o + (base[j] - o) * a)
                o = buf[j + 1]
                buf[j + 1] = int(o + (base[j + 1] - o) * a)
                o = buf[j + 2]
                buf[j + 2] = int(o + (base[j + 2] - o) * a)


def cloud(cv, rng, res, y, h, W, alpha=0.62, n=9):
    """云锁松腰 — a band of cloud across the trunk.

    White is the identity of a multiply composite, so a cloud cannot be
    painted; it can only be *un*painted. A row of overlapping lifts across
    the sheet takes the paper back wherever it passes, which hides the
    middle of the trunk and leaves the crown floating — the oldest device
    in 松石 painting and, here, nine lines of arithmetic.
    """
    if res is None:
        return
    for i in range(n):
        t = (i + 0.5) / n
        cx = W * (t * 1.24 - 0.12) + rng.gauss(0, W * 0.02)
        cy = y + math.sin(t * TAU * rng.uniform(0.8, 1.4)) * h * 0.34
        rr = h * rng.uniform(0.55, 1.05)
        res.lift(cv, cx, cy,
                 lambda th, _r=rr, _p=rng.uniform(0, TAU):
                 _r * (1.0 + 0.22 * math.sin(th * 2 + _p)
                       + 0.10 * math.sin(th * 5 - _p)),
                 alpha=alpha * rng.uniform(0.74, 1.0),
                 feather=rr * 0.55, squash=rng.uniform(0.26, 0.42))


def turf(cv, rng, x, y, w, st, k=0.6, n=14):
    """坡草 — a few blades of grass at the foot, to seat the tree on ground
    instead of on the bottom border."""
    lo, hi = dilute(rgb(st["wood"][0]), 0.5), dilute(rgb(st["wood"][1]), 0.7)
    for _ in range(n):
        bx = x + rng.gauss(0, w * 0.5)
        by = y + rng.uniform(-w * 0.03, w * 0.05)
        L = w * rng.uniform(0.06, 0.20)
        a = -math.pi * 0.5 + rng.gauss(0, 0.5)
        ex = bx + math.cos(a) * L + rng.gauss(0, L * 0.25)
        ey = by + math.sin(a) * L
        drag(cv, _arc(((bx, by), ((bx + ex) * 0.5 + rng.gauss(0, L * 0.2),
                                  (by + ey) * 0.5), (ex, ey)), 5),
             rng.uniform(0.6, 1.3), 0.3, lo, hi, rng.uniform(0.25, 0.6) * k,
             d0=0.80, d1=0.40, feather=0.8, dry=0.4)


# --------------------------------------------------------------------------
# 分枝 — the skeleton.
#
# 松之枝: 「下不生枝，枝必横出，梢必上仰」. Three rules, and all three are
# about *angle*, which is the only thing that distinguishes a pine skeleton
# from a plum one at a distance:
#
#   1. nothing branches from the lower third — the bare trunk is half of
#      what makes a pine look tall;
#   2. a 大枝 leaves the trunk at very nearly a right angle and runs out
#      flat, so the crown is an umbrella (偃盖) and not a cone;
#   3. every twig turns up at its end, and ends in a 蟹爪.
#
# The recursion carries an environment dict rather than eighteen arguments.
# This is the pavilion lesson in the other direction: `paint` there is five
# hundred lines of straight-line code whose locals *are* the painting, and
# the one bug in it came from a name collision. Here the state travels in
# one named box, so a child cannot shadow its parent by accident.
# --------------------------------------------------------------------------

def _plan_kids(E, pts, hws, L, depth):
    """Decide every branch that will leave this piece of wood.

    Planning is split out from drawing for one reason: the parent's own
    contour has to know where its children are going before it is drawn
    (出枝处断线). Returns a list of dicts, one per child, carrying where it
    leaves (`t`), which flank (`sgn`), how wide a hole it makes in the
    parent's silhouette (`hw`, in pixels along the parent) and the
    arguments `limb` will need.
    """
    rng, sc = E["rng"], E["sc"]
    if depth >= E["maxd"]:
        return []
    if depth == 0:
        nch, (lo, hi), sep = rng.randint(*sc["limbs"]), sc["span"], 0.16
    else:
        nch = rng.randint(*sc["whips"]) + (1 if depth == 1 else 0)
        lo, hi, sep = 0.34, 0.96, 0.18
    out = []
    sgn = rng.choice((1.0, -1.0))
    for ti in _spread(rng, nch, lo, hi, sep):
        bx, by, ta = _at(pts, ti)
        # 枝必横出 — a 大枝 leaves the trunk at close to a right angle; a
        # twig leaves its limb at a much smaller one, or the crown turns
        # into a starburst. And they alternate: 忌对节 all the way down.
        off = rng.uniform(1.12, 1.54) if depth == 0 else rng.uniform(0.40,
                                                                     0.84)
        a1 = ta + sgn * off
        if E["hang"] > 0.0 and depth == 0:
            # 悬崖松 — on a cliff the limbs on the void side drop.
            a1 += E["hang"] * rng.uniform(0.3, 0.9) * (1.0 if math.cos(a1) *
                                                       E["out"] > 0 else -0.2)
        cw = max(0.5, E["hwat"](hws, ti) * rng.uniform(0.44, 0.62))
        out.append(dict(
            t=ti, sgn=sgn, x=bx, y=by, a=a1, w=cw,
            # the hole the branch cuts in the parent's silhouette is its own
            # width projected along the parent — wider the more obliquely it
            # leaves — and a touch narrower than that, so the two lines come
            # close enough to read as one piece of wood.
            hw=cw * 0.92 / max(0.42, abs(math.sin(off))),
            # 梢必上仰 — the further out, the harder the wood turns up.
            pull=(0.45 + 0.30 * depth) * rng.uniform(0.7, 1.3),
            L=L * (rng.uniform(0.48, 0.68) if depth == 0
                   else rng.uniform(0.36, 0.54)) * (1.0 - 0.18 * ti)))
        sgn = -sgn
    return out


def _flat(E, a):
    """偃盖 — the growth attractor for one bough of a flat-crowned pine.

    The whole scene is this function. A pine's wood bends, over its length,
    toward one angle; give every bough in the tree the *same* angle and a
    flat-crowned pine comes out as a tree blown sideways — three drafts of
    it, each with the trunk laid down as a diagonal beam and the entire
    crown heaped against one border. A 偃盖 is not asymmetric: 「偃盖如伞」,
    an umbrella, and the boughs go out level on *both* sides. So the
    attractor is per-bough and it takes its quarter from the bough's own
    heading — flat, a fifth of a radian above the horizon, in whichever
    direction the wood already set off. Returns None when the scene is not
    偃盖, which means "use the tree's own attractor".
    """
    if not E["flat"]:
        return None
    return -0.20 if math.cos(a) >= 0.0 else math.pi + 0.20


def limb(E, x, y, a0, L, w0, depth, pull=0.6, side=None, par=None, grow=None):
    """Draw one piece of wood and everything that grows off it.

    `par` is the parent's `(points, half-widths)` — the child needs it to
    cut itself back to the parent's silhouette (`_emerge`). `grow` overrides
    the tree's growth attractor for this piece alone; see `_flat`.
    """
    cv, rng, st, air, sc = E["cv"], E["rng"], E["st"], E["air"], E["sc"]
    S, k, old = E["S"], E["k"], E["old"]
    # The trunk and the branches do not answer to the same attractor in
    # 偃盖: pulling the *stem* flat as well laid the whole tree down as one
    # diagonal beam from corner to corner, which is a fallen pine and not a
    # canopy. The stem keeps the tree's own attractor — it is called with
    # `grow=None` — and the wood that leaves it runs out level.
    up = E["up"] if grow is None else grow
    if L < S * 0.03 or w0 < 0.35:
        return
    segs = rng.randint(2, 4) if depth == 0 else rng.randint(2, 4)
    serp = (0.28 if depth == 0 else 0.26) * rng.uniform(0.7, 1.35)
    pts, joints = trunk_path(
        rng, x, y, a0, L, segs=segs,
        kink=sc["kink"] * (1.0 + 0.5 * depth) * rng.uniform(0.7, 1.3),
        serp=serp, waves=rng.uniform(1.0, 1.8), grow=up, pull=pull,
        cone=1.25 - 0.2 * depth, bounds=(cv.w, cv.h),
        step=max(2.2, L / (54.0 - 8.0 * depth)))
    if len(pts) < 4:
        return
    if par is not None:
        pts, _cut = _emerge(pts, par[0], par[1], w0)
        if len(pts) < 4:
            return
    w1 = max(0.46, w0 * (rng.uniform(0.30, 0.44) if depth == 0
                         else rng.uniform(0.38, 0.56)))
    sd = side if side is not None else rng.uniform(-0.5, 0.5)
    # `bough`'s `old` is not this module's `old`. In mei it drives the
    # swing of the 提按 as well as the dryness, and at 0.9 the width
    # modulates by a third of itself — which on plum is 屈铁 and on pine is
    # a caterpillar. 松干 is *even*: the tree's age is told by its bark and
    # its twist, not by lumps in its silhouette. So the wood gets half the
    # age the bark gets, and the two numbers are deliberately not the same
    # number.
    # 双勾 — and this is the difference between a pine and a pipe. Plum is
    # painted 没骨, the branch *is* the loaded stroke; pine is drawn with
    # its two edges and the tube between them is left nearly paper, so the
    # bark and the 渲染 have somewhere to live. A solid grey cylinder with
    # scales printed on it is a drainpipe in a jumper. Below a certain
    # girth there is no inside to leave empty and the twig goes back to
    # being one stroke.
    # — and the changeover has to be a *ramp*, not a switch. It was a
    # switch, and the switch was the second half of the ugly joint: a
    # 6 px limb came out one solid 浓墨 stroke at alpha 0.96 while the
    # 8 px trunk it left came out a pale 双勾 tube at 0.56, so the two
    # pieces of one tree met at the fork as two different substances —
    # a dark club laid on a pale pipe. 浓淡相接 wants a step, not a cliff.
    tw = S * 0.011
    tf = clamp01((tw - w0) / (tw * 0.55))   # 0 = 双勾, 1 = one stroke
    thin = tf > 0.92
    hws = bough(cv, rng, pts, w0, w1, st, air,
                alpha=0.56 + 0.40 * tf * tf,
                old=old * (0.50 - 0.10 * depth), bone=0.5 + 0.4 * old,
                k=k, side=sd)
    # The band this piece of wood occupies, kept for whatever has to be
    # painted *behind* it later — see `_infront` and the rock.
    if w0 > S * 0.004:
        E["wood"].append((pts, hws))

    # ---- where the children will leave, decided *before* the contour is
    # ---- drawn, because 出枝处断线: the parent's own 双勾 line has to open
    # ---- where a branch goes through it. A continuous line across the
    # ---- base of a branch is the thing that makes the branch look stuck
    # ---- on rather than grown out, and it is one half of why every fork
    # ---- on the sheet read as a dark patch with a line through it.
    kids = _plan_kids(E, pts, hws, L, depth)

    if not thin:
        wlo, whi = rgb(st["wood"][0]), rgb(st["wood"][1])
        nn = len(pts)
        _cum, _Lp = _arclen(pts)
        for s in (-1.0, 1.0):
            gaps = [g for g in
                    ((kd["t"] - kd["hw"] / max(1.0, _Lp),
                      kd["t"] + kd["hw"] / max(1.0, _Lp))
                     for kd in kids if kd["sgn"] * s > 0)]
            # 干不可圆 — and the two sides are not the same line. Offsetting
            # each edge by its own per-point jitter gives high-frequency
            # fuzz, which at any distance averages back to a smooth tube:
            # the trunk comes out a length of hose. What makes wood is
            # *low*-frequency — a handful of slow swellings along the
            # length, different on each side, with the odd local knuckle.
            fq = [(rng.uniform(1.2, 4.5), rng.uniform(0, TAU),
                   rng.uniform(0.035, 0.085)) for _ in range(3)]
            kx, kw = rng.uniform(0.15, 0.85), rng.uniform(0.03, 0.08)
            ka = rng.uniform(0.06, 0.20) * (0.4 + 0.8 * old)
            # 一边重一边轻 — the two flanks of a trunk are never the same
            # weight. Drawn at one alpha they read as a mechanical outline,
            # which is the 「像钢笔画」 fault mei §3 spent a round on.
            al = (0.92 if s * (1.0 if sd >= 0 else -1.0) > 0 else 0.64) \
                * (1.0 - tf) ** 0.8
            runs, edge = [], []
            for q in range(nn):
                u = q / max(1, nn - 1)
                if any(a < u < b for a, b in gaps):
                    if len(edge) > 2:
                        runs.append((edge, u))
                    edge = []
                    continue
                g = 1.0
                for f, ph, am in fq:
                    g += am * math.sin(u * f * TAU + ph)
                g += ka * math.exp(-((u - kx) / kw) ** 2)
                edge.append(_perp(pts, q, s * hws[q] * g
                                  * rng.uniform(0.985, 1.015)))
            if len(edge) > 2:
                runs.append((edge, 1.0))
            for seg, u1 in runs:
                u0 = u1 - (len(seg) - 1) / max(1, nn - 1)
                drag(cv, seg,
                     1.25 + (w0 + (w1 - w0) * max(0.0, u0)) * 0.075,
                     0.90 + (w0 + (w1 - w0) * u1) * 0.075, wlo, whi,
                     al * rng.uniform(0.92, 1.06) * k,
                     d0=0.34, d1=0.80, feather=0.74, core=0.22,
                     dry=0.34 + air["dry"] * 0.8, dry_k=2.6,
                     dry_ph=rng.uniform(0, TAU))
    E["occ"].add_path(pts, w0 * 1.25, w1 * 1.25)
    res = E["res"]
    # 斑驳 first, bark second — `blot` lifts paper and would wipe out any
    # line already drawn inside the mass. (mei §blot; the same ordering
    # trap, inherited with the function.)
    if w0 > S * 0.008:
        blot(cv, rng, res, pts, hws, air, k=k, old=old, side=sd,
             n=int(0.5 + 1.2 * old + max(0.0, S / 820.0 - 1.0)))
    if w0 > S * 0.006:
        flyback(cv, rng, res, pts, hws, air, k=k, side=sd,
                n=int((2 + 4 * old + 2 * air["dry"]) * max(1.0, S / 820.0)))
    if w0 > S * 0.0055:
        scales(cv, rng, pts, hws, air, k=k, old=old,
               dens=min(0.86, (0.34 + 0.40 * old) * (1.0 - 0.26 * depth)))
        if old > 0.45 and depth < 2:
            moss(cv, rng, pts, hws, k=k, n=int(1 + 4 * old))
    if depth >= 1 and w0 < S * 0.010 and rng.random() < 0.30 * old:
        cones(cv, rng, pts, hws, st, k=k, n=rng.randint(1, 2))

    # ---- what comes off it -------------------------------------------
    if depth >= E["maxd"]:
        ta = gb.tangent(pts, len(pts) - 1)
        claw(cv, rng, pts, w1, st, air, k=k, up=up)
        E["tips"].append((pts[-1][0], pts[-1][1], ta, depth))
        return
    for kd in kids:
        limb(E, kd["x"], kd["y"], kd["a"], kd["L"], kd["w"],
             depth + 1, pull=kd["pull"], par=(pts, hws),
             grow=_flat(E, kd["a"]))
    # 出梢 — the parent's own end goes on as the finest twig it has, and
    # forks once. A trunk that simply tapers to a point in open paper reads
    # as the end of a drawn line; wood ends by dividing.
    # 不可对生 — and the two of them must not leave from the *same* point.
    # They did, and because a `bough` carries its heaviest ink at its base
    # and ends in a round cap, two bases on one spot laid a symmetrical
    # dark teardrop into the crotch of every terminal fork: the single
    # most conspicuous blemish on the sheet, and a thing no hand would
    # ever write. The second one steps back down the parent.
    # The first of the two is the parent *continuing*, not a branch leaving
    # it, so it is the one child that must not be cut back to the parent's
    # silhouette: clipped, it started a half-width past the blunt end cap
    # and the wood came apart into sausages with daylight between them.
    for q in range(2 if depth < E["maxd"] - 1 else 1):
        ex, ey, ta = _at(pts, 1.0 if q == 0 else rng.uniform(0.84, 0.93))
        a1 = ta + rng.uniform(0.18, 0.62) * (1.0 if q else -1.0)
        limb(E, ex, ey, a1, L * rng.uniform(0.30, 0.46),
             max(0.5, w1 * rng.uniform(0.62, 0.86)), depth + 1,
             pull=0.5 + 0.3 * depth, par=None if q == 0 else (pts, hws),
             grow=_flat(E, a1))


def _hwat(hws, t):
    """The half-width a fraction `t` along a stroke."""
    if not hws:
        return 1.0
    f = clamp01(t) * (len(hws) - 1)
    i = min(len(hws) - 2, int(f))
    return hws[i] + (hws[i + 1] - hws[i]) * (f - i)


def tree(E, x, y, a0, L, w0, plates, taken):
    """One pine: the wood, then the needle plates that hang off its tips.

    The two passes are *not* interleaved, and that is the point. Needles
    are laid over finished wood so that a plate's `lift` can take back the
    branch behind it; and the plates are sorted by depth before any of them
    is drawn, so the sheet is painted back to front the way the hand does
    it — 淡墨 for the far plates, 浓墨 for the near, and each near one
    biting a hole in whatever it stands in front of.
    """
    cv, rng, sc, S = E["cv"], E["rng"], E["sc"], E["S"]
    E["tips"] = []
    limb(E, x, y, a0, L, w0, 0, pull=0.22 + 0.3 * rng.random())
    tips = E["tips"]
    if not tips:
        return taken, []
    R = S * rng.uniform(*sc["plate"]) * E["scale"]
    # 疏密 — but a pine's plates *touch*. `sep` is the fraction of the two
    # radii two groups must keep between them, and bamboo's 0.95 is right
    # for leaf groups on separate culms and much too much here: at 0.86 the
    # crown came out as a dozen separate pompoms floating on bare twigs.
    # Pine needle masses pile into layers, and a layer is made of plates
    # that overlap.
    sites, taken = cluster_sites(rng, tips, plates, R, sep=0.56,
                                 sheet=(cv.w, cv.h), edge=0.55, taken=taken)
    if not sites:
        return taken, []
    # 远近 — a plate's plane. Mostly front, because a crown read back to
    # front wants its weight at the front; the far ones are there to give
    # the near ones something to be in front of.
    planes = []
    for s in sites:
        planes.append((rng.random() ** 0.72, s))
    planes.sort(key=lambda p: p[0])
    out = []
    for pl, s in planes:
        sx, sy, sa = s[0], s[1], s[2]
        scl = s[4]
        rr = R * scl * rng.uniform(0.84, 1.18) * (0.78 + 0.34 * pl)
        # 针不离枝 — the plate sits *on the end of the twig*, pushed out
        # along the twig's own direction by rather less than its radius, so
        # the wood runs into the needles instead of stopping beside them.
        # 0.42 was too far: with a lift of half a radius under the plate
        # the twig's last stretch was bleached away and what was left was a
        # branch pointing at a pompom. The tip travels with the plate so
        # `plate` can run a stub of wood back to it.
        cx = sx + math.cos(sa) * rr * 0.28
        cy = sy + math.sin(sa) * rr * 0.28
        out.append((pl, cx, cy, sa, rr, sx, sy))
    return taken, out


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into song_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [w for v in TITLE_AIR.values() for w in v]
             + [v["cn"] for v in SCENES.values()]
             + [v["cn"] for v in AIR.values()]
             + [v["cn"] for v in STYLES.values()]
             + list(SCENES) + list(AIR) + list(STYLES)
             + [TIANGAN, DIZHI, "日月年写于图春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a blob baked for this vocabulary; fall back to the siblings'
    (the family shares most characters), then to no 题款."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("song_font", "mei_font", "grape_font", "orchid_font",
                "bamboo_font", "gongbi_font", "shanshui_font"):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        try:
            return BrushFont(m.DATA, m.EM)
        except Exception as exc:
            print("  (font blob %s unreadable: %s)" % (mod, exc),
                  file=sys.stderr)
    return None


FONT = _load_font()


def _ganzhi(rng):
    return TIANGAN[rng.randrange(10)] + DIZHI[rng.randrange(12)]


def inscribe(cv, rng, occ, S, title):
    """Measure the block, then let the 留白 decide where it goes — the same
    machinery as the other ten painters."""
    W, H = cv.w, cv.h
    gs = S * rng.uniform(0.042, 0.058)
    ss = gs * rng.uniform(0.50, 0.64)
    lead, lead_s = rng.uniform(1.12, 1.26), rng.uniform(1.18, 1.34)
    season = "春夏秋冬"[rng.randrange(4)]
    sign = _ganzhi(rng) + season + "日" + ACTION[rng.randrange(len(ACTION))]
    gap = gs * 0.45
    sign_dy = gs * rng.uniform(1.0, 2.0)
    seal_s = gs * rng.uniform(0.80, 1.0)
    bw = gs + gap + ss
    bh = max(len(title) * gs * lead,
             sign_dy + len(sign) * ss * lead_s + gs * 0.25 + seal_s)
    margin = S * 0.040
    r0 = min(W, H) * 0.15
    for fy in (0.34, 0.52, 0.70):
        occ.add(W * 0.5, H * fy, r0, ink=False)
    block = None
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.5)
        paint_column(cv, rng, FONT, title, bx + gap + ss, by, gs, lead=lead)
        end = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, ss,
                           lead=lead_s, alpha=(0.60, 0.84))
        paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                   bx + (ss - seal_s) * 0.5, end + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
        block = (bx, by, bw, bh)
    if rng.random() < 0.5:                        # 压角章 — set against the 款
        gb.corner_seal(cv, rng, FONT, SEAL_WORDS, occ, S, block)


# --------------------------------------------------------------------------
# 章法 — composition
# --------------------------------------------------------------------------

def compose(cv, rng, scene_key, air_key, style_key, stat=None):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc = SCENES[scene_key]
    air = dict(AIR[air_key])
    st = STYLES[style_key]
    gb.set_light(rng)
    ground = gb.paint_ground(cv, rng)
    occ = Occupancy(W, H)
    k = air["k"]
    side = rng.choice((1.0, -1.0))          # which way the sheet leans
    lean = (sc["lean"][0] + (sc["lean"][1] - sc["lean"][0]) * rng.random()
            + air["lean"]) * side

    if "veil" in air:
        veil(cv, rgb(air["veil"][0]), air["veil"][1])
    if air.get("moon"):
        mr = S * rng.uniform(0.055, 0.085)
        mx = W * (rng.uniform(0.16, 0.30) if side > 0
                  else rng.uniform(0.70, 0.84))
        my = H * rng.uniform(0.11, 0.24)
        moon(cv, rng, mx, my, mr, I("qing"))
        occ.add(mx, my, mr * 1.5)

    # ---- the photograph of the paper. Everything after this line can be
    # ---- lifted back out of the sheet; nothing before it can.
    res = Reserve(cv)

    # ---- 地 — what the tree stands on --------------------------------
    #
    # Every scene stands on the same 坡 now. The 偃盖 sheets used to build a
    # cliff here instead and hang the pine off its lip; see the scene's own
    # note in SCENES for why that is gone.
    roots = []                       # (x, y, scale) — where a trunk starts
    # 坡 — where the ground is. The trunk's own foot goes *below* it,
    # off the bottom of the sheet: a pine whose butt is drawn in full
    # ends in a rounded cap a few centimetres above the border, and
    # that cap reads as a post sawn off and stood on the paper. 立干
    # 出纸 is the norm in every hanging scroll for the same reason.
    yb = H * rng.uniform(0.86, 0.94)
    n = sc["n"]
    if scene_key == "偃盖":
        # 偃盖 spreads both ways and nearly to both borders, so unlike the
        # other single-tree scenes its trunk wants the middle of the sheet:
        # off to one side the canopy runs off that edge and leaves the far
        # half of the paper empty.
        xs = [rng.uniform(0.40, 0.60)]
    elif n > 1:
        xs = _spread(rng, n, 0.20, 0.80, 0.22)
    else:
        xs = [rng.uniform(0.30, 0.70)]
    # 一主一宾 — of two pines, one leads. Equal trees side by side is
    # the oldest compositional mistake there is.
    scales_ = [1.0] + [rng.uniform(0.58, 0.76) for _ in range(n - 1)]
    rng.shuffle(xs)
    for i in range(n):
        roots.append((W * xs[i], H * rng.uniform(1.00, 1.08), scales_[i]))
    rockp = None
    if rng.random() < sc["rock"]:
        # 石不可扁 — at three times as wide as it is tall a 坡石 comes
        # out a smooth grey dome with a rim round it, which is a hill
        # on a map and not a rock. Half as wide again as it is high is
        # enough to get a shoulder and a face into it.
        #
        # And 松石 means the two *touch*. Siting the rock from `side`
        # alone, before the trunks had positions, left a small grey
        # lump sitting on empty ground halfway across the sheet from
        # the tree — two subjects on one paper, which is the thing the
        # 题 names as one. It goes at the leading trunk's foot now,
        # overlapping it by a third of its own width, and it is big
        # enough to be a rock the pine is growing out of.
        #
        # Which rock is not a free choice either. bamboo's `rock` is a
        # 折带 slab — level shelves meeting straight flanks at hard
        # corners, which is exactly right under bamboo and, under a pine,
        # came out as a grey keep with battlements and a chimney standing
        # at the tree's foot. orchid.py met the same wall and wrote the
        # answer into `stone`'s docstring: beside a plant you want a 坡石,
        # a flat-topped lump with a cleft in it and a lower lump
        # shouldering out of its side. The coin-flip between the two is
        # gone; half the sheets in the scene were masonry.
        #
        # And the rock is only *sited* here. It is painted after the wood,
        # below, because of what has to happen where the two meet.
        rw = S * rng.uniform(0.30, 0.46)
        rh = rw * rng.uniform(0.52, 0.78)
        tx = W * xs[scales_.index(1.0)]
        rx = tx + rw * rng.uniform(0.18, 0.46) * (1.0 if side > 0 else -1.0)
        rx = min(W - rw * 0.30, max(rw * 0.30, rx))
        rockp = (rx, yb + rh * 0.08, rw, rh)
        occ.add(rx, yb - rh * 0.3, max(rw, rh) * 0.5)

    # ---- 树 — back to front, the smaller tree first -------------------
    roots.sort(key=lambda r: r[2])
    wind_a = math.pi * (0.0 if side > 0 else 1.0) + rng.uniform(-0.5, 0.5)
    taken = []
    crowns = []
    wood = []                        # every stroke of trunk, for `_infront`
    for rx, ry, rs in roots:
        scale = rs
        out = side
        a0 = -math.pi * 0.5 + lean * rng.uniform(0.8, 1.2)
        E = dict(cv=cv, rng=rng, st=st, air=air, sc=sc, occ=occ, res=res,
                 wood=wood,
                 S=S, k=k * (0.70 + 0.30 * scale), old=sc["old"],
                 up=-math.pi * 0.5 + lean * 0.5,
                 flat=(scene_key == "偃盖"),      # see `_flat`
                 hang=sc["hang"], out=out,
                 maxd=2, scale=scale, tips=[], hwat=_hwat)
        L = S * rng.uniform(*sc["L"]) * scale
        w0 = S * rng.uniform(*sc["w0"]) * (0.72 + 0.34 * scale)
        taken, pls = tree(E, rx, ry, a0, L, w0,
                          max(3, int(rng.randint(*sc["plates"]) * scale)),
                          taken)
        crowns.extend(pls)
        if stat is not None:
            stat.setdefault("trees", 0)
            stat["trees"] += 1

    # ---- 石 — after the wood, and then cut out of it -------------------
    #
    # 松石图 is two masses that touch, and the touching is the whole of the
    # subject. Painted before the trunk, under a multiply composite, the
    # rock was not behind it: the 皴, the 苔点 and the crest line all came
    # straight through the wood, and the trunk's own 龙鳞 came through the
    # rock, so the two read as a pane of smoked glass laid over a tree.
    # Neither mass was in front, which is the one thing a passage like this
    # has to say.
    #
    # Draw order cannot fix that — multiply has no draw order. What fixes it
    # is a second photograph of the paper, taken with the wood on it: the
    # rock goes down over everything, and then the wood is lifted straight
    # back out of the rock, edge and all. The rock stops at the trunk, the
    # trunk keeps its ink, and a hair of bare paper runs between them.
    if rockp is not None:
        wet = Reserve(cv)
        rx, ry, rw, rh = rockp
        stone(cv, rng, rx, ry, rw, rh, st, k=0.70 * k, moss=1.0)
        rx0, rx1 = rx - rw * 0.56, rx + rw * 0.56
        ry0, ry1 = ry - rh * 1.25, ry + rh * 0.40
        for pts, hws in wood:
            if (min(p[0] for p in pts) > rx1
                    or max(p[0] for p in pts) < rx0
                    or min(p[1] for p in pts) > ry1
                    or max(p[1] for p in pts) < ry0):
                continue
            _infront(wet, cv, pts, hws)
    if sc["turf"] > 0.0:
        for rx, _ry, rs in roots:
            turf(cv, rng, rx + rng.gauss(0, S * 0.04),
                 yb + H * rng.uniform(0.0, 0.05), S * 0.22 * rs, st,
                 k=0.55 * k, n=int(10 * sc["turf"]))

    # ---- 针 — every plate on the sheet, back to front ------------------
    #
    # Across *all* the trees, not tree by tree: 疏密 is a property of the
    # sheet (bamboo's `cluster_sites` docstring, learned there the hard
    # way), and so is depth. Sorted together, a far plate of the near tree
    # correctly goes down before a near plate of the far one.
    crowns.sort(key=lambda c: c[0])
    nc = len(crowns)
    cut = air.get("cloud", 0.0) and int(nc * 0.45)
    for idx, (pl, cx, cy, ca, rr, tx, ty) in enumerate(crowns):
        if cut and idx == cut:
            cloud(cv, rng, res, H * rng.uniform(0.42, 0.62),
                  H * rng.uniform(0.045, 0.075), W, alpha=0.60)
        plate(cv, rng, res, cx, cy, rr, ca, st, air, k=k,
              tone=0.34 + 0.66 * pl,
              lift=0.0 if pl < 0.42 else 0.30 + 0.42 * (pl - 0.42) / 0.58,
              wind=air["comb"], wind_a=wind_a,
              snow=air.get("snow", 0.0) * max(0.0, pl - 0.3),
              wheel=0.34 - 0.18 * pl, twig=(tx, ty))
        occ.add(cx, cy, rr * 0.86)
    if stat is not None:
        stat["plates"] = nc

    if air.get("snow"):
        snowfall(cv, rng, S, int(S * 0.22))

    title = (TITLE_A[rng.randrange(len(TITLE_A))] if rng.random() < 0.62
             else TITLE_AIR[air_key][rng.randrange(len(TITLE_AIR[air_key]))])
    title += TITLE_B[rng.randrange(len(TITLE_B))]
    inscribe(cv, rng, occ, S, title)
    return title, ground


def age(cv, rng):
    """A last breath of ink dust, as every painter in this family does."""
    W, H = cv.w, cv.h
    for _ in range(int(W * H * 0.00008)):
        disc(cv, rng.uniform(0, W), rng.uniform(0, H),
             rng.uniform(0.6, 1.8), I("dan"), rng.uniform(0.05, 0.22))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def paint(path, size=(900, 1300), seed=None, scene=None, air=None,
          style=None, quiet=False, stat=None):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    ak = air or sorted(AIR)[rng.randrange(len(AIR))]
    yk = style or sorted(STYLES)[rng.randrange(len(STYLES))]
    for key, tbl, what in ((sk, SCENES, "scene"), (ak, AIR, "air"),
                           (yk, STYLES, "style")):
        if key not in tbl:
            raise SystemExit("unknown %s %r; try --list" % (what, key))

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, ak, yk, stat=stat)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-24s %s  %s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk, AIR[ak]["cn"],
                 STYLES[yk]["cn"], ground, seed, size[0], size[1],
                 time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese ink-pine paintings (松).")
    ap.add_argument("-o", "--out", default="song.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="孤松|双松|偃盖|松石")
    ap.add_argument("--air", help="清晓|松风|积雪|月下")
    ap.add_argument("--style", help="墨松|浅绛|苍翠")
    ap.add_argument("--size", default="900x1300", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for key, v in SCENES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("风致 (air):")
        for key, v in AIR.items():
            print("  %-6s %-5s %s" % (key, v["cn"], v["en"]))
        print("设色 (colour):")
        for key, v in STYLES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("画松: 干如虬龙, 皮如龙鳞, 枝如蟹爪, 叶如攒针 — "
             "下不生枝, 枝必横出, 梢必上仰")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("松 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
