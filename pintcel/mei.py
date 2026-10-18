#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mei.py — 随机生成中国水墨梅花图 / Random Chinese ink-plum paintings (墨梅).

梅 is the first of the 四君子 and the oldest of them as a painting subject:
华光和尚 taught it in the eleventh century, 宋伯仁's 《梅花喜神谱》 gave it a
hundred named attitudes, and 王冕 fixed the form this generator follows —
blossoms drawn as *rings of line* round bare paper, on wood as hard as iron.
范成大's 《梅谱》 states the whole aesthetic in sixteen characters, and they
are the specification:

    贵稀不贵繁   sparse, not crowded
    贵老不贵嫩   old, not young
    贵瘦不贵肥   lean, not fat
    贵含不贵开   in bud, not in full flower

and the painters' own 画诀 says how:

    干如屈铁   the trunk is bent iron — one broad 侧锋 stroke, dry, with a
               single firm dark contour down its shaded flank. Not a tube.
    枝如鹿角   the branches are antlers: 梅 wood *breaks* direction at a
               joint, it does not curve through it. A smooth bezier is a
               willow.
    花分五出   five petals, drawn as five rings; and every flower is a
               different size, because a row of equal circles is a chain.
    蒂如丁香   the calyx is a clove — three short dark strokes, and it is
               the only thing that attaches a pale ring of line to the wood
               instead of letting it float over it.
    蕊如鼠须   the stamens are mouse whiskers, each with its dark anther.
               They are the flower's eye.
    有正有背   face, profile and back. A sheet of face-on flowers is a
               diagram of a flower, not a tree in bloom.
    忌对节     never two shoots from one joint, never two parallel, and
               where two branches cross they should make the character 女.

The structural problem this module exists to solve is **occlusion**.
Everything in this family composites by multiply — `buf = o + (o*c/255 - o)*a`
— and multiply is order-independent, so *draw order hides nothing*: a pale
mark laid over a dark one simply leaves the dark one showing. Every sibling
painter lives with that (grape's §7.2 cuts three stalks short because of it).
梅 cannot: a 圈花 blossom is a thin pale ring whose *interior is bare paper*,
it hangs in front of the branch it grows on, and the branch is the darkest
thing on the sheet. Painted in any order, the wood shows straight through
every flower.

The answer is `Reserve`: photograph the paper after the ground and the wash
and before the first stroke of ink, then 留白 by **restoring** that
photograph inside the shape of each blossom before drawing it. The flower is
not painted over the branch — the branch is *lifted out of it*, which is
also what the hand does, since the painter who works this way leaves the
paper alone and draws the wood around the flowers. Flowers are drawn far to
near and each lifts the ones behind it, so the sheet gets real depth out of
one bytes() copy. The snow in 踏雪 is the same trick, and so are the hollows
in the old trunk.

Usage
-----
    python3 mei.py                              # one random painting
    python3 mei.py -n 6                         # six of them
    python3 mei.py --scene 古干 --air 月下
    python3 mei.py --style 红梅 --seed 7
    python3 mei.py --size 900x1300 -o scroll.png
    python3 mei.py --list                       # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Noise, Occupancy, BrushFont, clamp01, disc,
                    mix, rgb, paint_column, paint_seal, TIANGAN, DIZHI)
# 一家之墨 — the ink stone, the 兑水 rule, the full-sheet veil, the 留白 moon
# and the snow are bamboo.py's. 梅 is painted with the same water as 竹 and
# 兰; two modules of this family disagreeing about what 淡墨 is would show at
# once on any sheet that carried both.
from bamboo import I, dilute, veil, moon, snowfall
# The family's two renderers. `sweep` is a brush dragged along a polyline,
# accumulated by maximum and composited once — the only line tool here that
# does not bead at its joints, which matters on wood that kinks every few
# centimetres. `touch` is one brush-touch with a polar outline and an ink
# field that returns *shade and coverage*; 点梅 petals are one call each.
from orchid import sweep
from grape import touch

TAU = math.tau


# --------------------------------------------------------------------------
# 设色 — the four schools.
#
# 墨梅 is the orthodox form and the others are departures from it, so all
# four keep ink wood: it is only the flower that takes colour. `ring` is the
# two stops of the 圈 line's own ramp, `fill` the wash inside the petal (or
# None, which means the petal stays paper), `method` whether the flower is
# drawn as line (圈花) or laid as touches (点梅).
#
# 白梅 is the interesting entry, and the reason the `halo` column exists.
# White cannot be painted here — white *is* the paper, and white paint over
# a 米黄 ground under multiply compositing is a no-op. So a white plum is
# made the way the hand makes it: a breath of 淡墨 washed *outside* the
# petal rim (衬托), after which the untouched paper inside reads as white.
# --------------------------------------------------------------------------

STYLES = {
    "墨梅": dict(cn="墨梅", en="plain ink, 圈花 outline — after 王冕",
                 method="圈", wood=("#8A8276", "#221E1A"),
                 ring=("#8E8678", "#241F1B"), fill=None,
                 heart="#1C1917", halo=0.0),
    "白梅": dict(cn="白梅", en="white — the petal is the paper (衬托)",
                 method="圈", wood=("#8E8679", "#262220"),
                 ring=("#A09686", "#3A332B"), fill=None,
                 heart="#241F1B", halo=1.0),
    "红梅": dict(cn="红梅", en="vermilion, 点梅 — the petal is one touch",
                 method="点", wood=("#857D70", "#241F1B"),
                 ring=("#BE8A80", "#7E2E22"), fill=("#E6B4A8", "#B03A2A"),
                 heart="#2A1410", halo=0.0),
    "绿萼": dict(cn="绿萼", en="green calyx — 绿萼梅",
                 method="圈", wood=("#857F6E", "#2A2620"),
                 ring=("#84887A", "#252A20"), fill=("#E4E8D4", "#BCC6A4"),
                 heart="#263019", halo=0.0),
}

# 章法 — the four framings.
#
# `w0` is the trunk's half-width as a fraction of the sheet's short side and
# `L` its length in the same units; `segs`/`kink` say how many times the wood
# breaks direction and how hard; `old` drives everything that says age at
# once (the dryness, the 侧锋 load, the bark, the hollows); `whips`/`Lw`/
# `fork` are the new shoots; `span` the stretch of the parent they may leave
# from; `clusters` how many knots of blossom each shoot carries and `bud` how
# much of the sheet is 贵含不贵开.
SCENES = {
    "横斜": dict(cn="疏影", en="one old bough across the sheet — 疏影横斜",
                 entry="side", w0=(0.024, 0.034), L=(0.80, 1.02),
                 segs=(3, 5), kink=0.40, old=0.70, limbs=(1, 2), whips=(4, 6),
                 Lw=(0.22, 0.36), fork=(0.50, 0.95), span=(0.08, 0.92),
                 spurs=(1, 3), clusters=(2, 4), bud=0.26, scars=(1, 2)),
    "垂枝": dict(cn="垂梅", en="branches hanging from the top edge (折枝)",
                 entry="top", w0=(0.022, 0.030), L=(0.70, 0.92),
                 segs=(3, 4), kink=0.34, old=0.50, limbs=(1, 2), whips=(4, 6),
                 Lw=(0.20, 0.33), fork=(0.45, 0.85), span=(0.10, 0.94),
                 spurs=(0, 2), clusters=(2, 4), bud=0.24, scars=(0, 1)),
    "折枝": dict(cn="折枝", en="a single cut branch — 贵稀不贵繁",
                 entry="cut", w0=(0.014, 0.020), L=(0.66, 0.88),
                 # 屈铁 — at least two 节. With `segs=(2, 4)` the common
                 # case was one break in 1000 px of wood, and a single
                 # alternating kink leaves the path going where it started:
                 # a straight tapered rod across empty paper, which is
                 # half of why this scene read as a western drawing even
                 # after every mark on it had been fixed (§2.13).
                 segs=(3, 4), kink=0.46, old=0.25, limbs=(1, 2), whips=(3, 5),
                 Lw=(0.18, 0.30), fork=(0.55, 1.00), span=(0.12, 0.90),
                 spurs=(0, 2), clusters=(2, 4), bud=0.34, scars=(0, 0)),
    "古干": dict(cn="老梅", en="an old trunk from the foot of the sheet",
                 entry="foot", w0=(0.032, 0.048), L=(0.78, 1.00),
                 segs=(3, 5), kink=0.46, old=1.00, limbs=(2, 3), whips=(5, 7),
                 Lw=(0.22, 0.38), fork=(0.55, 1.05), span=(0.14, 0.95),
                 spurs=(2, 4), clusters=(2, 4), bud=0.22, scars=(1, 3)),
}

# 风致 — one air for the whole sheet, as bamboo.py has one weather.
# `lean` is the slant everything that grows shares, `wet` scales the
# softness of every edge, `dry` the 飞白 in the long strokes, `k` the
# overall strength of the ink.
AIR = {
    "清晓": dict(cn="春信", en="still, at first light",
                 lean=0.00, wet=1.00, dry=0.30, k=1.00),
    "寒风": dict(cn="风梅", en="cold wind off the water",
                 lean=0.30, wet=0.84, dry=0.46, k=1.00),
    "踏雪": dict(cn="雪梅", en="snow lying on the wood — 雪里见春",
                 lean=0.08, wet=1.10, dry=0.24, k=0.94,
                 veil=("#8A94A2", 0.09), snow=1.0),
    "月下": dict(cn="月梅", en="暗香浮动月黄昏 — by moonlight",
                 lean=0.10, wet=1.30, dry=0.18, k=0.86,
                 veil=("#6E7A86", 0.10), moon=1.0),
}


# --------------------------------------------------------------------------
# 留白 — reservation, and why this module needed a new idea.
#
# Every mark in this family multiplies into the paper. That is what makes
# the ink read as ink: the ground's tone and grain stay visible through it,
# two washes over each other give the third value for free, and nothing can
# ever be brighter than the sheet. The price is that **multiply is
# order-independent**: compositing a over b and b over a give the same
# pixel, so nothing can be drawn *in front* of anything. Grape's report
# states the consequence (a highlight is not opaque, so three of its stalks
# stop short of where anatomy puts them); crab's goes further (two presses
# are two objects and no overlap dissolves the seam).
#
# 梅 is the subject where that stops being a limitation and becomes a wall.
# A 圈花 blossom is five thin pale rings and the flower *is* the paper
# inside them — and it hangs in front of the branch that carries it, which
# is the darkest mark on the sheet. There are only three ways out:
#
#   1. Give the ring a coverage channel, as `touch` has, and let the flower
#      paint its own interior opaque. But the interior is *paper*: covering
#      it with the paper's colour would be a sticker, flat, grainless, and
#      wrong on every ground but the one it was sampled from.
#   2. Reserve the blossom sites *before* the wood goes down and route the
#      branches around them, the way `leaf_sites` reserves before
#      `paint_leaf` in gongbi. Correct, and it ruins the wood: a branch is
#      one continuous stroke of the hand and it cannot detour.
#   3. Photograph the paper before the first stroke of ink and put the
#      photograph back inside the flower. Which is this class.
#
# `lift` is the whole idea, and it is nine lines of arithmetic over the
# same two laws as everything else here — a shape is an inequality, and the
# alpha at its edge is cov = d + 0.5. The flower's interior ends up holding
# the ground's own tone, weave, fibres and foxing, because it *is* the
# ground; and since each flower lifts whatever was painted before it, the
# blossoms drawn last are the blossoms in front. One bytes() copy per sheet
# buys this module depth that no sibling has.
# --------------------------------------------------------------------------

class Reserve:
    """A photograph of the paper, and the means to put it back.

    Snapshot after the ground, the veil and the moon and before the first
    stroke of ink; then `lift` restores it inside an arbitrary polar shape,
    which is 留白 done the way the hand does it — the paper was never
    painted, so it keeps its grain.
    """

    def __init__(self, cv):
        self.base = bytes(cv.buf)

    def lift(self, cv, cx, cy, rad, rot=0.0, squash=1.0, alpha=1.0,
             feather=0.9, pad=0.0, na=192):
        """Restore the paper inside the polar outline `rad(th) -> radius`.

        Same frame convention as `touch`: angle 0 is +x, `squash` flattens
        the shape across y and `rot` turns it. `alpha` below 1 leaves a
        ghost of whatever was there, which is what a flower with another
        flower behind it actually looks like.
        """
        rt = [rad(i * TAU / na) + pad for i in range(na)]
        rmax = max(rt)
        if rmax <= 0.3:
            return
        sq = max(0.05, squash)
        pd = rmax * (sq if sq > 1.0 else 1.0) + feather + 1.0
        x0, y0, x1, y1 = cv.clip(cx - pd, cy - pd, cx + pd, cy + pd)
        if x0 >= x1 or y0 >= y1:
            return
        base, buf = self.base, cv.buf
        ca, sa = math.cos(rot), math.sin(rot)
        inv_sq = 1.0 / sq
        k_ang = na / TAU
        inv_f = 1.0 / max(0.05, feather)
        cw = cv.w
        for py in range(y0, y1):
            dy = py + 0.5 - cy
            row = py * cw
            for px in range(x0, x1):
                dx = px + 0.5 - cx
                lx = dx * ca + dy * sa
                ly = (-dx * sa + dy * ca) * inv_sq
                d = math.sqrt(lx * lx + ly * ly)
                if d > pd:
                    continue
                th = math.atan2(ly, lx)
                if th < 0.0:
                    th += TAU
                fa = th * k_ang
                i0 = int(fa)
                fr = fa - i0
                if i0 >= na:
                    i0 -= na
                i1 = i0 + 1
                if i1 >= na:
                    i1 = 0
                r = rt[i0] + (rt[i1] - rt[i0]) * fr
                if d > r + feather:
                    continue
                cov = (r - d) * inv_f + 0.5
                if cov <= 0.004:
                    continue
                if cov > 1.0:
                    cov = 1.0
                a = alpha * cov
                i = (row + px) * 3
                o = buf[i]
                buf[i] = int(o + (base[i] - o) * a)
                o = buf[i + 1]
                buf[i + 1] = int(o + (base[i + 1] - o) * a)
                o = buf[i + 2]
                buf[i + 2] = int(o + (base[i + 2] - o) * a)


# --------------------------------------------------------------------------
# the small shared helpers
# --------------------------------------------------------------------------

def drag(cv, pts, w0, w1, lo, hi, alpha=1.0, d0=0.45, d1=0.92, taper=1.0,
         **kw):
    """One tapering stroke along `pts` — the convenience wrapper over
    `sweep` that most of the linework in this module wants."""
    n = len(pts) - 1
    if n < 1:
        return
    hws, dens = [], []
    for i in range(n + 1):
        t = i / n
        hws.append(max(0.25, w0 + (w1 - w0) * t ** taper))
        dens.append(clamp01(d0 + (d1 - d0) * t))
    sweep(cv, pts, hws, dens, lo, hi, alpha, **kw)


def _perp(pts, i, d):
    """The point `d` pixels to one side of `pts[i]`, along *its own* normal.

    gongbi's 皴 bug, which this module inherits the fix for: offsetting a
    texture from the straight chord between the ends of a curved stroke
    walks the texture off the stroke in the middle. The normal has to be
    measured at the node.
    """
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
    """`n` parameters in [lo, hi], no two closer than `sep` — 忌对节.

    Two shoots from one joint is the first of the 梅谱's prohibitions and
    the easiest one to break by accident, because uniform draws cluster.
    """
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


# --------------------------------------------------------------------------
# 枝干 — the wood.
#
# 梅 wood is the one thing in this family that must *not* be drawn as a
# smooth curve. 「枝如鹿角」: a plum branch runs nearly straight for a span,
# breaks at a joint, and runs nearly straight again in a new direction, and
# the breaks alternate in sign. Draw it as one bezier and it reads as a
# length of hose; draw it as a wandering line the way grape's vine does and
# it reads as a vine. So `wood_path` is a sequence of straight runs with
# kinks between them and only a trace of drift inside each run.
#
# `pull` is the one concession: a shoot leaves its parent at a real angle
# and then turns slowly toward the direction growth actually goes in (up,
# or down for 垂枝), which is what makes a sheet of branches agree with
# itself instead of pointing every way at once. It is the same correction
# the bamboo report had to make twice and the 虾蟹 report once: a random
# direction per object is not variety, it is noise.
# --------------------------------------------------------------------------

def wood_path(rng, x, y, a0, L, segs=4, kink=0.42, bow=0.10, grow=None,
              pull=0.0, cone=1.05, bounds=None, slack=0.07, step=None,
              want=None):
    """A branch: straight runs, hard joints. Returns (points, joint indices).

    `want`, if given, is a one-element list the function writes its
    intended number of steps into, so the caller can tell a path that
    finished from one the sheet's edge cut off."""
    want = want if want is not None else [0]
    step = step or max(2.5, L / 54.0)
    n = max(6, int(L / step))
    want[0] = n
    cuts = set()
    if segs > 1 and n > 10:
        lo, hi = int(n * 0.16), int(n * 0.90)
        if hi - lo > segs:
            cuts = set(rng.sample(range(lo, hi), min(segs - 1, hi - lo)))
    pts = [(x, y)]
    joints = []
    ang = a0
    sgn = rng.choice((1.0, -1.0))
    drift = rng.uniform(-bow, bow) / max(1, n // max(1, segs))
    for i in range(1, n + 1):
        if i in cuts:
            # 节 — the break. Alternating in sign, because two kinks the
            # same way is a curve and a curve is not plum wood.
            ang += sgn * kink * rng.uniform(0.55, 1.05)
            sgn = -sgn
            drift = rng.uniform(-bow, bow) / max(1, n // max(1, segs))
            joints.append(len(pts) - 1)
        ang += drift + rng.uniform(-0.010, 0.010)       # 颤笔
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
            if not (-slack * W < x < (1.0 + slack) * W
                    and -slack * H < y < (1.0 + slack) * H):
                break
        pts.append((x, y))
    return pts, joints


def bough(cv, rng, pts, w0, w1, st, air, alpha=0.95, old=0.0, bone=0.0,
          k=1.0, side=None, knots=2.2):
    """干如屈铁 — one stroke of wood. Returns its half-widths.

    Built as crab's §3 insists: *one* mass laid with a single side-loaded
    stroke, and everything else — the contour, the bark, the moss — drawn
    as line inside it. Two overlapping strokes would be two pieces of wood
    however much they overlapped, because each wears a complete contour.
    """
    n = len(pts) - 1
    if n < 1:
        return []
    lt, dp = rgb(st["wood"][0]), rgb(st["wood"][1])
    ph, ph2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
    hws, dens = [], []
    swing = 0.09 + 0.26 * old
    cap = 0.20 + 0.26 * old
    # How much of "wood" versus "line" this stroke is, as a number rather
    # than a flag. The first version switched on `w0 > 5`, which put a
    # 6 px shoot and a 40 px trunk under the same pale wide-wood ramp and
    # made every twig a grey shadow; the parameters all want to slide.
    f = clamp01((w0 - 3.5) / 9.5)
    thin = 1.0 - f
    # 墨分五色, along the length as well as across it. A branch whose value
    # only varies with its width comes out one tone with a ripple in it —
    # which is what "a grey tube" means — so the load gets a slow term of
    # its own, at a period unrelated to the 提按.
    ph3 = rng.uniform(0, TAU)
    f3 = rng.uniform(1.2, 2.6)
    # 新枝用浓墨 — thin wood is the *darkest* mark on the sheet, not the
    # palest. The first version read its load off the width alone, which is
    # backwards: it gave a 40 px trunk a heavy load and a 3 px whip a light
    # one, so every shoot came out as a pale grey band — and a pale grey
    # band of even width is exactly the thing that reads as a western
    # painting of a twig rather than a brush stroke of one. The hand does
    # the opposite: it saves the water for the big mass and writes the new
    # wood with the tip of a loaded brush, in one quick 浓墨 line.
    base = 0.56 - 0.06 * f * old + 0.17 * thin
    for i in range(n + 1):
        t = i / n
        # 提按 — the hand presses and rises, and its *value* moves with its
        # width, because pressing a brush down puts out more ink as well as
        # more hair. `win` is the repair: the modulation has to be zero at
        # the root of the stroke. Without it a short twig came out as a
        # lens — thin where it joins its parent, fat in the middle, pointed
        # at the tip — which reads as a dark leaf stuck to the branch, and
        # a dozen of those was the worst fault on the first sheets.
        win = min(1.0, t * 3.2)
        p = win * (swing * math.sin(t * knots * TAU + ph)
                   + 0.45 * swing * math.sin(t * knots * 2.7 * TAU + ph2)
                   + rng.uniform(-0.04, 0.04))
        if p > cap:
            p = cap
        hws.append(max(0.42, (w0 + (w1 - w0) * t ** 1.15) * (1.0 + p)))
        dens.append(clamp01(base + 0.22 * (1.0 - t) + 0.22 * p
                            + 0.12 * math.sin(t * f3 * TAU + ph3)))
    sd = side if side is not None else rng.uniform(-0.35, 0.35)
    # 飞白 — the streaks a dry brush leaves run *along* the drag and are
    # spaced by the hairs of the brush, not by the width of the stroke. So
    # the frequency has to scale with the width or a wide trunk comes out
    # corrugated: three broad stripes down its whole length, which is what
    # the first 古干 looked like. Wide wood also gets *less* of it, because
    # its dryness belongs to the bark pass below.
    # 侧锋 — and it has to be *hard*. `sh` is density plus side*sl, clamped
    # into a 25-entry ramp, so a stroke whose density sits near 0.7 saturates
    # the whole width into mid-grey however much side is asked for: the way
    # to get a flank of 浓墨 against a flank of dry paper is a low load and a
    # steep tilt, not a high load and a gentle one.
    # ...and `lo` must be pale, but it must **not** be water, which is the
    # one correction in this function that had to be made twice in
    # opposite directions. The first version ran the ramp from
    # dilute(lt, 0.88) — 88 % of full pigment — so the palest tone the
    # stroke could reach still darkened the paper to mid-grey and no
    # side-loading could produce a dry flank at all: a grey rubber tube.
    # The fix took `lo` down to dilute(lt, 0.14), which is 86 % water, and
    # bought a worse fault: `sh = dens + side·sl` is *linear* across the
    # stroke, so a ramp whose low end is paper and whose high end is 焦墨
    # is a smooth sweep from black to white across the width — which is
    # the definition of an airbrushed cylinder. Every trunk came out lit.
    #
    # 飞白 is not a pale *tone*, it is bare paper **in streaks**, and the
    # way to get both is to put the body of the ramp in the middle and let
    # `flyback` lift the paper back out in hard-edged lines. So the low end
    # is a light wood grey — 淡墨, the second of 墨分五色, not 白 — and 白
    # arrives as a separate mark with an edge on it.
    # ...and only *old* wood is dry at all. Width alone was the wrong gate:
    # a 折枝 branch is 16 px across and young, and giving it the pale
    # wide-wood ramp turned it into a grey tube with a dark wire down one
    # side.
    # ...and the amount of water in it is a question about *age*, not about
    # width. The `-f·0.10` term applied to every wide stroke whatever its
    # age, so 折枝 — wide and young — kept a near-paper flank running its
    # whole length. One continuous pale stripe beside one continuous dark
    # band is a lit cylinder no matter how much bark is drawn on it; young
    # wood is painted in one fairly even 中墨 and only old wood is thirsty.
    lo = dilute(mix(lt, dp, 0.34 * thin), 0.92 - f * (0.04 + 0.26 * old))
    # 干如屈铁 — and iron has a ragged edge. This is the correction the
    # first version got wrong twice over, and both mistakes are worth
    # keeping because they are easy to make again:
    #
    #   * **roughness has a scale, and the scale was the node spacing.**
    #     Jitter was added to `hws` in the loop above, one sample per node —
    #     but `wood_path` lays its nodes `L/54` apart, which is 17–20 px on
    #     a trunk. A ±10 % wobble sampled every 18 px is not a rough edge,
    #     it is a *slow* edge: the eye reads it as the drawn curve of the
    #     silhouette and the stroke stays smooth. The amplitude was never
    #     the problem. So the path is resampled here to a few pixels and
    #     the jitter applied to *that*, which puts the raggedness at the
    #     scale of a hair rather than of a branch.
    #   * **it was measured in fractions of the half-width.** A 1.4 px whip
    #     got ±0.14 px of wobble and came out drawn with a ruler. Wood is
    #     ragged by about the same *absolute* amount whatever its size —
    #     the bark is the same bark — so the floor here is in pixels.
    #
    # And the jitter goes on the centreline as well as on the width, which
    # is what actually buys two independent edges: `sweep` has one
    # half-width per node, so jittering only `hws` swells and pinches both
    # flanks *together*, and a symmetric bulge is read as 提按, not as a
    # torn edge. Width jitter plus an equal and opposite wander of the
    # spine gives edges that disagree.
    spts, shws = _densify(pts, hws, max(4.0, min(13.0, 3.0 + w0 * 0.42)))
    _, sdens = _densify(pts, dens, max(4.0, min(13.0, 3.0 + w0 * 0.42)))
    # ...and the floor was still only in pixels at the *thin* end. `thin`
    # is zero for anything above about thirteen pixels of half-width, so a
    # 折枝 — a 13 px branch, the one scene that is nothing but one branch —
    # got 0.34 px of absolute wobble plus ±8.5 % of its width, which on a
    # 27 px bough is ±2 px of mostly-correlated swell. It came out a
    # drawing-office pipe. The brush's own raggedness does not shrink with
    # the stroke after the first few pixels: it is the paper's tooth and
    # the splay of the hairs, worth a pixel and a half on anything the
    # brush is actually loaded for.
    rough = 0.34 + 0.52 * thin
    jpts = list(spts)
    for i in range(len(shws)):
        hw = shws[i]
        # ...and the absolute part belongs to *wide* wood only. A pixel and
        # a half of unconditional wobble is bark on a 40 px trunk and more
        # than half the whole width of a 2.6 px whip: drawn independently
        # at 4 px node spacing it pinched the thin strokes to nothing and
        # swelled them to double, and every limb and whip on the close-up
        # came out a string of beads — a caterpillar, not a branch. The
        # paper's tooth is only visible where enough ink is dragged across
        # it to show, so the floor fades in over the first few pixels of
        # belly rather than applying to the tip.
        aj = rough * min(1.0, hw / 1.2) + 1.5 * clamp01((hw - 4.0) / 9.0)
        shws[i] = max(0.40, hw * (1.0 + rng.uniform(-0.085, 0.085))
                      + rng.uniform(-aj, aj))
        if 0 < i < len(shws) - 1:
            off = rng.uniform(-1.0, 1.0) * (aj * 0.85 + hw * 0.040)
            jpts[i] = _perp(spts, i, off)
        # 墨色斑驳 along the length, at the same short scale
        sdens[i] = clamp01(sdens[i] + rng.uniform(-0.055, 0.055))
    sweep(cv, jpts, shws, sdens, lo, dp, alpha * k,
          feather=0.95 * air["wet"] ** 0.3,
          core=0.08 + thin * (0.07 + 0.18 * old),
          # ...but a *steep* tilt empties half the stroke onto the pale end
          # of the ramp, and the two corrections fight. At side = 1.16 and a
          # load near 0.45 the arithmetic is `sh = dens + side*sl`, so every
          # pixel with `sl < -0.4` clamped to index 0 — which is to say the
          # lower **half of the trunk was flat, at the tone of water**, and
          # all of the value in the mark lived in the narrow band where the
          # ramp crossed. A dark edge above a wide smooth nothing is a
          # cylinder with a shadow under it, and that is the single most
          # western thing a brush can be made to draw.
          #
          # 侧锋 wants sh to span roughly [0.3, 1.0]: 浓墨 on the loaded
          # flank, a *dry grey* on the other, paper only in the last pixel
          # or two where `core` pulls the rim down. That is a middling tilt
          # over a middling load, not a steep tilt over a light one.
          side=sd * (0.20 + 0.56 * f * old),
          # ...and `sweep`'s own dryness is **least** use on thin wood,
          # which is where it had been turned up hardest. `dry` multiplies
          # coverage by `1 - q·sin(sl·dry_k·π + φ)²` — one phase for the
          # whole stroke — so it is strictly a *corrugation across the
          # width*, continuous from root to tip. On a trunk at
          # dry_k = w0·0.62 that is fine, a dozen fine grain lines; on a
          # 7 px limb at dry_k ≈ 2.7 with the `thin` term pushing q past
          # 0.6 it is five even stripes running the whole length — a zebra,
          # or a machined groove, and under magnification the loudest
          # remaining pen-mark on the sheet. 新枝用浓墨 says the same thing
          # from the other side: new wood is one quick line from a loaded
          # brush and has no 飞白 in it at all. So dryness now belongs to
          # age, and the thin end is damped rather than amplified — the
          # streaks that *should* be there arrive from `flyback`, which
          # draws them along the length with an edge on them.
          dry=air["dry"] * (0.30 + 0.42 * old) * (1.0 - 0.55 * thin),
          dry_k=max(2.2, w0 * 0.62) if f > 0.5 else 2.0 + 1.2 * old,
          dry_ph=rng.uniform(0, TAU))
    if bone > 0.0 and w0 > 2.6:
        # 屈铁 — the firm dark contour down the loaded flank. Without it a
        # dry broad stroke is a smudge; with it the smudge is a log.
        #
        # But *one* contour running the whole length at one weight is a pen
        # outline, and it was the single loudest western note on the sheet:
        # a brush cannot hold an even line over 900 px, and 笔断意连 says it
        # should not try. So it is laid in two to four pieces with paper
        # between them, each with its own weight and its own distance out
        # from the spine — 意到笔不到, the eye closes the gaps.
        d = -1.0 if sd >= 0.0 else 1.0
        i = rng.randint(0, max(1, n // 7))
        while i < n - 2:
            j = min(n, i + rng.randint(max(2, n // 6), max(3, int(n * 0.42))))
            if j - i >= 2:
                # ...and it has to sit *on* the edge, not inside it. At
                # 0.82–0.96 of the half-width the contour ran parallel to
                # the silhouette with a sliver of the ramp's pale end
                # showing outside it for hundreds of pixels — a dark line
                # with a bright rim beyond it, which is the single most
                # literal rendering of a lit cylinder available. 骨线 is
                # the edge of the mass; it is the same stroke's own tip
                # coming back, so it straddles the boundary.
                fr = rng.uniform(0.94, 1.06)
                sw = rng.uniform(0.74, 1.30)
                off = [_perp(pts, q, d * hws[q] * (fr + rng.uniform(-.03, .03)))
                       for q in range(i, j + 1, 2)]
                if len(off) >= 2:
                    drag(cv, off, max(0.50, w0 * 0.11 * sw),
                         max(0.42, w1 * 0.15 * sw),
                         I("zhong"), I("jiao"), bone * rng.uniform(0.40, 0.62)
                         * k, d0=0.55, d1=0.95, feather=0.78, core=0.12,
                         dry=0.25 + air["dry"] * 0.9,
                         dry_ph=rng.uniform(0, TAU))
            i = j + rng.randint(1, max(2, n // 8))
        # 两边见笔 — and one or two short dry pieces on the *starved* flank,
        # for the plainest of reasons: that side of the mass fades into the
        # paper, so without them the trunk has no bottom. A whole second
        # contour there would make it an outlined tube — 双勾, which is the
        # wrong technique for this subject — so it gets a third of the
        # weight, 淡墨 rather than 焦墨, and only where the wood turns.
        if f > 0.45 and n > 6:
            for _ in range(rng.randint(1, 2)):
                i = rng.randrange(1, max(2, int(n * 0.72)))
                j = min(n, i + rng.randint(2, max(3, n // 5)))
                if j - i < 2:
                    continue
                fr = rng.uniform(0.92, 1.04)
                off = [_perp(pts, q, -d * hws[q]
                             * (fr + rng.uniform(-0.03, 0.03)))
                       for q in range(i, j + 1, 2)]
                if len(off) < 2:
                    continue
                drag(cv, off, max(0.46, w0 * 0.070), max(0.40, w1 * 0.10),
                     dilute(I("zhong"), 0.72), I("nong"),
                     bone * rng.uniform(0.20, 0.34) * k, d0=0.40, d1=0.88,
                     feather=0.82, core=0.16, dry=0.5 + air["dry"],
                     dry_k=3.0, dry_ph=rng.uniform(0, TAU))
    return hws


def snap(cv, rng, res, pts, hws, air, k=1.0):
    """断口 — the cut face at the butt of a 折枝.

    A snapped branch is the one place in this module where the *end* of a
    stroke has to be an object rather than a taper, and it is the mark that
    tells the reader the sprig was broken off rather than drawn badly: a
    bough that simply stops in open paper reads as a line that ran out
    (§4.4 again, at the other end of the same stroke).

    Three marks. The face is a `lift` — fresh wood is paler than bark, and
    the only pale this family has is the paper. Then a broken dark rim.
    Then one or two splinters running back up the grain, because a plum
    does not break cleanly; that is what 屈铁 means when it fails.

    What the face must **not** be is the whole cross-section. It was: a
    full ellipse, lifted to 0.52–0.78, with one unbroken arc of itself
    drawn round it. That is a sawn tube drawn in perspective, and on a
    折枝 sheet — one branch, nothing else — it was the first thing the eye
    reached and the loudest western note in the module. 透视 is not a tool
    this painting owns. So the lift is windowed to a wedge of the face,
    which reads as wood torn open along the grain rather than as a lid.
    """
    hw = hws[0] if hws else 0.0
    if hw < 1.6 or len(pts) < 3:
        return
    ta = gb.tangent(pts, 0)
    x = pts[0][0] - math.cos(ta) * hw * 0.10
    y = pts[0][1] - math.sin(ta) * hw * 0.10
    sq = rng.uniform(0.24, 0.44)
    ph = rng.uniform(0, TAU)
    ph2, ph3 = rng.uniform(0, TAU), rng.uniform(0, TAU)

    # 断 — torn, not turned. The face used to be
    # `hw·(1 + 0.11·sin 3θ)` lifted to 0.52–0.78 and ringed by one
    # unbroken arc of itself, and that is a bright ellipse with a clean
    # outline round it: a sawn pipe drawn in perspective, which is the most
    # western mark the module made. Wood does not break along a conic. The
    # high harmonics are what make it a tear, the lift keeps far more tone
    # (fresh wood is paler than bark, not paler than paper), and the rim
    # goes down in two or three pieces below.
    def edge(th):
        return hw * (1.00 + 0.15 * math.sin(th * 2.0 + ph2)
                     + 0.13 * math.sin(th * 3.0 + ph)
                     + 0.09 * math.sin(th * 5.0 + ph3)
                     + 0.05 * math.sin(th * 7.0 - ph2))

    # the tear opens one way and dies the other: a wedge of the face, not
    # the face
    wa = rng.uniform(0, TAU)
    ws = rng.uniform(1.5, 2.9)

    def rad(th):
        d = (th - wa + math.pi) % TAU - math.pi
        return edge(th) * max(0.0, 1.0 - (abs(d) / (ws * 0.5)) ** 2.4)

    res.lift(cv, x, y, rad, rot=ta + math.pi * 0.5, squash=sq,
             alpha=rng.uniform(0.26, 0.46), feather=1.0)
    # the rim, in two or three broken pieces — and *only across the tear*.
    # Two arcs of 0.9–2.2 radians dropped at a random start between them
    # traced most of the ellipse, so the lid kept its outline even after
    # the pale face inside it had been cut back to a wedge. The break shows
    # where it shows; the rest of the butt is simply the end of the stroke.
    ca, sa = math.cos(ta + math.pi * 0.5), math.sin(ta + math.pi * 0.5)
    npc = rng.randint(2, 3)
    th = wa - ws * 0.5 - rng.uniform(0.0, 0.3)
    for _ in range(npc):
        span = (ws + 0.7) / npc * rng.uniform(0.55, 0.95)
        nn = 11
        rim = []
        for i in range(nn + 1):
            t = th + span * i / nn
            r = edge(t) * rng.uniform(1.00, 1.04)
            lx, ly = r * math.cos(t), r * math.sin(t) * sq
            rim.append((x + lx * ca - ly * sa, y + lx * sa + ly * ca))
        drag(cv, rim, max(0.52, hw * 0.10), max(0.44, hw * 0.07),
             I("zhong"), I("jiao"), rng.uniform(0.52, 0.82) * k, d0=0.42,
             d1=0.92, feather=0.80, core=0.16,
             dry=0.25 + air["dry"] * 0.5, dry_ph=rng.uniform(0, TAU))
        th += span + rng.uniform(0.25, 0.75)
    # 木刺 — the splinters
    m = len(pts) - 1
    for _ in range(rng.randint(1, 2)):
        u = rng.uniform(-0.70, 0.70)
        sx, sy = _perp(pts, 0, u * hw)
        j = min(m, rng.randint(2, 5))
        seg = [_perp(pts, q, u * hws[q] * rng.uniform(0.92, 1.08))
               for q in range(0, j + 1)]
        if len(seg) < 3:
            continue
        drag(cv, seg, rng.uniform(0.44, 0.78), 0.34,
             I("zhong"), I("nong"), rng.uniform(0.34, 0.58) * k,
             d0=0.36, d1=0.86, feather=0.75, core=0.20,
             dry=0.45 + air["dry"], dry_k=3.0, dry_ph=rng.uniform(0, TAU))


def cun(cv, rng, pts, hws, air, n=14, k=1.0, old=1.0):
    """皴 — the bark of an old plum: short dry strokes *inside* the mass.

    Mostly along the grain with a few across it, all of them placed on the
    node's own normal and all of them short, because a long one stops being
    texture and becomes a second branch.

    Both sets are drawn in 浓墨 at half a pixel of half-width, which looks
    like a contradiction and is the only setting that works. The first
    version used 淡墨 at the same width and 古干 came out as a smooth pale
    pipe with fifty-nine invisible strokes on it: at sub-pixel width `cov`
    is already a fraction, `dry` takes another bite out of it, and the
    *product* of two small numbers against a ramp that starts near the
    wood's own tone is nothing at all. Bark is a hairline of dark ink, not
    a wide stroke of pale ink — 皴 is 骨法, not 墨法.
    """
    m = len(pts) - 1
    if m < 3:
        return
    for _ in range(n):
        i = rng.randrange(1, m)
        hw = hws[i]
        if hw < 2.0:
            continue
        u = rng.uniform(-0.80, 0.80)
        x, y = _perp(pts, i, u * hw)
        ta = gb.tangent(pts, i)
        # 横皴 — and old wood wants a good many of them. The strokes that
        # run *with* the grain agree with the shape of the mass and with
        # `sweep`'s own shading, so however many of them there are they
        # leave the long smooth reading intact; it is the ones that cut
        # across that break it, the way the rings on a cut log do. A third
        # of them, and long enough to cross a real part of the width.
        across = rng.random() < (0.28 + 0.22 * old)
        if across:
            # inward, from wherever on the width it started — a 横皴 aimed
            # outward from the far flank leaves the wood altogether
            sgn = (-1.0 if u > 0.0 else 1.0) if abs(u) > 0.30 \
                else rng.choice((1.0, -1.0))
            a = ta + math.pi * 0.5 * sgn + rng.gauss(0, 0.22)
            L = (3.0 + hw * 0.80) * rng.uniform(0.42, 1.05)
        else:
            # 擦 has a *size*, and it is not a fraction of the branch. Every
            # length here used to be `hw × something`, which is right for
            # the strokes that cross the width and wrong for the ones that
            # run along it: on a 13 px half-width branch it made all forty
            # marks between five and nineteen pixels long at sub-pixel
            # width, so 折枝 was textured on paper and smooth to the eye —
            # the texture was *drawn* and then could not be *seen*. A
            # brush-hair streak is as long as the hand's gesture; it gets
            # a floor in pixels and only a mild slope in the width.
            a = ta + rng.gauss(0, 0.30)
            L = (6.0 + hw * 0.95) * rng.uniform(0.60, 1.70) * (0.6 + 0.6 * old)
        # 皴在干内 — and *nothing above enforced it*, which is the one fault
        # in this function that read as western rather than merely weak. A
        # stroke starts up to 0.8 of the half-width off the centreline and
        # then runs straight; at 0.30 radians off the tangent a 95 px grain
        # stroke drifts 28 px sideways, so on a 42 px trunk a good many of
        # them left the silhouette and went on into the paper. Long, pale,
        # straight lines radiating out of a form are hatching — the
        # draughtsman's shading, not bark. Clamp the far end back inside the
        # mass instead of shortening every stroke: the ones that would have
        # escaped get cut where they would have crossed the edge, which is
        # also what the hand does when it feels the brush run off the wood.
        lat = math.sin(a - ta)
        if abs(lat) > 1e-6:
            lim = (0.92 * hw * (1.0 if lat > 0.0 else -1.0) - u * hw) / lat
            if lim < L:
                L = max(hw * 0.14, lim)
        x1, y1 = x + math.cos(a) * L, y + math.sin(a) * L
        # 一笔有一笔的轻重 — and the weights have to differ by a *lot*. Fifty
        # hairlines all drawn at 0.6–1.0 px and a third of a value apart is
        # an even stubble, which is a texture map; the hand puts down three
        # that bite and a dozen that barely touch. So the width spans a
        # factor of three and the strength a factor of five, correlated:
        # the fat ones are the dark ones, because both come from pressing.
        # ...and every one of them has to come off the paper at the end. At
        # 0.62 → 0.46 the two ends are the same width, so a mark that is
        # dark enough to see at all is a blunt bar — a scratch, or a little
        # twig lying on the branch. 收笔 is not decoration here; it is what
        # keeps a bark stroke from being read as an object.
        g = rng.random() ** 1.7                  # mostly light, a few heavy
        drag(cv, _arc(((x, y), ((x + x1) * 0.5 + rng.gauss(0, L * 0.13),
                                (y + y1) * 0.5 + rng.gauss(0, L * 0.13)),
                       (x1, y1)), 5),
             0.62 + 1.28 * g, 0.26 + 0.42 * g,
             I("zhong"), I("jiao"), (0.22 + 0.64 * g) * k,
             d0=0.35, d1=0.85, feather=0.70, core=0.18,
             dry=0.35 + air["dry"], dry_ph=rng.uniform(0, TAU))
    # 纹 — and a few long ones with the grain. These are what actually
    # break the tube: short dashes alone read as stubble on a smooth pipe,
    # because the thing that says "pipe" is the unbroken length.
    for _ in range(max(2, n // 3)):
        i = rng.randrange(1, max(2, int(m * 0.80)))
        hw = hws[i]
        if hw < 3.0:
            continue
        j = min(m, i + rng.randint(int(m * 0.08) + 2, int(m * 0.34) + 4))
        if j - i < 3:
            continue
        d = rng.uniform(-0.75, 0.75)
        seg = [_perp(pts, q, d * hws[q] * rng.uniform(0.94, 1.06))
               for q in range(i, j + 1)]
        drag(cv, seg, rng.uniform(0.62, 1.15), rng.uniform(0.44, 0.78),
             I("zhong"), I("nong"), rng.uniform(0.26, 0.54) * k,
             d0=0.30, d1=0.80, feather=0.80, core=0.20,
             dry=0.55 + air["dry"], dry_k=3.4, dry_ph=rng.uniform(0, TAU))


def blot(cv, rng, res, pts, hws, air, k=1.0, old=0.0, n=4, side=0.0):
    """墨色斑驳 — the ink in patches, because a gradient is an airbrush.

    `sweep` shades a stroke as a smooth function of the across-stroke
    coordinate. That is the right model for one wet touch of a round brush
    and the wrong one for a trunk: it makes the value a surface property,
    so the mass comes out as a lit cylinder — light down one flank, dark
    down the other, the transition perfectly even — and a lit cylinder is
    the whole vocabulary of western rendering. No amount of bark drawn on
    top of it helps, because the bark is line and the fault is tone.

    A 积墨 mass is instead built in layers that each dry at their own edge,
    so its value moves in *patches* whose boundaries run with the grain and
    have nothing to do with the silhouette. That is what this lays: a few
    irregular stains, elongated along the tangent (`rot=ta, squash<1`),
    half of them dark and half of them paper.

    The pale half is the half that matters. Dark blots alone only make a
    dirtier cylinder — it takes 留白 *inside* the mass to destroy the
    reading, and the only white this family has is the sheet, so they are
    `lift`s. Which is why this runs **before** the bark and the moss: a
    lift restores the snapshot and would wipe out any line drawn first.
    """
    m = len(pts) - 1
    if m < 3:
        return
    # ...spread along the length, not drawn uniformly. Independent draws
    # put three of the ten within a few nodes of each other, and three
    # overlapping stains with hard rims on the same flank read as a heap of
    # pebbles lying against the trunk — the same clustering failure as
    # `flyback`'s beads, arriving by the same route. 积墨 patches are
    # separated by plain wood.
    sep = max(2, int(m / (n + 1.5)))
    used = []
    for _ in range(n):
        for _try in range(6):
            i = rng.randrange(1, m)
            if all(abs(i - j) >= sep for j in used):
                break
        used.append(i)
        hw = hws[i]
        if hw < 2.8:
            continue
        ta = gb.tangent(pts, i)
        pale = rng.random() < 0.44
        # A pale patch belongs on the starved flank. Dropped at random it
        # punches a hole in the loaded flank's 浓墨 instead, and a pale hole
        # in a dark band is not 斑驳 — it is a bubble, which is how the
        # first version of this read: a row of beads down the trunk.
        if pale and side:
            u = rng.uniform(-0.80, 0.22) * (-1.0 if side >= 0.0 else 1.0)
        else:
            u = rng.uniform(-0.62, 0.62)
        x, y = _perp(pts, i, u * hw)
        R = hw * (rng.uniform(0.30, 0.68) if pale else rng.uniform(0.42, 1.10))
        # A pale one has to be a *streak*, and a lumpy one. At squash 0.42
        # with small harmonics the lift comes out a clean bright oval lying
        # on the wood, which is an egg; the dark ones can be rounder because
        # nothing bright is ever mistaken for an object sitting on top.
        sq = rng.uniform(0.11, 0.25) if pale else rng.uniform(0.16, 0.42)
        k1, k2 = ((rng.uniform(0.22, 0.46), rng.uniform(0.12, 0.30)) if pale
                  else (rng.uniform(0.10, 0.28), rng.uniform(0.05, 0.18)))
        p1, p2 = rng.uniform(0, TAU), rng.uniform(0, TAU)

        def rad(th, R=R, k1=k1, k2=k2, p1=p1, p2=p2):
            return R * (1.0 + k1 * math.cos(th * 2.0 + p1)
                        + k2 * math.cos(th * 3.0 + p2))

        if pale:
            # and the edge has to be *crisp*. The first attempt softened it
            # — feather `2 + hw·0.22`, nine pixels of transition on a trunk
            # — on the theory that a legible boundary inside the mass would
            # read as an object (crab's §3). It does not: a wide soft pale
            # oval inside a long form reads as a **highlight**, so the pass
            # meant to destroy the airbrushed cylinder was quietly shading
            # one. 墨 does the opposite of airbrushing: it pools and dries
            # from the edge inward, so a patch of 积墨 has a hard rim and a
            # flat middle — 水渍痕. Low alpha, hard edge.
            res.lift(cv, x, y, rad, rot=ta, squash=sq,
                     alpha=rng.uniform(0.16, 0.34) * (0.6 + 0.5 * old),
                     feather=1.0 + hw * 0.03)
        else:
            # ...and the dark ones dry the same way, which is why the shade
            # here runs *up* toward the rim.
            def tone(u, th):
                return (0.66 + 0.34 * u, clamp01(1.0 - u ** 6.0))

            touch(cv, x, y, rad, tone, I("zhong"), I("nong"),
                  alpha=rng.uniform(0.18, 0.40) * k, feather=1.1 + hw * 0.03,
                  squash=sq, rot=ta, na=96, nr=10)


def flyback(cv, rng, res, pts, hws, air, k=1.0, n=5, side=0.0):
    """飞白 — the paper a dry brush skips, as 留白 rather than as pale paint.

    `sweep` has a `dry` of its own and on a big mass it is the wrong shape
    twice over. Its streaks are `sin(sl·dry_k·π)²` — a *periodic* function
    of the across-stroke coordinate, so they come out as corrugation, an
    even comb of stripes — and their depth is `dry·tg²`, which is zero at
    the root of the stroke and only bites near the tip. That is exactly
    right for a 兰叶, which is what it was written for: a leaf dries as the
    brush runs out. An old trunk is dry *from the first inch*, and in
    streaks of no particular spacing that wander, merge, stop and start.
    The pale flank of the trunk was therefore a smooth even wash — the
    single most western thing left on the sheet, because a smooth tonal
    gradient across a long form is what a pencil does and what a brush
    never does.

    So the streaks are laid here instead, as what they physically are: the
    sheet, where the hairs did not touch it. Each one is a chain of
    elongated `lift`s following the grain, a few per cent of the width
    across, with its own alpha, its own lateral wander, and a death
    somewhere short of where it started.
    """
    m = len(pts) - 1
    if m < 4 or not hws or max(hws) < 4.0:
        return
    pp, hh = _densify(pts, hws, 9.0)
    m = len(pp) - 1
    for _ in range(n):
        i = rng.randrange(0, max(1, int(m * 0.82)))
        # ...and a streak is a few tens of pixels long, full stop. The
        # length used to be a *fraction of the path* — up to 30 % of a
        # thousand-pixel branch, which at 9 px per node is a three-hundred
        # pixel hairline of bare paper lying parallel to the edge with only
        # three pixels of wander in it. That is not 飞白, it is a specular
        # highlight, and putting six of them down the starved flank drew
        # the chrome rim of the pipe this function exists to destroy.
        j = min(m, i + rng.randint(2, max(3, min(int(m * 0.30), 9))))
        # 偏于淡处 — mostly down the flank the brush starved on, which is
        # the one `side` tilted away from, because that is where the wash
        # it is breaking up actually is.
        u0 = rng.uniform(0.12, 0.92) * (-1.0 if side >= 0.0 else 1.0)
        if rng.random() < 0.3:
            u0 = -u0
        wf = rng.uniform(0.018, 0.105)
        al = rng.uniform(0.22, 0.58)
        ph = rng.uniform(0, TAU)
        # ...and they must not stay parallel. Nine streaks each holding its
        # own lateral station came out as corrugated iron — a veneer of
        # wood grain, which is a *texture* and not a brush running out of
        # ink. The hairs of a real brush splay, so the streaks converge,
        # cross and merge; one wandering sine per streak, at a quarter of
        # the width, is enough to stop the set reading as a comb.
        fr = rng.uniform(0.7, 2.4)
        dr = rng.uniform(-0.30, 0.30)
        q = i
        while q < j:
            r = min(j, q + rng.randint(2, 3))
            mid = (q + r) // 2
            hw = hh[mid]
            if hw < 3.0:
                q = r
                continue
            t = (mid - i) / max(1, j - i)
            u = u0 + 0.26 * math.sin(t * fr * TAU + ph) + dr * t
            if u > 0.95:
                u = 0.95
            elif u < -0.95:
                u = -0.95
            x, y = _perp(pp, mid, u * hw)
            ta = gb.tangent(pp, mid)
            L = max(2.0, math.hypot(pp[r][0] - pp[q][0],
                                    pp[r][1] - pp[q][1]) * 0.80 + 2.0)
            # ...plus a pixel, absolutely. `hw * wf` alone gave a 13 px
            # branch streaks 0.24–1.4 px wide: 飞白 drawn at a width the
            # sheet cannot show. A gap between hairs is a gap between
            # hairs whatever the brush is doing.
            wdt = max(0.45, hw * wf + 0.85)
            sq = max(0.03, wdt / L)
            p1 = rng.uniform(0, TAU)

            def rad(th, L=L, p1=p1):
                return L * (1.0 + 0.16 * math.cos(th * 2.0 + p1))

            # 意到笔不到 — the streak fades out rather than stopping
            a = al * (0.45 + 0.55 * math.sin(math.pi * min(1.0, t * 1.25)))
            res.lift(cv, x, y, rad, rot=ta, squash=sq, alpha=a,
                     feather=0.9 + hw * 0.015, na=48)
            q = r


def moss(cv, rng, pts, hws, k=1.0, n=5):
    """苔点 — moss. Clusters of small dark dots on the old wood, never
    evenly spread: 点苔 goes where the wood turns and where it is wettest."""
    m = len(pts) - 1
    if m < 3:
        return
    for _ in range(n):
        i = rng.randrange(1, m)
        hw = hws[i]
        if hw < 1.6:
            continue
        cx, cy = _perp(pts, i, rng.uniform(-0.7, 0.7) * hw)
        for _ in range(rng.randint(2, 6)):
            disc(cv, cx + rng.gauss(0, hw * 0.45), cy + rng.gauss(0, hw * 0.5),
                 rng.uniform(0.7, 1.7), I("jiao"), rng.uniform(0.30, 0.70),
                 soft=0.4)


def scar(cv, rng, res, pts, hws, air, k=1.0):
    """洞 — a hollow in the trunk. 「干要有洞，洞要透气」.

    Lifted out of the wood rather than painted into it, for the same reason
    the blossoms are: a hollow is a hole, and a hole is paper. Then a broken
    dark line round part of it, without which the blank reads as a flaw in
    the painting instead of a wound in the tree.
    """
    m = len(pts) - 1
    if m < 6:
        return
    i = rng.randrange(max(1, int(m * 0.08)), max(2, int(m * 0.80)))
    hw = hws[i]
    if hw < 6.0:
        return                    # below this a hollow reads as a dust speck
    ta = gb.tangent(pts, i)
    x, y = _perp(pts, i, rng.uniform(-0.26, 0.26) * hw)
    R = hw * rng.uniform(0.46, 0.74)
    p1, p2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
    # The wobble used to be 0.22 on the second harmonic, which on a
    # squashed frame produced a dimple at each end and the hollow came out
    # heart-shaped. A hollow is a lopsided oval; the lopsidedness belongs
    # on the odd harmonics.
    sq = rng.uniform(0.52, 0.76)
    p3 = rng.uniform(0, TAU)

    def rad(th):
        # ...and a good deal more lopsided than it was. At 0.11 and 0.07 on
        # the first and third harmonics this is an oval to within a tenth,
        # and a pale oval of even tone on a dark branch is read by the eye
        # as a *convex* thing lying on it — an egg — not as a cavity. The
        # fifth harmonic is what makes it a torn opening.
        return R * (1.0 + 0.17 * math.cos(th + p1)
                    + 0.12 * math.cos(th * 3.0 + p2)
                    + 0.07 * math.cos(th * 5.0 + p3))

    # not a clean lift: a hollow in wood keeps some tone in it, and a fully
    # restored one reads as a hole punched through the paper. Less of one
    # than before, too — what tells the eye this is a hole is the drawn rim
    # below, not the brightness of the blank, and the brighter the blank the
    # more it looks lit.
    res.lift(cv, x, y, rad, rot=ta, squash=sq,
             alpha=rng.uniform(0.30, 0.48), feather=1.3)
    # 勾洞 — most of the rim, in 浓墨, broken. The first version drew a
    # quarter of it at hairline weight and the blank it was supposed to
    # enclose stayed unenclosed; 「洞要透气」 still wants the hole *drawn*.
    a0 = rng.uniform(0, TAU)
    span = rng.uniform(2.4, 3.8)
    ca, sa = math.cos(ta), math.sin(ta)
    arc = []
    nn = 19
    for j in range(nn + 1):
        th = a0 + span * j / nn
        r = rad(th) * rng.uniform(1.00, 1.05)
        lx, ly = r * math.cos(th), r * math.sin(th) * sq
        arc.append((x + lx * ca - ly * sa, y + lx * sa + ly * ca))
    drag(cv, arc, max(0.7, hw * 0.085), max(0.55, hw * 0.055),
         I("nong"), I("jiao"), 0.80 * k, d0=0.42, d1=0.95, feather=0.80,
         core=0.12, dry=0.8 + air["dry"], dry_ph=rng.uniform(0, TAU))


def _densify(pts, hws, dmax):
    """Resample a path so no two points are more than `dmax` apart.

    The snow band needs this and nothing else does: `wood_path` lays its
    nodes 20 px apart on a trunk, and a band built one lift per node came
    out as a row of white beads down the middle of the wood — the lifts
    were 7 px across and 21 px apart. Overlap or spots; there is no third
    option.
    """
    out, ow = [pts[0]], [hws[0]]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        m = max(1, int(math.hypot(x1 - x0, y1 - y0) / dmax))
        for j in range(1, m + 1):
            t = j / m
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            ow.append(hws[i - 1] + (hws[i] - hws[i - 1]) * t)
    return out, ow


def snow_on(cv, rng, res, pts, hws, up, keep, k=1.0):
    """积雪 — snow on a bough, which is 留白 again: the wood the snow hides.

    A band of overlapping lifts along the upper flank, with a thin cool
    line drawn under its edge so the snow has a bottom. Two things it has
    to get right, both found by looking at the first attempt — which was a
    row of white polka dots down the middle of the trunk:

      * the lifts must overlap. Spaced wider than their own radius they
        read as spots, and spots on a branch are lichen, not snow.
      * snow lies on wood that is near horizontal and slides off wood that
        is near vertical, so the depth is a function of the slope and most
        of a rising trunk gets none at all.

    Any lift that would land on a blossom is skipped: snow may cover wood,
    it may not erase a flower that is in front of it.
    """
    pts, hws = _densify(pts, hws, 1.6)
    n = len(pts) - 1
    if n < 4:
        return
    # The drift has to be *lumpy*. One slow sine down the length gave a
    # single smooth white lens along the trunk, which does not read as snow
    # at all — it reads as a specular highlight on a plastic pipe. Snow
    # gathers in patches, with bare wood between them, so the depth gets
    # two periods and a threshold it can fall below.
    ph = rng.uniform(0, TAU)
    ph2 = rng.uniform(0, TAU)
    f1 = rng.uniform(2.4, 4.2) * TAU / max(1, n)
    f2 = rng.uniform(6.0, 10.0) * TAU / max(1, n)
    edge = []
    for i in range(1, n):
        hw = hws[i]
        if hw < 3.0:
            # A twig thinner than this would want a drift one pixel wide,
            # which comes out as a dotted line — a zip of white beads down
            # the branch, which is how the second attempt failed too.
            edge.append(None)
            continue
        ta = gb.tangent(pts, i)
        slope = abs(math.cos(ta)) ** 1.4          # steep wood sheds it
        dep = (0.56 + 0.40 * math.sin(i * f1 + ph)
               + 0.20 * math.sin(i * f2 + ph2)) * slope
        if dep < 0.38:
            edge.append(None)
            continue
        x, y = _perp(pts, i, up * hw * (0.66 - 0.26 * dep))
        r = hw * 0.74 * dep
        if r < 1.4:
            edge.append(None)
            continue
        for fx, fy, fr in keep:
            if (fx - x) ** 2 + (fy - y) ** 2 < (fr + r) ** 2:
                r = 0.0
                break
        if r <= 0.0:
            edge.append(None)
            continue
        res.lift(cv, x, y, (lambda rr: (lambda th: rr))(r),
                 alpha=0.94, feather=1.0)
        edge.append(_perp(pts, i, up * (hw * (1.0 - 0.52 * dep) - r * 0.82)))
    # 雪痕 — the under-edge. Lifted paper with nothing below it is a hole;
    # one pale cool line is what makes it a drift sitting on the bough.
    run = []
    for p in edge + [None]:
        if p is None:
            if len(run) > 3:
                drag(cv, run, 0.7, 0.5, I("qing"), I("dan"), 0.30 * k,
                     d0=0.3, d1=0.7, feather=1.1, core=0.3, dry=0.5,
                     dry_ph=rng.uniform(0, TAU))
            run = []
        else:
            run.append(p)


# --------------------------------------------------------------------------
# 花 — the blossom, and the reason this module is not a 工笔 module.
#
# 「花分五出」. A 圈花 plum blossom is five circles of line, laid on a ring
# round a common centre, each one left open where it faces the middle —
# because the petals of a plum do not meet at the centre, the stamens come
# up between them. Drawn closed it is a flower made of five bubbles; drawn
# open it is a flower.
#
# What makes it read:
#
#   * every flower a different size (R varies by a fifth) and every petal
#     within a flower a different size again, with its angle jittered off
#     the exact fifth. Five equal circles on an exact pentagon is a logo.
#   * 有正有背 — 正面 is the full ring, 侧面 is three or four petals over
#     half a turn with the calyx in plain sight, 背面 is the calyx itself,
#     a five-pointed dark star with the petals showing behind it. A sheet
#     of face-on flowers is a plate from a botany book.
#   * 蒂如丁香 — the calyx, and then a stalk to the wood. This is the
#     single highest-yield mark on the sheet: without it a pale ring of
#     line floats somewhere in front of the branch, and with it the flower
#     is *on* the branch.
#   * 蕊如鼠须 — the stamens, with a dark anther on each. They are the only
#     真浓墨 in the flower, which is what gives it an eye.
#   * 前实后虚 — the ones behind are paler and lift less of what is under
#     them, so a cluster has a front.
# --------------------------------------------------------------------------

def _petals(rng, R, view, attach):
    """The petal discs as (ox, oy, pr, angle) in the flower's own frame."""
    out = []
    if view == "侧":
        n = rng.randint(3, 4)
        mid = attach + math.pi + rng.gauss(0, 0.20)
        span = rng.uniform(1.9, 2.6)
        for i in range(n):
            a = mid + span * ((i / max(1, n - 1)) - 0.5) + rng.gauss(0, 0.10)
            pr = R * rng.uniform(0.34, 0.42)
            o = R * rng.uniform(0.54, 0.62)
            out.append((math.cos(a) * o, math.sin(a) * o, pr, a))
        return out
    # The petals have to be very nearly *tangent*. Adjacent centres sit
    # 2*o*sin(36 deg) = 1.18*o apart, so pr near 0.59*o is the tangent case
    # and anything much above it buries a third of each circle inside its
    # neighbours. The first version used pr = 0.45 R against o = 0.56 R —
    # 37 % overlap — and five near-closed arcs overlapping that hard do not
    # read as a flower, they read as a flower-of-life diagram.
    n = 5
    rot = rng.uniform(0, TAU)
    for i in range(n):
        a = rot + i * TAU / n + rng.gauss(0, 0.11)
        pr = R * rng.uniform(0.35, 0.41)
        o = R * rng.uniform(0.58, 0.64)
        out.append((math.cos(a) * o, math.sin(a) * o, pr, a))
    return out


def _union_rad(pets, pad=0.0):
    """The outer boundary of a union of circles, as a polar radius.

    Exact, and three lines: along direction th the far intersection with
    circle (c, p) is c·u + sqrt(p² - |c|² + (c·u)²), and the union's
    boundary is the largest of those. The flower's centre is outside every
    petal — the petals ring it — so the region this bounds includes the
    middle of the flower, which is correct: 花心 is in front of the branch
    too.
    """
    def rad(th):
        ux, uy = math.cos(th), math.sin(th)
        best = 0.0
        for ox, oy, pr, _a in pets:
            b = ox * ux + oy * uy
            disc_ = b * b - (ox * ox + oy * oy - pr * pr)
            if disc_ <= 0.0:
                continue
            t = b + math.sqrt(disc_)
            if t > best:
                best = t
        return best + pad
    return rad


def _ring(cv, rng, cx, cy, ox, oy, pr, a, lo, hi, w, air, alpha=0.92,
          gap=0.86, k=1.0):
    """圈 — one petal, as **two** strokes of the brush: 两笔一瓣.

    The first version drew it as one arc of a circle with the width swelling
    smoothly in the middle, and that is a *pen* line: constant curvature,
    constant weight, both ends identical and blunt, the same tone from end
    to end. Five of them is a stencil. Rendered at 60 px the fault was
    unmistakable, and all four of its causes are things a brush cannot do:

      * **a circle.** A hand drawing a petal does not describe a circle, and
        it certainly does not describe five of them the same. Each petal
        here gets its own low-harmonic radius, its own squash and its own
        tilt, so no two are the same shape.
      * **one continuous stroke.** 圈花 is painted 两笔一瓣 — two strokes,
        each starting at the open side near the stamens and curving out to
        meet (or just miss) the other at the far rim. The junction is the
        mark that says "brush": it is a little off, sometimes crossed,
        sometimes short of touching.
      * **even weight.** A brush stroke begins with a press (起笔, 顿) and
        ends with a lift (收笔), so it runs fat-to-thin, not thin-fat-thin.
      * **even tone.** The ink is heaviest where the brush landed and
        starves toward the lift, and at this width a little 飞白 breaks the
        tail outright: 意到笔不到.
    """
    # the petal's own outline — not a circle, and not shared with its
    # neighbours
    ph1, ph2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
    k1 = rng.uniform(0.05, 0.12)
    k2 = rng.uniform(0.03, 0.08)
    # ...but only a little squash. `tilt` is free, so a squash of 0.84 at a
    # random tilt pushes one flank of the petal a sixth of its radius into
    # whichever neighbour happens to be there, and five petals deformed
    # that way stop being a ring of five and become a knot.
    sq = rng.uniform(0.90, 1.11)
    tilt = rng.uniform(0, TAU)
    ct, stl = math.cos(tilt), math.sin(tilt)

    def P(th, jit=0.0):
        r = pr * (1.0 + k1 * math.cos(th * 2.0 + ph1)
                  + k2 * math.cos(th * 3.0 + ph2) + jit)
        lx, ly = math.cos(th) * r, math.sin(th) * r * sq
        return (cx + ox + lx * ct - ly * stl, cy + oy + lx * stl + ly * ct)

    a0 = a + math.pi + gap * 0.5 * rng.uniform(0.85, 1.15)
    a1 = a + math.pi + TAU - gap * 0.5 * rng.uniform(0.85, 1.15)
    # where the two strokes meet, and by how much they miss it. Positive
    # laps them over each other (the commoner case in a real 圈), negative
    # leaves the rim open.
    mid = (a0 + a1) * 0.5 + rng.gauss(0, 0.17)
    over = rng.uniform(-0.05, 0.22)
    n = 9
    # ...and 两笔 only while two strokes can be *seen*. Below about seven
    # pixels of petal radius the two halves are each four pixels long with
    # a press at one end, they overlap at the junction, and the flower
    # comes out as a tangle of dark scribble rather than as the clean small
    # blossom a distant one should be — which is a worse fault than the one
    # this was fixing, because at that size nobody can see the junction
    # anyway. The taper and the broken tone still do the work.
    if pr < 7.0:
        runs = ((a0, a1 + (0.0 if a1 > a0 else TAU)),)
    else:
        runs = ((a0, mid + over), (a1, mid - over))
    for ta, tb in runs:
        # 笔断 — the stroke that stops. Rarely, and never by much: this is
        # half a petal, and taking a third off it leaves a hook rather than
        # a gap. 意到笔不到 is a break of two or three pixels, not a
        # missing quarter of the flower.
        if rng.random() < 0.06:
            tb = ta + (tb - ta) * rng.uniform(0.70, 0.88)
        p0 = rng.uniform(0.92, 1.28)          # 起笔 — the press
        p1 = rng.uniform(0.22, 0.46)          # 收笔 — the lift
        d0 = rng.uniform(0.76, 0.98)
        d1 = rng.uniform(0.24, 0.54)
        qh = rng.uniform(0, TAU)
        pts, hws, dens = [], [], []
        for i in range(n + 1):
            t = i / n
            th = ta + (tb - ta) * t
            pts.append(P(th, rng.gauss(0, 0.014)))
            sw = (p0 + (p1 - p0) * t ** 0.78) * (
                1.0 + 0.11 * math.sin(t * 4.3 + qh))
            hws.append(max(0.34, w * sw))
            dens.append(clamp01(d0 + (d1 - d0) * t ** 0.85
                                + rng.uniform(-0.05, 0.05)))
        sweep(cv, pts, hws, dens, lo, hi, alpha * k,
              feather=0.70 * air["wet"] ** 0.25, core=0.26,
              dry=0.22 + air["dry"] * 0.55, dry_k=2.0,
              dry_ph=rng.uniform(0, TAU))


def _wash(cv, rng, cx, cy, ox, oy, pr, a, lo, hi, air, alpha, back):
    """点 — one petal as a single touch: dark at the rim where a wet mark
    dries (水渍痕), pale in the body, with a nick of bare paper near the
    base. 点梅 is this five times and nothing else."""
    ph = rng.uniform(0, TAU)
    wob = rng.uniform(0.05, 0.10)

    def rad(th):
        return pr * (1.0 + wob * math.cos(th * 2.0 + ph)
                     + wob * 0.55 * math.cos(th * 3.0 - ph * 1.3))

    nx, ny = -math.cos(a) * 0.34, -math.sin(a) * 0.34
    base = 0.18 + 0.18 * rng.random() - 0.12 * back
    rim = 0.62 + 0.20 * rng.random()

    def tone(u, th):
        sh = base + rim * u ** 2.4
        px, py = u * math.cos(th), u * math.sin(th)
        dd = (px - nx) ** 2 + (py - ny) ** 2
        cov = 1.0 - 0.46 * math.exp(-dd * 7.0) * (1.0 - 0.5 * back)
        return sh, cov

    touch(cv, cx + ox, cy + oy, rad, tone, lo, hi, alpha=alpha,
          feather=1.0 * air["wet"] ** 0.35, na=120, nr=12)


def _stamens(cv, rng, cx, cy, R, ink, air, n, lo, hi, k=1.0):
    """蕊如鼠须 — whiskers, each with its anther."""
    rot = rng.uniform(0, TAU)
    for i in range(n):
        a = rot + (lo + (hi - lo) * (i + rng.uniform(0.2, 0.8)) / n)
        # Whiskers, and they stay *inside* the flower. The first version
        # ran them out to 0.92 R, which put every anther on top of the ring
        # and turned each blossom into a black asterisk — at 15 px across,
        # the strongest mark on the sheet was the one that should be the
        # faintest.
        # ...and they fade on a small one. At 16 px across, five whiskers
        # with a dark anther on each end are not stamens: they are a black
        # asterisk sitting on a pale ring, and a sheet of them reads as a
        # sheet of insects. Detail has to be affordable at the size it is
        # drawn — 远花无蕊.
        sm = min(1.0, R / 17.0)
        L = R * rng.uniform(0.26, 0.54)
        x1, y1 = cx + math.cos(a) * L, cy + math.sin(a) * L
        bow = rng.gauss(0, L * 0.13)
        mx = (cx + x1) * 0.5 - math.sin(a) * bow
        my = (cy + y1) * 0.5 + math.cos(a) * bow
        drag(cv, _arc(((cx, cy), (mx, my), (x1, y1)), 5),
             max(0.30, R * 0.034), max(0.26, R * 0.024),
             dilute(I("zhong"), 0.55 + 0.45 * sm), ink,
             rng.uniform(0.34, 0.56) * k * (0.40 + 0.60 * sm),
             d0=0.40, d1=0.88, feather=0.62, core=0.10)
        disc(cv, x1, y1, max(0.42, R * rng.uniform(0.032, 0.050)), ink,
             rng.uniform(0.40, 0.70) * k * (0.34 + 0.66 * sm), soft=0.35)
    if R > 12.0:
        # Only on a flower big enough to have a middle. At thumbnail size
        # this dot was the darkest thing in the blossom and every flower on
        # the sheet looked like it was staring.
        disc(cv, cx, cy, R * 0.055, ink, 0.30 * k, soft=0.5)


def _calyx(cv, rng, x, y, ang, s, ink, air, k=1.0, n=3, spread=0.46):
    """蒂如丁香 — the clove. Three (or five, from behind) short dark strokes
    at the flower's base, which is the mark that fastens it to the wood."""
    for i in range(n):
        a = ang + (i - (n - 1) * 0.5) * spread * rng.uniform(0.85, 1.15)
        L = s * rng.uniform(0.75, 1.20)
        x1, y1 = x + math.cos(a) * L, y + math.sin(a) * L
        bow = rng.gauss(0, L * 0.16)
        drag(cv, _arc(((x, y),
                       ((x + x1) * 0.5 - math.sin(a) * bow,
                        (y + y1) * 0.5 + math.cos(a) * bow),
                       (x1, y1)), 5),
             max(0.42, s * 0.26), max(0.34, s * 0.12),
             I("zhong"), ink, rng.uniform(0.70, 0.95) * k,
             d0=0.55, d1=0.95, feather=0.68, core=0.12)


def _stalk(cv, rng, x0, y0, x1, y1, w, air, k=1.0):
    """花柄 — the short stalk from the wood to the calyx."""
    bow = rng.gauss(0, math.hypot(x1 - x0, y1 - y0) * 0.16)
    a = math.atan2(y1 - y0, x1 - x0)
    pts = _arc(((x0, y0),
                ((x0 + x1) * 0.5 - math.sin(a) * bow,
                 (y0 + y1) * 0.5 + math.cos(a) * bow),
                (x1, y1)), 7)
    drag(cv, pts, w * 1.2, w * 0.85, I("zhong"), I("jiao"), 0.82 * k,
         d0=0.60, d1=0.95, feather=0.72, core=0.14)


def blossom(cv, rng, res, x, y, R, st, air, view="正", attach=0.0,
            back=0.0, hang=None, k=1.0):
    """One flower. Returns (x, y, R) for the occupancy map.

    `attach` is the direction from the flower's centre toward the wood it
    grows on; `hang` is that point on the wood, if there is one. `back` in
    [0, 1] pushes it behind the front of its cluster — paler line, softer
    edge, fewer stamens, and a lift that leaves a ghost of what is under it.
    """
    b = clamp01(back)
    pets = _petals(rng, R, view, attach)
    urad = _union_rad(pets, pad=max(0.5, R * 0.045))
    # 留白 — the flower is the paper. Everything already on the sheet is
    # lifted back out of its outline before a single petal is drawn, which
    # is the only way a pale ring can sit in front of a dark branch when
    # all compositing is by multiply.
    res.lift(cv, x, y, urad, alpha=0.96 - 0.30 * b,
             feather=0.85 * air["wet"] ** 0.3)

    r_lo, r_hi = rgb(st["ring"][0]), rgb(st["ring"][1])
    ink = rgb(st["heart"])
    if b > 0.0:
        r_lo = dilute(r_lo, 1.0 - 0.30 * b)
        r_hi = dilute(r_hi, 1.0 - 0.34 * b)

    if st["halo"] > 0.0:
        # 衬托 — a breath of 淡墨 outside the rim, so untouched paper reads
        # as a white petal instead of as a gap in the drawing.
        def hrad(th):
            return urad(th) * rng.uniform(1.30, 1.42)

        def htone(u, _th):
            return 0.12 + 0.10 * u, clamp01((u - 0.70) * 3.4) * 0.55

        touch(cv, x, y, hrad, htone, I("qing"), I("dan"),
              alpha=0.30 * st["halo"] * (1.0 - 0.4 * b), feather=2.2,
              na=96, nr=10)

    if st["fill"] is not None:
        f_lo, f_hi = rgb(st["fill"][0]), rgb(st["fill"][1])
        if b > 0.0:
            f_lo, f_hi = dilute(f_lo, 1.0 - 0.3 * b), dilute(f_hi, 1.0 - 0.3 * b)
        al = (0.90 if st["method"] == "点" else 0.34) * (1.0 - 0.25 * b) * k
        for ox, oy, pr, a in pets:
            _wash(cv, rng, x, y, ox, oy, pr, a, f_lo, f_hi, air, al, b)

    if st["method"] == "圈" and view != "背":
        # 瘦 — and the width must **not** be proportional to the flower.
        # That is the mistake this line has now made twice. A brush does not
        # change nib when it moves from a small flower to a large one: 圈花
        # is written with the tip (中锋) at whatever weight the hand likes,
        # so a 60 px blossom wears very nearly the same hairline as a 30 px
        # one — a little more, because a bigger flower is a nearer flower,
        # not because it is scaled. At the first R*0.115 a 60 px flower
        # carried a 13 px line; at R*0.078 it still carried 9 px at the
        # press, which is the width of the petal's own radius divided by
        # two, and five of those overlapping read as a heap of fat commas
        # rather than as a flower. Mostly-constant plus a slow slope:
        # 15 px → 0.8, 34 px → 1.4, 58 px → 2.1.
        w = max(0.46, 0.36 + R * 0.030)
        for ox, oy, pr, a in pets:
            _ring(cv, rng, x, y, ox, oy, pr, a, r_lo, r_hi, w, air,
                  alpha=0.90 - 0.22 * b, gap=rng.uniform(0.72, 1.06), k=k)

    if view == "背":
        # 背面 — what you see is the calyx, a five-pointed dark star, with
        # the petals showing behind it.
        if st["method"] == "圈":
            w = max(0.42, 0.30 + R * 0.024)
            for ox, oy, pr, a in pets[:3]:
                _ring(cv, rng, x, y, ox, oy, pr, a, dilute(r_lo, 0.8),
                      dilute(r_hi, 0.85), w * 0.9, air, alpha=0.62, k=k)
        _calyx(cv, rng, x, y, attach + math.pi, R * 0.62, ink, air,
               k=k * (1.0 - 0.3 * b), n=5, spread=TAU / 5.0)
    else:
        # 蕊如鼠须 — and a whisker has to stay a whisker. A 14 px flower
        # cannot carry nine of them with an anther on each: the dots merge
        # and the blossom turns into a black asterisk again, this time from
        # density rather than from length. So the count follows the
        # flower's size *in pixels*, which is the only honest unit here.
        n = max(3, int(rng.randint(5, 9) * min(1.0, R / 13.0)
                       * (1.0 - 0.40 * b)))
        if view == "侧":
            mid = attach + math.pi
            _stamens(cv, rng, x, y, R * 0.92, ink, air, n,
                     mid - 1.0, mid + 1.0, k=k * (1.0 - 0.35 * b))
        else:
            _stamens(cv, rng, x, y, R * 0.92, ink, air, n, 0.0, TAU,
                     k=k * (1.0 - 0.35 * b))

    # 蒂 and 柄. A face-on flower hides its own calyx behind itself, so it
    # gets the stalk only; a profile flower shows both.
    bx = x + math.cos(attach) * R * 0.74
    by = y + math.sin(attach) * R * 0.74
    if view == "侧":
        _calyx(cv, rng, bx, by, attach, R * 0.40, ink, air,
               k=k * (1.0 - 0.3 * b))
    if hang is not None:
        _stalk(cv, rng, hang[0], hang[1], bx, by, max(0.45, R * 0.075),
               air, k=k * (1.0 - 0.25 * b))
    return (x, y, R)


def bud(cv, rng, res, x, y, R, st, air, attach=0.0, back=0.0, hang=None,
        k=1.0):
    """蕾 — 贵含不贵开. One or two small lobes and a prominent calyx.

    A sheet wants plenty of these: they are what says the tree is coming
    into flower rather than standing at its peak, and they are the cheapest
    way to vary a cluster, because a bud beside three flowers reads as the
    fourth flower not yet open.
    """
    b = clamp01(back)
    r = R * rng.uniform(0.34, 0.52)
    n = 1 if rng.random() < 0.45 else 2
    pets = []
    for i in range(n):
        a = attach + math.pi + (0.0 if n == 1
                                else (i - 0.5) * rng.uniform(0.7, 1.1))
        o = 0.0 if n == 1 else r * rng.uniform(0.28, 0.42)
        pets.append((math.cos(a) * o, math.sin(a) * o,
                     r * rng.uniform(0.86, 1.06), a))
    urad = _union_rad(pets, pad=max(0.4, r * 0.05))
    res.lift(cv, x, y, urad, alpha=0.95 - 0.28 * b, feather=0.85)

    ink = rgb(st["heart"])
    if st["method"] == "点" and st["fill"] is not None:
        f_lo, f_hi = rgb(st["fill"][0]), rgb(st["fill"][1])
        for ox, oy, pr, a in pets:
            _wash(cv, rng, x, y, ox, oy, pr, a, f_lo, f_hi, air,
                  0.92 * (1.0 - 0.2 * b) * k, b)
    else:
        r_lo = dilute(rgb(st["ring"][0]), 1.0 - 0.3 * b)
        r_hi = dilute(rgb(st["ring"][1]), 1.0 - 0.3 * b)
        if st["fill"] is not None:
            for ox, oy, pr, a in pets:
                _wash(cv, rng, x, y, ox, oy, pr, a, rgb(st["fill"][0]),
                      rgb(st["fill"][1]), air, 0.30 * k, b)
        for ox, oy, pr, a in pets:
            _ring(cv, rng, x, y, ox, oy, pr, a, r_lo, r_hi,
                  max(0.48, 0.34 + r * 0.055), air, alpha=0.88 - 0.2 * b,
                  gap=rng.uniform(0.70, 1.08), k=k)
    bx = x + math.cos(attach) * r * 0.70
    by = y + math.sin(attach) * r * 0.70
    _calyx(cv, rng, bx, by, attach, r * 0.95, ink, air, k=k * (1.0 - 0.3 * b),
           n=3, spread=0.44)
    if hang is not None:
        _stalk(cv, rng, hang[0], hang[1], bx, by, max(0.42, r * 0.14),
               air, k=k)
    return (x, y, r)


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

TITLE_A = ["墨梅", "疏影", "暗香", "寒香", "一枝春", "香雪", "铁骨", "孤山",
           "冷艳", "老梅", "玉骨", "岁寒", "梅花", "清标", "横斜", "寒枝"]
# ...and the words that are a *claim about the weather* are held back and
# handed out by the weather. 雪梅图 on a sheet with no snow on the wood is
# not a stylistic liberty, it is a caption contradicting the picture it is
# written on — and the inscription is *in* the painting, so the reader sees
# both at once. 香雪 stays in the open pool above: it is a metaphor for the
# blossom itself, which is why 香雪海 is a place in 苏州 and not a forecast.
TITLE_AIR = {
    "清晓": ["早春", "春信", "东风", "清晓"],
    "寒风": ["寒枝", "岁寒", "风梅", "寒香"],
    "踏雪": ["雪梅", "香雪", "雪窗", "岁寒"],
    "月下": ["月梅", "暗香", "月下", "疏影"],
}
TITLE_B = ["图", "小景", "清趣", "写意", "图卷", "册页", "小品"]
ACTION = ["写生", "戏作", "写意", "拟古", "画梅", "灯下戏墨", "南窗写梅",
          "并记", "补梅", "雪窗消寒"]
SEAL_WORDS = ["清赏", "墨缘", "寒香", "一枝", "写心", "铁骨", "孤芳", "老圃",
              "香雪", "春信", "玉洁", "疏影", "安之"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into mei_font.py."""
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
    for mod in ("mei_font", "grape_font", "orchid_font", "bamboo_font",
                "gongbi_font", "shanshui_font"):
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
    """Measure the block, then let the 留白 decide where it goes.

    Same machinery as the other eight painters. A 梅 sheet needs it the way
    a 兰 sheet does: the wood crosses the paper on one diagonal and leaves
    the opposite corner empty, and that corner is where the writing goes —
    which is why the occupancy map is fed the blossoms as well as the wood.
    """
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

def _entry(rng, sc, W, H, S, side, air):
    """Where the wood comes in, how it is aimed, and which way growth goes.

    The entry is what decides whether the sheet reads as 折枝 — a cut
    branch, hung off an edge, which is the orthodox plum composition — or
    as a tree, which only 古干 attempts and then only from the knees up.
    Nobody paints the root of a plum.
    """
    e = sc["entry"]
    L = S * rng.uniform(*sc["L"])
    if e == "foot":
        x = W * (rng.uniform(0.20, 0.42) if side > 0
                 else rng.uniform(0.58, 0.80))
        y = H * rng.uniform(1.01, 1.05)
        a0 = -math.pi * 0.5 + side * rng.uniform(0.10, 0.46)
        grow = -math.pi * 0.5 + side * rng.uniform(0.0, 0.30)
        return x, y, a0, L, grow, 0.55
    if e == "cut":
        # 折枝 — a snapped sprig, and the thing that makes it one is that
        # its butt is *inside* the paper. With `entry="side"` it came in
        # off an edge like 横斜 did, and the whole scene was then just a
        # thinner 横斜: four leaves of the 16-cell scene-by-air grid were
        # indistinguishable from the four above them. A 折枝 hangs in the
        # paper on nothing, with a cut face at the thick end — that is the
        # entire conceit of the form, and it has to be in the geometry
        # rather than in the width.
        x = W * (rng.uniform(0.07, 0.26) if side > 0
                 else rng.uniform(0.74, 0.93))
        low = rng.random() < 0.52
        if low:
            y = H * rng.uniform(0.62, 0.88)
            tilt = rng.uniform(0.34, 0.72)
        else:
            y = H * rng.uniform(0.12, 0.36)
            tilt = -rng.uniform(0.26, 0.58)
        a0 = (0.0 if side > 0 else math.pi) - side * tilt
        g = rng.uniform(0.72, 1.08) if low else rng.uniform(1.96, 2.34)
        grow = -math.pi * 0.5 + side * g
        return x, y, a0, L, grow, 0.34
    if e == "top":
        x = W * (rng.uniform(0.14, 0.40) if side > 0
                 else rng.uniform(0.60, 0.86))
        y = -H * rng.uniform(0.01, 0.05)
        a0 = math.pi * 0.5 - side * rng.uniform(0.20, 0.62)
        grow = math.pi * 0.5 - side * rng.uniform(0.0, 0.26)
        return x, y, a0, L, grow, 0.45
    # "side" — in from the left or the right edge: 疏影横斜.
    #
    # Two ways, and the sheet has to pick one, because entering halfway up
    # and leaning either way is how all nine leaves of the 册页 ended up
    # with their blossom in the same band across the upper middle. 仰枝
    # comes in at the bottom and climbs; 俯枝 comes in at the top and
    # falls. Both are orthodox and they fill opposite halves of the paper.
    x = -W * 0.04 if side > 0 else W * 1.04
    rise = rng.random() < 0.55
    # `tilt` is measured so that *positive* climbs, for both signs of
    # `side`: with the base angle at 0 or pi, sin(base + tilt) is negative
    # either way, and screen y runs down. The first version had the sign
    # inverted on the 仰枝 branch, so the bough that was supposed to climb
    # out of the bottom corner set off downwards and then got dragged flat
    # by `grow` — which is why three leaves of the 册页 had a horizontal
    # branch lying along the bottom edge instead of a diagonal.
    if rise:
        y = H * rng.uniform(0.68, 0.92)
        tilt = rng.uniform(0.30, 0.62)
    else:
        y = H * rng.uniform(0.16, 0.34)
        tilt = -rng.uniform(0.22, 0.52)
    a0 = (0.0 if side > 0 else math.pi) - side * tilt
    # 疏影*横*斜 — the second character is the instruction. At 0.30-0.72
    # off vertical the bough climbed at 50-70 degrees and took the whole
    # composition into the top third of the sheet, leaving the bottom half
    # dead; the 册页 showed it on every 横斜 leaf at once.
    # 忌横 — and the slope has to be a real slope. A 俯枝 that falls at ten
    # degrees runs parallel to the top edge of the paper, which is the one
    # direction a branch may not take; it also keeps the whole subject in a
    # strip along that edge. Both ways now cross the sheet diagonally.
    g = rng.uniform(0.70, 1.05) if rise else rng.uniform(1.98, 2.38)
    grow = -math.pi * 0.5 + side * g
    return x, y, a0, L, grow, 0.38


def _reach(pts, want, W, H):
    """How much of the gesture survived — 势要贯.

    grape's trick, and every painter in the family that hangs its subject
    off one path needs it: draw the main line a few times and keep the one
    that gets somewhere, because a path truncated at the second kink takes
    the whole composition with it.

    Scored on the *fraction of the intended length that got drawn*, not on
    how far the tip ended up from the root. Distance was the obvious
    measure and it is a trap: the shortest way to be far from where you
    started is to go straight, so the first version of this reliably chose
    the one candidate with no kinks in it — and a plum branch with no kinks
    is a willow switch. Keeping the sheet's own proportion as a faint
    tie-break is enough.

    The second term is 忌横, and it is the reason this is scored at all
    rather than taken on the first try. `_entry` already aims every bough
    25-30 degrees off the horizontal, which over 900 px of wood is 400 px
    of descent — but the 节 are ±0.4 rad and they alternate, so now and
    then they cancel the slope exactly and the sheet gets a branch lying
    flat across it with the blossom in a strip and two-thirds of the paper
    dead. It is not a bad *stroke*; it is a bad *crossing*, so it has to be
    measured on the path's bounding box rather than on any of its parts:
    the shorter side of that box is how much of the second dimension the
    gesture actually used, and a diagonal has both.

    It is *capped*, and the cap is the whole point. Uncapped it is the same
    trap as `far` wearing a different hat — the box of a straight 45-degree
    line is a square, and every 节 you put in the wood shrinks the box, so
    rewarding span without limit picks the willow switch all over again.
    At 0.22 any honest diagonal saturates the term and the kinks are free;
    only the branch lying flat across the paper misses it.
    """
    if len(pts) < 4:
        return -1e9
    x0, y0 = pts[0]
    far = max(math.hypot(x - x0, y - y0) for x, y in pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = min(max(xs) - min(xs), max(ys) - min(ys)) / min(W, H)
    return (len(pts) / max(1, want) + 0.12 * far / min(W, H)
            + 0.55 * min(span, 0.22))


def _twig(rng, parent, sc, air, S, grow, W, H, gen, frac=None, Lk=1.0,
          off=None, segs=None):
    """One new shoot off `parent`, or None if there is nowhere to put it.

    `frac`, `Lk`, `off` and `segs` let the caller ask for a *limb* — thick,
    long and leaving at a shallow angle — instead of a twig. They are the
    same stroke; a plum's skeleton is self-similar, and the only difference
    between the bough that carries the composition and the spur that
    carries one knot of blossom is three numbers.
    """
    pts = parent["pts"]
    n = len(pts) - 1
    if n < 4:
        return None
    t = parent["_t"]
    i = max(2, min(n - 1, int(t * n)))
    x, y = pts[i]
    ta = gb.tangent(pts, i)
    wn = parent["w0"] + (parent["w1"] - parent["w0"]) * (i / n)
    off = rng.uniform(*sc["fork"]) if off is None else off
    # 鹿角 — the shoot leaves at a real angle, and it leaves on the side it
    # was told to (the caller alternates), because two shoots off the same
    # flank of a bough is 并枝 and reads as a mistake. Then `pull` turns it
    # slowly toward where growth goes, which is what keeps a sheet of
    # branches agreeing with itself: a random direction per shoot is not
    # variety, it is noise — the lesson bamboo.py had to learn twice.
    a0 = ta + off * parent["_side"]
    a0 += air["lean"] * rng.uniform(0.3, 1.0) * (1.0 if grow < 0 else -1.0)
    L = (S * rng.uniform(*sc["Lw"]) * Lk * (1.0 - 0.34 * t)
         * (0.56 ** (gen - 1)))
    # 枝如鹿角 — and an antler is much thinner than the head it leaves.
    # At 0.40-0.60 of the parent, a gen-1 shoot off a 40 px trunk came out
    # 20 px wide, which does not read as a shoot at all: it reads as a
    # second grey log laid across the first.
    # ...but the *fraction* has to grow as the parent thins, or two
    # generations of 0.25 take a 4 px twig down to a quarter of a pixel
    # and the shoots come out as antennae with flowers on the ends.
    if frac is None:
        frac = 0.20 + 0.24 * (1.0 - clamp01((wn - 4.0) / 10.0))
    w0 = max(0.85, wn * frac * rng.uniform(0.82, 1.18))
    # ...and now the length has to answer to that width, because `Lw` is a
    # fraction of the *sheet* and the taper is a fraction of `w0`. A whip
    # that came off a thin shoot got 1 px of width and still took its full
    # 220 px of length, with `hws` clamped at the 0.42 floor from end to
    # end — a dead-straight uniform hairline 200 px long, drawn across the
    # blossom. Six of them per sheet, and they read exactly like what they
    # are: pins stuck through the paper.
    #
    # A branch is a tapering solid, so its length scales with its butt.
    # 40:1 is about what a plum 条 actually runs, and it means a 1 px twig
    # is 35 px long (a spur) while a 6 px limb still gets its 240.
    L = min(L, w0 * rng.uniform(30.0, 46.0))
    # 鹿角 — more kink and less pull than the first version, which turned
    # every shoot into a straight stick: `pull` is what makes a sheet of
    # branches agree about which way is up, and at 0.55 it also ironed the
    # joints out of them.
    out, joints = wood_path(
        rng, x, y, a0, L,
        segs=(segs if segs is not None else
              max(2, min(4, int(L / (S * 0.085)) + rng.randint(1, 2)))),
        kink=sc["kink"] * rng.uniform(1.2, 1.9), bow=0.16, grow=grow,
        pull=0.30 if gen == 1 else 0.20, cone=1.15, bounds=(W, H))
    if len(out) < 5:
        return None
    return dict(pts=out, joints=joints, w0=w0,
                w1=max(0.30, w0 * rng.uniform(0.13, 0.24)), gen=gen,
                at=(x, y), _t=t)


def _sites(rng, tw, sc, R0, pull, bud_p):
    """Where the blossoms go on one shoot — 聚散, and 花在节上.

    Plum does not flower evenly along a shoot. It flowers in knots, at the
    joints and on the short spurs, and thins to buds at the tip. So: a few
    cluster centres with a minimum spacing (never two at one joint), an odd
    number of flowers in each because odd numbers do not read as pairs, and
    the count falling off with distance from the sheet's 主聚.
    """
    pts = tw["pts"]
    n = len(pts) - 1
    if n < 4:
        return []
    # 聚散. The first version gave every shoot a knot or two whatever its
    # distance from the 主聚, which spread the blossom evenly along every
    # piece of wood on the sheet — popcorn on a string, and the thing the
    # 册页 showed after the skeleton was fixed. A plum sheet wants one
    # place where the flower gathers, two or three subordinate groups, and
    # a good deal of bare wood: 贵稀不贵繁.
    if pull < 0.42 and rng.random() < 0.45:
        return []                                   # this shoot stays bare
    nc = max(1, int(round(rng.uniform(*sc["clusters"])
                          * (0.30 + 1.05 * pull))))
    out = []
    for t in _spread(rng, nc, 0.10, 0.98, 0.17):
        i = max(1, min(n, int(t * n)))
        bx, by = pts[i]
        ta = gb.tangent(pts, i)
        wn = tw["w0"] + (tw["w1"] - tw["w0"]) * (i / n)
        # 聚散 — one knot of five, the rest of three or one. A cluster size
        # drawn independently per knot gives a sheet with no main group.
        m = (7 if rng.random() < 0.26 * pull ** 2 else
             (5 if rng.random() < 0.46 * pull else
              (3 if rng.random() < 0.24 + 0.46 * pull else 1)))
        sgn = rng.choice((1.0, -1.0))
        for j in range(m):
            R = R0 * rng.uniform(0.72, 1.26) * (1.0 - 0.16 * t)
            # The flower stands off the wood on the node's own normal and
            # the flanks alternate. It used to stand off by up to 1.7 of
            # its own diameter, which with a stalk measured from the
            # *centreline* gave flowers floating two radii clear of the
            # twig on the end of a long dark spike. Both ends moved: the
            # offset is tighter, and the stalk starts at the edge of the
            # wood, which is where a stalk starts.
            fl = sgn if j else rng.choice((1.0, -1.0))
            d = (R + wn) * rng.uniform(0.34, 1.00) * fl
            sgn = -sgn
            k = max(1, min(n, i + int(rng.gauss(0, n * 0.030))))
            hx, hy = _perp(pts, k, fl * wn * 0.85)
            fx, fy = _perp(pts, k, d)
            fx += rng.gauss(0, R * 0.22)
            fy += rng.gauss(0, R * 0.22)
            attach = math.atan2(hy - fy, hx - fx)
            # 贵含不贵开 — buds crowd the tip of the shoot
            is_bud = rng.random() < clamp01(bud_p + 0.42 * t - 0.18 * pull)
            if is_bud:
                view = "正"
            else:
                r = rng.random()
                view = "正" if r < 0.52 else ("侧" if r < 0.84 else "背")
            out.append(dict(x=fx, y=fy, R=R, attach=attach, hang=(hx, hy),
                            view=view, bud=is_bud,
                            back=(0.0 if j == 0 else
                                  clamp01(rng.uniform(0.10, 0.85))),
                            t=t, ta=ta))
    return out


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
    side = rng.choice((1.0, -1.0))

    if "veil" in air:
        veil(cv, rgb(air["veil"][0]), air["veil"][1])
    if air.get("moon"):
        # 月 — in the quarter the wood is coming *from*, high, where the
        # branches have not got to yet.
        mr = S * rng.uniform(0.055, 0.085)
        mx = W * (rng.uniform(0.16, 0.30) if side < 0
                  else rng.uniform(0.70, 0.84))
        my = H * rng.uniform(0.12, 0.26)
        moon(cv, rng, mx, my, mr, I("qing"))
        occ.add(mx, my, mr * 1.5)

    # ---- the photograph of the paper. Everything after this line can be
    # ---- lifted back out of the sheet; nothing before it can.
    res = Reserve(cv)

    # 主干 — the gesture, drawn five times and the best one kept.
    best = None
    for _ in range(5):
        x, y, a0, L, grow, pull = _entry(rng, sc, W, H, S, side, air)
        want = [0]
        pts, joints = wood_path(
            rng, x, y, a0, L, segs=rng.randint(*sc["segs"]),
            kink=sc["kink"], bow=0.22, grow=grow, pull=pull,
            cone=1.25, bounds=(W, H), want=want)
        sco = _reach(pts, want[0], W, H)
        if best is None or sco > best[0]:
            best = (sco, pts, joints, grow)
    _, trunk_pts, trunk_joints, grow = best
    w0 = S * rng.uniform(*sc["w0"])
    trunk = dict(pts=trunk_pts, joints=trunk_joints, w0=w0,
                 w1=max(0.8, w0 * rng.uniform(0.14, 0.24)), gen=0,
                 at=trunk_pts[0], _t=0.0)
    old = sc["old"]
    tw_side = rng.choice((1.0, -1.0))
    tw_load = tw_side * rng.uniform(0.55, 0.95)
    thws = bough(cv, rng, trunk_pts, trunk["w0"], trunk["w1"], st, air,
                 alpha=0.96, old=old, bone=0.55 + 0.45 * old, k=k,
                 side=tw_load)
    occ.add_path(trunk_pts, trunk["w0"] * 1.3, trunk["w1"] * 1.3)
    # 斑驳 first, bark second: `blot` lifts paper and would wipe out any
    # line already drawn inside the mass.
    blot(cv, rng, res, trunk_pts, thws, air, k=k, old=old, side=tw_load,
         n=int(4 + 6 * old + 3 * max(0.0, S / 820.0 - 1.0)))
    flyback(cv, rng, res, trunk_pts, thws, air, k=k, side=tw_load,
            n=int((3 + 5 * old + 3 * air["dry"]) * max(1.0, S / 820.0)))
    if sc["entry"] == "cut":
        snap(cv, rng, res, trunk_pts, thws, air, k=k)
    # 皴 is **not** gated on age. It was, at `old > 0.3`, and 折枝 is
    # defined with `old=0.25` — so the one scene whose whole subject is a
    # single cut branch across an empty sheet got no bark whatsoever, and
    # came out as exactly what the complaint said: a smooth grey tube with
    # an even gradient across it and an elliptical lid on the end. Age
    # decides *how much* texture there is, never whether there is any; a
    # brush with hair in it leaves 擦 on a one-year shoot. Moss does belong
    # to old wood only — that one is a fact about trees.
    cun(cv, rng, trunk_pts, thws, air,
        n=int((26 + 46 * old) * max(1.0, S / 820.0)), k=k, old=old)
    if old > 0.3:
        moss(cv, rng, trunk_pts, thws, k=k, n=int(2 + 5 * old))
    for _ in range(rng.randint(*sc["scars"])):
        scar(cv, rng, res, trunk_pts, thws, air, k=k)

    # ---- 枝如鹿角 -------------------------------------------------------
    #
    # The 册页 diagnostic — nine sheets mounted small, crab's §5.1 — said
    # one thing about this module louder than anything else, and it was not
    # a thing any single sheet showed: **every sheet was the same sheet**.
    # One bare diagonal pole from corner to corner with blossom clinging
    # along its middle, nine times. The faults were not in the wood or in
    # the flowers; they were that the skeleton had exactly one branch in it
    # and the flowers had nowhere to go but onto the trunk.
    #
    # So the trunk now *divides*. Two kinds of wood come off it:
    #
    #   * 大枝, limbs — thick (half the parent), long, leaving at a shallow
    #     angle, and carrying twigs of their own. One of them is the
    #     terminal fork: the trunk does not taper away to a needle in open
    #     paper, it forks near its end, which is what wood does and what
    #     stops the point being read as the end of a drawn line.
    #   * 小枝, whips — as before, off the trunk *and* off the limbs.
    #
    # 忌对节 all the way down: a limb and a whip never leave the same node.
    twigs = []
    limbs = []
    flank = rng.choice((1.0, -1.0))

    def _add(tg, par=None, limb=False):
        """Register one piece of wood and return its index.

        The index matters because the blossom decides which wood gets
        drawn, and a shoot that flowered needs its *parent* drawn too or
        the knot of blossom hangs in the air on the end of nothing.
        """
        if tg is None:
            return None
        tg["_par"] = par
        if limb:
            tg["_limb"] = True
            limbs.append(tg)
        twigs.append(tg)
        return len(twigs) - 1

    trunk["_t"] = rng.uniform(0.52, 0.70)
    trunk["_side"] = flank
    _add(_twig(rng, trunk, sc, air, S, grow, W, H, 1,
               frac=rng.uniform(0.60, 0.86), Lk=rng.uniform(1.25, 1.85),
               off=rng.uniform(0.20, 0.46), segs=rng.randint(2, 3)),
         limb=True)
    for t in _spread(rng, rng.randint(*sc["limbs"]), 0.24, 0.76, 0.20):
        trunk["_t"] = t
        flank = -flank
        trunk["_side"] = flank
        _add(_twig(rng, trunk, sc, air, S, grow, W, H, 1,
                   frac=rng.uniform(0.38, 0.56), Lk=rng.uniform(0.95, 1.45),
                   off=rng.uniform(0.42, 0.82), segs=rng.randint(2, 4)),
             limb=True)

    # 梢 — the tip. The division above moved down the trunk to where the
    # wood is still thick enough to fork into two real boughs, which left
    # the trunk's own end doing what it did before: tapering to a pale
    # needle and stopping, in open paper, halfway across the sheet. A
    # branch that ends in mid-air reads as a drawn line that ran out, not
    # as wood — wood ends in a spray of fine shoots. So the last node gets
    # one near-collinear continuation (`off` small, so it carries the
    # gesture on rather than turning off it) and that continuation gets
    # whips of its own. It is marked as a limb, i.e. drawn whether or not
    # the blossom chose it: the tip of the main gesture is structure.
    trunk["_t"] = rng.uniform(0.93, 0.99)
    trunk["_side"] = rng.choice((1.0, -1.0))
    tip = _add(_twig(rng, trunk, sc, air, S, grow, W, H, 1,
                     frac=rng.uniform(0.80, 1.00), Lk=rng.uniform(1.1, 1.6),
                     off=rng.uniform(0.06, 0.26), segs=rng.randint(2, 3)),
               limb=True)
    if tip is not None:
        tg = twigs[tip]
        for t in _spread(rng, rng.randint(1, 3), 0.20, 0.92, 0.16):
            tg["_t"] = t
            tg["_side"] = rng.choice((1.0, -1.0))
            _add(_twig(rng, tg, sc, air, S, grow, W, H, 2), par=tip)

    n_wh = rng.randint(*sc["whips"])
    for t in _spread(rng, n_wh, sc["span"][0], sc["span"][1], 0.11):
        trunk["_t"] = t
        # 忌并枝 — alternate the flank, except that a bough carries more of
        # its wood on the side growth is going (two thirds up, one third
        # down, which is what a plum actually does and what keeps the
        # alternation from reading as a fishbone).
        ta = gb.tangent(trunk_pts, max(1, min(len(trunk_pts) - 1,
                                              int(t * (len(trunk_pts) - 1)))))
        up = -1.0 if _turn(ta, grow) < 0 else 1.0
        s_i = flank if rng.random() < 0.78 else -flank   # alternate, loosely
        if rng.random() < 0.17:
            s_i = up                   # ...and the sunny side carries more
        trunk["_side"] = s_i
        flank = -s_i
        _add(_twig(rng, trunk, sc, air, S, grow, W, H, 1))
    # ...and the limbs carry whips of their own, which is where most of the
    # blossom ends up: 老干不生花, 花在小枝上.
    for li, lb in enumerate(list(limbs)):
        pi = twigs.index(lb)
        for t in _spread(rng, rng.randint(2, 4), 0.14, 0.94, 0.14):
            lb["_t"] = t
            lb["_side"] = rng.choice((1.0, -1.0))
            _add(_twig(rng, lb, sc, air, S, grow, W, H, 2), par=pi)
    # A third generation off the whips was in the first version and is out
    # again: 0.56 of 0.56 of a shoot is 70 px long and 1.7 px wide, with
    # two segments over four nodes, so it comes out as a dead-straight
    # hairline — a pin stuck through the paper with a flower on the end.
    # Depth comes from the 短刺 below instead, which is what a plum
    # actually flowers on.
    # 短刺 — the spurs: stubs off the old wood that carry one knot of
    # blossom each. 老干生花 happens on these, never on the long span.
    for t in _spread(rng, rng.randint(*sc["spurs"]), 0.18, 0.92, 0.12):
        trunk["_t"] = t
        trunk["_side"] = rng.choice((1.0, -1.0))
        i = max(2, min(len(trunk_pts) - 2, int(t * (len(trunk_pts) - 1))))
        ta = gb.tangent(trunk_pts, i)
        a = ta + rng.uniform(0.75, 1.35) * trunk["_side"]
        sx, sy = _perp(trunk_pts, i, thws[i] * 0.7 * trunk["_side"])
        L = S * rng.uniform(0.020, 0.044)
        sp, sj = wood_path(rng, sx, sy, a, L, segs=2, kink=0.30, bow=0.45,
                           grow=grow, pull=0.40, cone=0.8, bounds=(W, H),
                           step=2.5)
        if len(sp) < 3:
            continue
        _add(dict(pts=sp, joints=sj, w0=max(0.7, thws[i] * 0.16),
                  w1=0.42, gen=4, at=(sx, sy), _t=t))

    # 花 — 主聚: the sheet needs one place where the blossom gathers, and
    # everything else thins away from it.
    #
    # That point has to be **on the wood**. The first version drew it from
    # the sheet's own coordinates, which looks reasonable and is not: the
    # branch is somewhere else by then, so on a sheet where the two missed
    # each other *every* shoot came out below the threshold and the leaf
    # went nearly bare. Anchoring it on a limb guarantees one group with
    # pull near 1, and the rest of the sheet still thins away from it.
    host = rng.choice(limbs)["pts"] if limbs else trunk_pts
    hx0, hy0 = host[int(len(host) * rng.uniform(0.35, 0.75))]
    fx = hx0 + rng.gauss(0, S * 0.045)
    fy = hy0 + rng.gauss(0, S * 0.045)
    R0 = S * rng.uniform(0.0185, 0.0245)

    def _plan(boost):
        out = []
        for j, tg in enumerate(twigs):
            if not tg["pts"] or tg["gen"] == 0:
                continue
            mx, my = tg["pts"][len(tg["pts"]) // 2]
            pull = 1.0 / (1.0 + math.hypot(mx - fx, my - fy) / (S * 0.30))
            for f in _sites(rng, tg, sc, R0, min(1.0, pull + boost),
                            sc["bud"]):
                f["tw"] = j
                out.append(f)
        return out

    plan = _plan(0.0)
    if len(plan) < 14:
        # 贵稀不贵繁 is not 空. A sheet whose wood all came out far from the
        # 主聚 gets a second pass with the thinning relaxed.
        plan = _plan(0.34)

    # 忌堆砌 — the flowers may overlap (that is what the lift is for) but
    # they may not pile up. Reject any that sits more than halfway inside
    # one already accepted, and keep the sheet off its own edges.
    keep = []
    for f in plan:
        if not (-R0 * 0.5 < f["x"] < W + R0 * 0.5
                and -R0 * 0.5 < f["y"] < H + R0 * 0.5):
            continue
        r = f["R"] * (0.45 if f["bud"] else 1.0)
        bad = False
        for g in keep:
            rg = g["R"] * (0.45 if g["bud"] else 1.0)
            if math.hypot(f["x"] - g["x"], f["y"] - g["y"]) < 0.74 * (r + rg):
                bad = True
                break
        if not bad:
            keep.append(f)

    # ---- 枝 ------------------------------------------------------------
    #
    # Drawn only now, because which of it gets drawn depends on where the
    # blossom went. A spur exists to carry one knot of flower; a spur that
    # drew none is a stray hair across the paper, and the 册页 had several
    # on every leaf — two or three fine lines crossing the empty half of
    # the sheet, reading as scratches in the silk. Anything structural
    # still gets drawn whether it flowered or not.
    used = set(f["tw"] for f in keep)
    for j in list(used):                      # ...and everything it hangs off
        p = twigs[j].get("_par")
        while p is not None and p not in used:
            used.add(p)
            p = twigs[p].get("_par")
    for j, tg in enumerate(twigs):
        if j not in used and not tg.get("_limb"):
            continue
        if tg["w0"] < 1.15 and not tg.get("_limb"):
            continue                      # 1 px of ink is a hair, not a twig
        ag = old * (0.70 if tg.get("_limb") else 0.30)
        hw = bough(cv, rng, tg["pts"], tg["w0"], tg["w1"], st, air,
                   alpha=0.94, old=ag,
                   bone=0.42 if tg["w0"] > 3.0 else 0.0,
                   k=k, knots=rng.uniform(1.6, 2.8))
        if tg.get("_limb") and tg["w0"] > 4.0:
            blot(cv, rng, res, tg["pts"], hw, air, k=k * 0.9, old=ag,
                 n=rng.randint(1, 3))
            flyback(cv, rng, res, tg["pts"], hw, air, k=k,
                    n=rng.randint(2, 4))
        if tg.get("_limb") and tg["w0"] > 4.0:
            cun(cv, rng, tg["pts"], hw, air,
                n=int(5 + 18 * old), k=k * 0.85, old=ag)
        occ.add_path(tg["pts"], tg["w0"] * 1.2, tg["w1"] * 1.4)

    # 前实后虚 — far first, so each flower lifts the ones behind it. This
    # is the whole payoff of `Reserve`: the sheet gets real occlusion out
    # of a compositing model that cannot express it.
    keep.sort(key=lambda f: -f["back"])
    spots = []
    nf = nb = 0
    for f in keep:
        if f["bud"]:
            spots.append(bud(cv, rng, res, f["x"], f["y"], f["R"], st, air,
                             attach=f["attach"], back=f["back"],
                             hang=f["hang"], k=k))
            nb += 1
        else:
            spots.append(blossom(cv, rng, res, f["x"], f["y"], f["R"], st,
                                 air, view=f["view"], attach=f["attach"],
                                 back=f["back"], hang=f["hang"], k=k))
            nf += 1
    for sx, sy, sr in spots:
        occ.add(sx, sy, sr * 1.15)

    if air.get("snow"):
        up = -1.0 if rng.random() < 0.8 else 1.0
        snow_on(cv, rng, res, trunk_pts, thws, up,
                [(a, b, c * 1.1) for a, b, c in spots], k=k)
        for tg in twigs:
            if tg["w0"] < 1.8:
                continue
            hw = [tg["w0"] + (tg["w1"] - tg["w0"]) * (i / (len(tg["pts"]) - 1))
                  for i in range(len(tg["pts"]))]
            snow_on(cv, rng, res, tg["pts"], hw, up,
                    [(a, b, c * 1.1) for a, b, c in spots], k=k)
        snowfall(cv, rng, S, int(S * 0.42))

    if stat is not None:
        stat["flowers"] = nf
        stat["buds"] = nb
        stat["twigs"] = len(twigs)
        stat["side"] = side

    head = (air["cn"] if rng.random() < 0.18
            else (TITLE_AIR[air_key][rng.randrange(len(TITLE_AIR[air_key]))]
                  if rng.random() < 0.46
                  else TITLE_A[rng.randrange(len(TITLE_A))]))
    title = head + TITLE_B[rng.randrange(len(TITLE_B))]
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
        description="Generate random Chinese ink-plum paintings (墨梅).")
    ap.add_argument("-o", "--out", default="mei.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="横斜|垂枝|折枝|古干")
    ap.add_argument("--air", help="清晓|寒风|踏雪|月下")
    ap.add_argument("--style", help="墨梅|白梅|红梅|绿萼")
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
        print("梅有四贵: 贵稀不贵繁, 贵老不贵嫩, 贵瘦不贵肥, 贵含不贵开 — "
              "干如屈铁, 枝如鹿角, 花分五出, 蒂如丁香")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("墨梅 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
