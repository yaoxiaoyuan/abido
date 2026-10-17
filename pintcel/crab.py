#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crab.py — 随机生成中国画虾蟹图 / Random Chinese ink paintings of 虾蟹.

虾蟹 is one subject and not two. The pair is 齐白石's, and he painted them
together because they answer each other: 虾 is almost all water — pale,
jointed, translucent, carried by its whiskers — and 蟹 is almost all bone,
eight hard legs and a shell, planted on the paper. Put one of each on a
sheet and the water between them becomes real. That is the whole trick, and
it is the same trick fish.py plays with fish, under the same law:

    **不画水.** The water is not painted. The 留白 *is* the pond.

The 画诀 this module implements, in the order the brush does it:

    壳用三笔   a carapace is three presses of a side-loaded brush — one in
               the middle reaching a little further forward, one each side
               raked out. The seams between them are paper, not ink.
    八足二螯   eight legs and two claws. The count is not negotiable; a crab
               with six legs is a spider and nobody is fooled.
    足分三节   every leg is three joints, and the brush *stops* at each one
               (顿笔) — a leg is drawn with force, not sketched.
    螯有大小   the two claws are never equal. Real crabs are handed, and two
               matching claws make the whole creature symmetrical, which is
               the one thing it must not be.
    蟹横行     a crab walks sideways. Its shell faces one way and it travels
               another, so the legs on the leading side reach and the legs
               behind trail.
    虾身透明   a shrimp is 淡墨 from end to end with exactly one 重墨 dab on
               the carapace. That dab is what makes the rest read as glass.
    虾贵在须   the whiskers are longer than the shrimp, finer than anything
               else on the sheet, and all dragged the same way by the same
               water.
    一水之势   every shrimp on the sheet swims in one current. The fan is
               about ±25°, which is variety; past that each shrimp is a
               separate drawing and the group stops being a group.
    忌排队     shrimp gather 聚散 — a knot and a straggler, strung on a 势
               line laid *oblique* to the way they are swimming. Along it
               they would be in one another's wake; across it they would be
               a rank. Oblique, they are 雁阵.
    聚散有别   and the 聚 is a *different rule* from the 散, not a wider
               tolerance on one rule. Two shrimp may close to two-thirds of
               their footprints; a crab keeps its full stand-off. One gap
               law for every pair does not permit even spacing, it enforces
               it, and even spacing is 匀 — no 聚, therefore no 散.
    取势       the group leans. Its box is aimed at a quarter of the sheet
               and not the middle, so the 留白 collects on one diagonal as
               one crossable void instead of four equal empty margins.
    主从       one creature is the biggest and keeps full ink; the rest fall
               away in size, and the ink ramp is held back for all of them
               but one. 远 is worth a single creature — size and pallor
               multiplied together do not read as distance, they erase.
    虾有伸屈   a third of the shrimp are nearly straight (伸, darting) and
               the rest are bent hard (屈, hanging in the water). One narrow
               band of curl gives one silhouette at five sizes.
    壳宜润     a carapace is painted wet, in one go, and is smooth. 擦 is for
               a leg dragged over dry paper; on a shell its streaks read as
               corrugation and the creature becomes a woodlouse.

虾 already exists in fish.py, where it is a *companion*: a 90-pixel
creature dropped into the corner of a fish painting, built out of `Blade`
arcs with four-point polyline whiskers. Rendered at the 250 pixels a subject
needs, it comes apart in ways that are all about scale rather than
correctness — the whiskers show their joints as kinks, the abdomen reads as
a pinecone of pale beads, the claws are jointless sticks and the body has no
dark mark anywhere on it to be translucent *against*. So this module writes
its own shrimp, out of presses and swept curves, and fish.py keeps its
own. Two shrimps in the family, each the right one for its sheet.

Everything structural is borrowed:

    gongbi.py   paper, noise, PNG, occupancy, calligraphy, seals
    bamboo.py   the ink table, 兑水, the full-sheet veil
    orchid.py   `sweep` — one brush dragged along a polyline, composited
                once, so a three-jointed leg has no beads at its knuckles
    grape.py    `touch` — a radial field with shade *and coverage*, which is
                what 留白 needs; this module's own primitive, `press`, is a
                side-loaded 侧锋 stroke built on it
    fish.py   the water furniture (水草, 浮萍, 气泡). 虾蟹 live in the same
                pond as 游鱼 and must not disagree with it about what weed
                looks like. Its 忌三点一线 test is *not* borrowed: a shoal
                here is laid out along a line on purpose, so that taboo had
                to be re-expressed rather than imported (`creature_sites`).

Usage
-----
    python3 crab.py                               # one random painting
    python3 crab.py -n 6                          # six of them
    python3 crab.py --scene 双蟹 --air 水动
    python3 crab.py --style 赭墨 --seed 7
    python3 crab.py --size 900x1200 -o scroll.png
    python3 crab.py --list                        # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Occupancy, BrushFont, clamp01, disc, jitter, mix,
                    rgb, paint_column, paint_seal, TIANGAN, DIZHI)
from bamboo import I, dilute, veil
from orchid import sweep
from grape import touch
# The water furniture. fish.py's `_collinear` — "is this creature strung on
# a line through two others?" — used to be imported here as well, and is not
# any more: a shoal of shrimp is now laid out *along* a line on purpose
# (see `creature_sites`), so the 三点一线 taboo had to be re-expressed as the
# stagger across that line rather than as a test that refuses lines.
from fish import water_weed, duckweed, paint_bubbles

TAU = math.tau


# --------------------------------------------------------------------------
# 设色 — the four schools.
#
# 水墨 is the orthodox form: 齐白石 painted hundreds of these sheets in
# nothing but ink, and the others are small departures from it rather than
# colour schemes. A crab's shell is 赭石 in life and painting it that way is
# allowed; painting it the red of a *cooked* crab is a dinner menu.
#
# Each entry gives the two stops of a ramp — the pale flank and the loaded
# core. Everything here multiplies into the paper rather than covering it,
# so the pale stop is a dilution and never a grey: the paper supplies all the
# warmth there is.
#
# `shell` is the carapace and the shrimp's body, `limb` the legs, claws and
# every other line with bone in it, and `weed` is fish.py's 水草 pair.
# --------------------------------------------------------------------------

STYLES = {
    "水墨": dict(cn="水墨", en="plain ink, after 齐白石",
                 shell=("#A39B90", "#332E28"), limb=("#6E6659", "#221E1A"),
                 ink="#201C19", weed=("#DCD9CF", "#6A6659")),
    "赭墨": dict(cn="赭墨", en="ochre shell, ink legs",
                 shell=("#B6A287", "#4A3324"), limb=("#6B6154", "#241E18"),
                 ink="#2A201A", weed=("#D8D6C6", "#6B6752")),
    "青墨": dict(cn="青墨", en="cool ink — deep water",
                 shell=("#9AA1A2", "#2C343A"), limb=("#67696A", "#1E2328"),
                 ink="#1C2024", weed=("#D2D8D2", "#5F6A60")),
    "淡彩": dict(cn="淡彩", en="ink with one warm wash",
                 shell=("#B2A694", "#3C352C"), limb=("#6C6354", "#231F18"),
                 ink="#241F1A", weed=("#D0D5B8", "#5E6A42")),
}

# 章法 — the four framings.
#
# `shrimp` and `crab` are how many of each; `ss` and `cs` are their sizes as
# a fraction of the sheet's short side (a shrimp's is its body length, a
# crab's is the width of its shell, which is the measure a crab painter
# actually uses). `spread` is how far the headings may wander from the
# current, `low` how far down the sheet the crabs are kept.
#
# Note that no framing has more than two crabs and none has none at all
# except 群虾. A crab is a heavy, planted, eight-legged mark that takes four
# times the paper its shell does; three of them on a sheet is a bucket.
#
# The sizes went up and the crabs went down after a nine-sheet 册页 was read
# at thumbnail size, which is the scale 章法 is actually judged at and which
# showed what no single sheet had: nine pictures of creatures floating in the
# middle of the paper with all four margins empty. Small figures in the
# centre of a large sheet is the one composition that cannot be rescued by
# anything inside it.
SCENES = {
    "虾蟹": dict(cn="虾蟹", en="the orthodox pair — shrimp in the water, "
                             "crab on the bottom",
                 shrimp=(3, 5), crab=(1, 2), ss=(0.19, 0.26), cs=(0.20, 0.26),
                 spread=0.55, low=0.58, weed=0.62, duck=0.55, bub=0.30),
    "群虾": dict(cn="群虾", en="a shoal of shrimp, 聚散有致",
                 shrimp=(4, 7), crab=(0, 0), ss=(0.19, 0.27), cs=(0.14, 0.18),
                 # 群虾 has no crab to weight the bottom of the sheet, so the
                 # weed has to: at 0.45 better than half these sheets came
                 # out as shrimp in open paper with nothing rooted anywhere.
                 spread=0.70, low=0.55, weed=0.64, duck=0.62, bub=0.40),
    "双蟹": dict(cn="双蟹", en="two crabs, close up",
                 shrimp=(0, 1), crab=(2, 2), ss=(0.16, 0.21), cs=(0.22, 0.28),
                 spread=0.40, low=0.50, weed=0.62, duck=0.40, bub=0.12),
    "芦间": dict(cn="芦间", en="down among the weed",
                 shrimp=(2, 4), crab=(1, 1), ss=(0.18, 0.23), cs=(0.18, 0.23),
                 spread=0.60, low=0.60, weed=1.00, duck=0.70, bub=0.22),
}

# 风致 — one body of water for the whole sheet.
#
# `flow` is how strongly the creatures share a heading, `wet` scales every
# soft edge, `dry` the 飞白 in the long strokes, `k` the overall strength of
# the ink.
AIR = {
    "静水": dict(cn="止水", en="still water",
                 flow=0.00, wet=1.00, dry=0.16, k=1.00),
    "水动": dict(cn="活水", en="a current running",
                 flow=0.55, wet=0.86, dry=0.30, k=1.00),
    "秋深": dict(cn="秋水", en="late autumn, a drier brush",
                 flow=0.20, wet=0.78, dry=0.48, k=0.94),
    "雨后": dict(cn="雨余", en="after rain, everything swollen",
                 flow=0.10, wet=1.70, dry=0.04, k=0.90,
                 veil=("#6E7A86", 0.07)),
}


# --------------------------------------------------------------------------
# 侧锋一笔 — one press of a side-loaded brush.
#
# This is the module's own primitive and it is a thin shell over grape.py's
# `touch`, because a 蟹壳 stroke and a 葡萄 berry turn out to be the same
# object: a closed radial field with an ink function that returns shade
# *and coverage*. What 蟹 needs that 葡萄 did not is that the load sit on one
# flank instead of in the middle — the brush is laid down on its side, so one
# edge of the mark is wet and dark and the other is where the hairs ran out.
#
# Two things the coverage channel buys here, neither of which pale ink can:
#
#   * 壳三笔 only reads as three strokes if the seams between them are
#     *paper*. Painted as pale ink lines they are three strokes with two
#     grey lines ruled across them.
#   * the dry flank of a press is bare sheet in streaks parallel to the
#     drag (擦), which is the difference between a shell and a pebble.
#
# `na`/`nr` default to None and are sized from the stroke, which is the one
# thing grape.py's §9.1 said its `touch` call sites should have done: a
# pincer 12 pixels long does not need 224 angular samples, and this module
# draws a lot of very small presses.
# --------------------------------------------------------------------------

def press(cv, x, y, L, W, ang, lo, hi, load=0.80, grad=0.45, flank=1.0,
          rim=0.10, bare=0.0, streak=0.0, tip=0.0, tail=0.0, ragged=0.05,
          bulge=0.0, splay=0.0, box=0.0, notch=0.0, back=0.0,
          phase=0.0, alpha=0.90, feather=1.25, na=None, nr=None):
    """One brush press, `L` long along `ang` and `W` wide across it.

    `load` is how much ink the brush is carrying, `grad` how much of that is
    lost across the stroke and `flank` (±1) which side keeps it. `rim` darkens
    the very edge, where a wet mark dries dark. `bare` lifts coverage off the
    dry flank and `streak` breaks what is left into 擦 streaks along the drag.
    `tip` and `tail` point the leading and trailing ends.

    The last four dials are the mark's *silhouette*, and between them they are
    the difference between a brush press and an ellipse:

    * `bulge` widens the loaded flank and narrows the dry one — a side-loaded
      brush spreads where the ink is and comes off the paper where it has run
      out, so the mark is not symmetric about its own axis.
    * `splay` makes the loaded flank *longer* as well as fatter: the mark
      reaches further along the drag on the inked side than on the dry one,
      which turns the pear into a fan. See the 蟹壳 for why this is the one
      that matters.
    * `box` squares the mark off, so it has four corners instead of none.
    * `notch` dents the middle of the loaded edge.
    * `back` cuts the dry edge off straight, which is the one thing none of
      the others can do: every dial above modulates a radius, and no radius
      modulation turns a round end into a flat one.

    All five default to zero, which leaves a plain squashed disc.
    """
    R = L * 0.5
    if R < 0.5 or W < 0.4:
        return
    if na is None:
        na = max(48, min(216, int(L * 1.7)))
    if nr is None:
        nr = max(7, min(18, int(L * 0.06) + 7))
    w1 = ragged * math.sin(phase * 2.7 + 1.1)
    w2 = ragged * 0.6 * math.cos(phase * 1.9 + 0.3)

    def rad(th):
        c, sn = math.cos(th), math.sin(th)
        f = 1.0
        if tip:
            f -= tip * (c if c > 0.0 else 0.0) ** 1.8
        if tail:
            f -= tail * (-c if c < 0.0 else 0.0) ** 1.8
        if bulge:
            # fatter on the loaded flank (where sn * flank < 0), leaner on
            # the dry one — a pear rather than an ellipse
            f += bulge * (-0.72 * sn * flank
                          - 0.28 * math.copysign(sn * sn, sn * flank))
        if box:
            # The fourth harmonic, and it has to be the fourth: it is the
            # lowest one that leaves both axes where they are and lifts the
            # four diagonals, which is what having corners means. (The second
            # harmonic just stretches the mark into another ellipse.) Written
            # as `1 - cos 4θ` rather than `-cos 4θ` so it only ever adds —
            # a dial that pulled the axes *in* while pushing the diagonals
            # out would put a waist in the middle of each side.
            f += box * 0.5 * (1.0 - math.cos(th * 4.0)) * (0.60 - 0.40 * sn * flank)
        if notch:
            f -= notch * max(0.0, -sn * flank) ** 6.0
        if back:
            # A straight dry edge. `back` is the only dial here that is not a
            # radius: it caps the across-coordinate at the depth the mark
            # already has on its dry flank, so the round end is *cut off*
            # rather than pulled in. The cap is a 6-norm instead of a `min`,
            # which rounds the two corners it makes over a few pixels — a
            # hard polygon vertex in the middle of a brush mark looks sawn.
            y = f * sn * flank
            if y > 0.0:
                y0 = (1.0 - bulge) * (1.0 - 0.34 * back)
                f *= 1.0 / (1.0 + (y / y0) ** 8.0) ** (1.0 / 8.0)
        if splay:
            # The loaded flank reaches further *along* the drag. `bulge` is
            # additive on the radius and cannot do this: growing the radius
            # near the loaded edge moves that edge outward, not sideways, so
            # the widest point of the mark stays at its own middle. This is a
            # shear instead — the half-length is `1 + splay·(-y·flank)`, so
            # the silhouette is a fan whose depth and whose widest half-width
            # are both exactly what they were. Solved on the ray by two
            # fixed-point steps, which at splay < 0.5 agrees with the closed
            # form to four decimals, and `rad` is tabulated once per press.
            g = 1.0
            for _ in range(2):
                a = 1.0 - splay * g * sn * flank
                g = 1.0 / math.sqrt(c * c / (a * a) + sn * sn)
            f *= g
        # The raggedness is *asymmetric*: the loaded flank of a side-loaded
        # stroke is a clean wet edge and only the dry flank frays, which is
        # the difference between a brush mark and a pebble.
        r = ragged * (1.0 + 1.6 * max(0.0, sn * flank))
        return R * (f + w1 * math.cos(th * 2.0 + phase) * r / max(1e-6, ragged)
                    + w2 * math.cos(th * 5.0 - phase * 1.7))

    def tone(u, th):
        # `th` is measured in the press's own circular frame, so u*sin(th)
        # is the distance *across* the stroke in [-1, 1] however much the
        # frame was squashed — which is what "one flank" has to mean.
        v = u * math.sin(th) * flank
        g = 0.5 + 0.5 * v
        # 浓淡 across the stroke, biased so the ink falls away fast once it
        # leaves the loaded edge rather than fading evenly: a 侧锋 mark is
        # dark on one side and nearly gone on the other, not a gradient.
        sh = load * (1.0 - grad * g ** 1.35) + rim * u ** 5 * (1.25 - g)
        cov = 1.0
        if bare:
            cov -= bare * g ** 2.2
        if streak:
            # 擦 — the hairs run out in lines *along* the drag, so the
            # modulation is a function of the across-stroke coordinate only.
            # Two sines of incommensurable period rather than one: a single
            # sine over [-1, 1] gives three evenly spaced bands, which is a
            # corduroy texture and not a brush running dry.
            cov *= 1.0 - streak * g * (0.52 + 0.30 * math.sin(v * 17.3 + phase)
                                       + 0.18 * math.sin(v * 6.1 - phase * 1.4))
        return sh, (cov if cov > 0.0 else 0.0)

    touch(cv, x, y, rad, tone, lo, hi, alpha=alpha, feather=feather,
          squash=max(0.06, W / L), rot=ang, na=na, nr=nr)


# --------------------------------------------------------------------------
# 足 — the jointed line.
#
# A crab's leg is three joints and a shrimp's claw arm is two, and in both
# cases the whole thing is drawn as *one* stroke: the brush goes down at the
# shell, presses at each knuckle, and comes off the paper at the point. So
# it is one call to orchid.py's `sweep`, which accumulates by maximum and
# composites once — `brush_path` would lay ink twice at every joint and a
# leg would come out as a chain of dark beads, which is precisely how a leg
# must not look.
#
# 顿笔 is the knuckle bulge: the half-width swells and the ink darkens for a
# point or two either side of each bend. It is a small effect and it is the
# difference between a bent wire and a leg.
# --------------------------------------------------------------------------

def _chain(x, y, segs, n=6):
    """Points along a chain of (angle, length[, bow]) segments.

    Each segment is *bowed* rather than ruled: `bow` is the total turn it
    makes along its own length, so a leg is made of three shallow arcs. The
    knuckle indices come back with the points, because that is where the
    pressure goes.
    """
    pts = [(x, y)]
    knots = []
    cx, cy = x, y
    for si, seg in enumerate(segs):
        a, ln = seg[0], seg[1]
        bow = seg[2] if len(seg) > 2 else 0.0
        if si:
            knots.append(len(pts) - 1)
        step = ln / n
        aa = a - bow * 0.5
        for _ in range(n):
            aa += bow / n
            cx += math.cos(aa) * step
            cy += math.sin(aa) * step
            pts.append((cx, cy))
    return pts, knots


def limb(cv, rng, x, y, segs, hw0, hw1, lo, hi, alpha=0.90, knuckle=0.55,
         d0=0.55, d1=1.00, taper=1.25, dry=0.0, core=0.30, feather=0.9,
         n=6):
    """One appendage, one stroke. Returns (end point, end angle)."""
    pts, knots = _chain(x, y, segs, n)
    m = len(pts) - 1
    if m < 2:
        return (x, y), (segs[0][0] if segs else 0.0)
    hws, dens = [], []
    for i in range(m + 1):
        t = i / m
        w = hw0 + (hw1 - hw0) * t ** taper
        d = d0 + (d1 - d0) * t
        for ki in knots:
            # 顿笔 — the brush stops and presses where the leg bends.
            b = math.exp(-((i - ki) / 1.35) ** 2)
            w += knuckle * w * b
            d += 0.22 * b
        hws.append(max(0.35, w))
        dens.append(clamp01(d))
    sweep(cv, pts, hws, dens, lo, hi, alpha, feather=feather, core=core,
          dry=dry, dry_ph=rng.random() * TAU)
    ex, ey = pts[-1]
    px, py = pts[-3] if m >= 3 else pts[0]
    return (ex, ey), math.atan2(ey - py, ex - px)


def pincer(cv, rng, x, y, a, s, lo, hi, dark, alpha=0.90, gape=0.46,
           air=None, k=1.0):
    """钳 — the hand: a palm and two fingers with daylight between them.

    The gap matters more than the fingers do. Drawn closed, a claw is a
    blunt dark lump and the creature has a mitten; the gape is the whole
    reason a painted crab looks like it could take hold of something.
    """
    ca, sa = math.cos(a), math.sin(a)
    wet = air["wet"] if air else 1.0
    # 掌 — the palm: a broad press, not a line. A claw is the heaviest mark
    # on a crab and the first version of this drew it narrow, which turned
    # the two fingers into a beak on the end of an antenna.
    press(cv, x + ca * s * 0.28, y + sa * s * 0.28, s * 0.80, s * 0.34, a,
          lo, hi, load=0.90, grad=0.44, flank=-1.0, rim=0.26, bare=0.14,
          tip=0.34, tail=0.24, alpha=alpha, phase=rng.random() * 6.0,
          feather=0.9 * wet)
    bx, by = x + ca * s * 0.46, y + sa * s * 0.46
    # 两指 — the fixed finger and the shorter movable one, short and thick,
    # curving toward each other with daylight between them and both ending
    # in the darkest marks anywhere on the creature. The gap matters more
    # than the fingers do: drawn closed, a claw is a mitten, and the gape is
    # the whole reason a painted crab looks able to take hold of something.
    for sgn, f in ((1.0, 1.0), (-1.0, rng.uniform(0.60, 0.76))):
        a1 = a + sgn * gape * 1.05
        pts, _ = _chain(bx, by, [(a1, s * 0.72 * f, -sgn * gape * 0.80)], 6)
        m = len(pts) - 1
        # thin enough that the daylight survives: at 0.095 s the two fingers
        # are each as wide as the gap between them, and they multiply into one
        # dark hook with a notch rather than into a claw
        hws = [max(0.45, s * (0.072 - 0.060 * (i / m) ** 1.4))
               for i in range(m + 1)]
        dens = [clamp01(0.70 + 0.30 * (i / m)) for i in range(m + 1)]
        sweep(cv, pts, hws, dens, lo, dark, alpha * 0.98, feather=0.8,
              core=0.22)


# --------------------------------------------------------------------------
# 墨色 — how far away a creature is
# --------------------------------------------------------------------------

def _stops(pair, depth, k=1.0):
    """The two stops of a ramp, pushed back toward the paper by distance.

    远淡近浓 — and this module needs it badly, because every creature on the
    sheet is the same creature. Two crabs at one value are a diagram of two
    crabs; one dark and one pale is a near crab and a far one, which is the
    only depth a painting with no water in it can have.
    """
    f = (1.0 - 0.42 * depth) * k
    return dilute(rgb(pair[0]), f), dilute(rgb(pair[1]), f)


# --------------------------------------------------------------------------
# 蟹 — 壳三笔, 八足二螯
# --------------------------------------------------------------------------

def crab(cv, rng, x, y, s, ang, st, air, depth=0.0, k=1.0):
    """一只蟹. `s` is the width of the carapace, `ang` where its front points.

    The crab is built in its own frame: `u` runs across the shell (the way
    the creature travels) and `v` forward (the way it looks). Returns the
    radius of the footprint, which is a great deal bigger than the shell —
    1.35 s, measured off the pixels of one crab alone on a blank canvas at
    forty headings — because the claws, not the shell, are the extent.
    """
    wet = air["wet"]
    al = (0.62 + 0.38 * (1.0 - depth)) * k
    sh_lo, sh_hi = _stops(st["shell"], depth, k)
    lb_lo, lb_hi = _stops(st["limb"], depth, k)
    dark = dilute(mix(I("jiao"), rgb(st["ink"]), 0.35), (1.0 - 0.5 * depth))
    ca, sa = math.cos(ang), math.sin(ang)

    def pt(u, v):
        return x - sa * u + ca * v, y + ca * u + sa * v

    # 横行 — which way the creature is travelling. The legs on that side
    # reach and the ones behind trail, which is the only thing that keeps a
    # top view of a crab from reading as a pressed specimen.
    lead = rng.choice((1.0, -1.0))

    # --- 八足 ----------------------------------------------------------
    #
    # Four to a side, and they go down *before* the shell: their bases sit
    # at the shell's rim, where the overlap of a dark leg and a pale shell
    # reads as a socket. They cannot be tucked under the carapace, because
    # ink multiplies — a 淡墨 shell laid over a 浓墨 leg root does not cover
    # it, it stains it darker. Draw order hides nothing, which is the lesson
    # grape.py paid for three times.
    #
    # Each row is (v along the body, u at the rim, rake). The rakes are
    # deliberately uneven — 忌齐足, four legs at equal angles is a garden
    # rake — and every one of them is **behind the shoulder**: the forward
    # sector of a crab belongs to the claws and nothing else. Legs drawn
    # into it turn the creature into a spider on the spot, which is exactly
    # what the first render of this function produced.
    # `u0` puts every base out **on** the shell rim rather than under it.
    # Tucked inside, a leg root does not disappear behind the carapace: the
    # shell is translucent 淡墨 and ink can only darken, so a dark root under
    # a pale shell stains straight through it and the creature grows eight
    # dark spokes across its own back. draw order hides nothing.
    #
    # Every (u0, v0) is checked against the carapace's own silhouette, as a
    # fraction of its radius in that direction — `press`'s `rad`, which was an
    # ellipse of 1.00 s by 0.62 s when these numbers were first set and is a
    # fan now. The rearmost pair was at (0.38, -0.23), 1.13 of the rim, and it
    # showed: on a 230-pixel crab the last leg on each side began in bare
    # paper a few pixels clear of the shell and read as a detached bracket
    # lying beside the creature. So when the shell became a fan, every root
    # moved to hold the fraction it had — out at the front, where the shell
    # grew, and in at the back, where it narrowed. The rear pair converging on
    # the narrow back is anatomy as well as arithmetic.
    #
    # Holding the fraction stopped being enough once `back` cut the rear edge
    # off straight: the rim came forward to 0.161 s and the last row, at
    # v = -0.21, was no longer merely poking through the outline — it was
    # 0.05 s *behind the whole shell*, which is the detached bracket again by
    # another route. So the ladder is re-pitched along the new side edge,
    # from just behind the front corner to the rear corner, and the roots
    # measure 1.00, 1.02, 1.08, 1.07 of the rim — on it, or a whisker out.
    rows = ((0.150, 0.49, 0.35), (0.045, 0.44, 0.00),
            (-0.055, 0.38, -0.45), (-0.145, 0.29, -0.95))
    for side in (1.0, -1.0):
        # 横行 wants the two sides *unequal*, and at ±0.12 they were not
        # unequal enough to read: eight legs of a length, evenly fanned, is a
        # spider seen from above however carefully each one is jointed. The
        # leading side now reaches a fifth further than nominal and the
        # trailing side pulls a fifth shorter, so the creature is caught
        # mid-scuttle — one side pushing, one side gathered under the shell.
        reach = 1.0 + (0.21 if side == lead else -0.19)
        for i, (v0, u0, rake) in enumerate(rows):
            bu = side * u0 * s * rng.uniform(0.96, 1.04)
            bv = v0 * s * rng.uniform(0.94, 1.06)
            bx, by = pt(bu, bv)
            a1 = ang + side * (math.pi * 0.5 - rake - rng.gauss(0.0, 0.08))
            # 两笔一足 — the knee folds hard away from the front and the
            # pointed last joint keeps folding, so a leg is a bracket and
            # not a spoke. The fold has to be *sharp*: a gently bent leg
            # reads as a tentacle however well it tapers.
            b1 = side * rng.uniform(0.90, 1.25)
            b2 = side * rng.uniform(0.34, 0.66)
            f = reach * rng.uniform(0.88, 1.10) * (1.0 - 0.07 * i)
            # kept a touch under a quarter of the shell's width per joint:
            # longer than that and eight of them out-draw the body, which is
            # a spider's proportion — a crab's legs are short, thick and
            # crowded round a shell that dominates them.
            segs = [(a1, s * 0.22 * f, side * rng.uniform(0.04, 0.14)),
                    (a1 + b1, s * 0.23 * f, side * rng.uniform(0.10, 0.24)),
                    (a1 + b1 + b2, s * 0.15 * f, side * rng.uniform(0.0, 0.2))]
            limb(cv, rng, bx, by, segs,
                 s * rng.uniform(0.023, 0.029), s * 0.0030,
                 lb_lo, lb_hi, alpha=al * 0.76,
                 knuckle=rng.uniform(0.40, 0.65),
                 d0=0.58, d1=1.00, taper=1.45, dry=air["dry"] * 0.5,
                 feather=0.8 * wet)

    # --- 壳 ------------------------------------------------------------
    #
    # ONE mass, drawn **across** the creature, plus two margin strokes that
    # are lines and not masses. Getting here took six versions and the last
    # three all failed the same way, which is the lesson:
    #
    #   1. three presses lying front-to-back, side by side, *touching* so the
    #      seams would be paper. Three eggs in a row.
    #   2. the same, elongated 1.7 : 1 and overlapped by a quarter. Still
    #      lobes — and because the middle press has to be the pale one (the
    #      pale dome is the light on the shell), the two dark outer presses
    #      read as two circles with a gap between them.
    #   3. one transverse press spanning the whole carapace, with two
    #      shoulder presses only widening it at the front. Correct geometry,
    #      and the shoulders still read as two objects: heavy, they were dark
    #      ears on the ends of the shell; light, they were pale balloons
    #      glued to it; soft-edged, the balloons got bigger. Every setting
    #      between those is one of those two pictures.
    #
    # Which is grape.py's lesson and the shrimp's, arrived at a third time: a
    # `press` wears a complete contour — `rim` darkens its whole outline,
    # `tip`/`tail` round its ends — so **two presses are two things**, and no
    # amount of overlapping or fading dissolves the seam. 墨块与骨线是两件事:
    # build the mass with one stroke, and draw everything else *as lines*.
    #
    # So the shoulders moved out of the ink and into the outline. `bulge`
    # makes the press fat on its loaded flank and lean on its dry one, which
    # is what a side-loaded brush actually does and which is also exactly a
    # crab's proportion — wide across the front, tapering behind. One press
    # now has the whole silhouette and there is no seam to hide.
    #
    # And for a long time that silhouette was still an egg, reported as such.
    # Raising `bulge` (0.26 → 0.38) was the first attempt and it failed twice
    # over: fatter pears, and the margin strokes left stranded inside the
    # grown mass as a dark slot across the shell. The measurement says why it
    # could not have worked. `bulge` adds to the *radius* near the loaded
    # edge, which pushes that edge forward rather than sideways, so the mark's
    # widest point stays at its own middle — at +0.03 s of a 0.37 s front
    # depth — and a shape whose widest point is in the middle and whose ends
    # are both round is an egg no matter how unequal the two ends are. A 蟹壳
    # is widest **at the front corners**: a flat, slightly dented front edge
    # between the eyes, a hard corner at each end of it, straight sides raking
    # back from there and a narrow flat back. Three dials, measured rather
    # than guessed — `splay` shears the mass forward so the widest point lands
    # at +0.13 s, `box` puts the corners on it, `notch` dents the front edge
    # between the eyes — and the half-width of the front quarter against the
    # rear quarter goes 1.37 → 1.57. `L` comes down to 0.91 s to pay for the
    # shear, so the carapace is exactly as wide as it was: `s` still means the
    # width of the shell, and everything else on the creature is placed in s.
    #
    # The other half of the fix is bookkeeping, and skipping it is what made
    # the first attempt fail: **the silhouette is a surface other marks are
    # placed against.** Eight leg roots sit on its rim, two claw arms start
    # just inside it, the 侧缘两笔 trace its shoulder, the 口 and 目 sit under
    # its front edge. Each of those was measured against the old outline as a
    # fraction of its radius in that direction and moved to hold the same
    # fraction against the new one, which is the whole of §5.3's lesson:
    # change the shape and the shape's dependents move with it.
    #
    # `flank=1.0`, not -1.0, and the sign is worth deriving rather than
    # guessing. `press` loads the side where its local across-coordinate is
    # negative; its local *along* axis is `rot`, so its across axis is
    # `rot + 90°`. Here `rot = ang + 90°`, so across is `ang + 180°` —
    # backward — and `flank=-1.0` therefore loads the **rear**. That is what
    # a dozen renders of this function actually drew: a shell with its weight
    # of ink behind, a pale front, and a 口 dab hanging off the bare front
    # edge with no mass to sit in.
    #
    # The *weight* is the other thing, and it is not a fault of drawing at
    # all. A shell at `load=0.72, grad=0.80` is 0.72 of the ramp on its loaded
    # edge and 0.14 on the other, under `alpha=0.66` — so the carapace came
    # out at the value of the paper while the legs, drawn at full alpha into a
    # near-black stop, came out at the value of ink. A crab whose body is
    # lighter than its own legs is a crab made of wire, and on the sheet it
    # read as a tick. 壳是全身最重的一块: the shell is the heaviest mass on the
    # creature and every other mark hangs off it.
    # 壳宜润不宜燥 — and the last fault left in this stroke was the 擦. At
    # `streak=0.24` the dry-flank modulation's short term runs about five and
    # a half periods across the press, and across a 200-pixel carapace those
    # five bands are *visible as bands*: the shell came out corrugated, a
    # scallop or a woodlouse rather than a crab, and that texture was the
    # loudest thing about the creature at thumbnail size. 擦 belongs on a leg
    # dragged over dry paper; a carapace is painted wet in one go and is
    # smooth. So the streak is cut to a trace and left to 风致 to restore in
    # the dry airs, where the whole sheet is scratchy and it reads as paper.
    #
    # With the bands gone the shell could also take its proper weight —
    # 壳是全身最重的一块 — so `load` goes up, `bare` (which was lifting a
    # quarter of the coverage off the rear flank) comes down, and `rim`
    # carries more: the mass is dark at its wet front edge and keeps a pale
    # dome behind, which is one shell lit from the front rather than a grey
    # oval of even value lighter than its own legs.
    # 壳不是蛋. Raising `bulge` was the first attempt at that and it failed
    # twice over — fatter pears, and the margin strokes left stranded inside
    # the grown mass as a dark slot across the shell. The measurement says why
    # it could not work: `bulge` adds to the *radius* near the loaded edge,
    # which pushes that edge forward rather than sideways, so the mark's
    # widest point stays at its own middle — +0.03 s of a 0.37 s front depth —
    # and a shape widest in the middle with two round ends is an egg however
    # unequal the ends are. A 蟹壳 is widest **at the front corners**: a flat,
    # slightly dented front edge between the eyes, a hard corner at each end
    # of it, straight sides raking back and a narrow flat back. So `splay`
    # shears the mass forward until the widest point lands at +0.18 s of the
    # 0.34 s front depth, `box` puts the corners on it and `notch` dents the
    # front edge; the half-width of the front quarter against the rear quarter
    # goes 1.37 → 1.64. `L` comes down to 0.85 s to pay for the shear, so the
    # shell is exactly as wide as it was — 0.50 s either side of the middle —
    # and `s` still means the width of the carapace.
    #
    # Those three leave the *back* round, and a broad front on a round back is
    # a teardrop. That showed up only in the third size — a 3× crop of a
    # finished sheet, where the front read correctly and the rear read as a
    # cap. No radius dial can fix it: pulling a radius in near the dry pole
    # makes the end pointed, not flat. `back` is the one term here that is not
    # a radius — it caps the across-coordinate — and at 0.62 the rear edge is
    # straight (within 8% of its depth) out to 58% of the half-width instead
    # of 29%. It costs rear depth, 0.206 → 0.161 s, so `W` goes 0.62 → 0.66 to
    # put the body depth back: front 0.365, rear 0.161, width against depth
    # 1.90 where the old egg was 1.84. Wide, blunt-backed, front-heavy.
    #
    # `ragged` comes down with it, 0.08 → 0.05, and that is not tidiness. The
    # raggedness is a second harmonic, and a second harmonic *is* an ellipse:
    # at 0.08 of R it was ±0.04 s of elongation in an arbitrary direction,
    # which is the same order as the shape being drawn here, so it rounded the
    # corners back off again at about one phase in three. Six variants of the
    # bare press rendered side by side at 300 px is what showed that — the
    # mark at the size the mark wants, with nothing else on the paper.
    #
    # The other half of the fix is bookkeeping, and skipping it is what made
    # the first attempt fail: the silhouette is a surface that other marks are
    # placed against. Eight leg roots sit on its rim, two claw arms start just
    # inside it, the 侧缘两笔 trace its shoulder, the 口 and 目 sit under its
    # front edge. Every one of those was measured against the old outline as a
    # fraction of its radius in that direction and moved to hold the same
    # fraction of the new one.
    ph = rng.random() * 6.0
    press(cv, *pt(0.0, s * rng.uniform(-0.03, 0.01)),
          L=s * 0.85 * rng.uniform(0.97, 1.04), W=s * 0.66, ang=ang + TAU / 4,
          lo=sh_lo, hi=sh_hi, load=0.93 * k, grad=0.42, flank=1.0,
          rim=0.42, bare=0.13, streak=0.07 + air["dry"] * 0.30,
          tip=0.08, tail=0.08, ragged=0.05, bulge=0.22,
          splay=0.42 + rng.uniform(-0.05, 0.05), box=0.22,
          notch=0.06 + rng.uniform(0.0, 0.03),
          back=0.62 + rng.uniform(-0.04, 0.04), phase=ph,
          alpha=min(1.0, al * 0.97), feather=1.35 * wet)

    # 侧缘两笔 — the front-side margins, raking back from each front corner.
    #
    # These are the two strokes the shoulder presses were trying to be, done
    # as lines: a 蟹壳 has a hard serrated margin from the eye socket back to
    # the widest point, and two short dark strokes along it do the whole job
    # the second and third presses could not. They sit *inside* the mass,
    # which is the only place a drawn edge can sit when ink multiplies — laid
    # on the silhouette they would be a wire outline round a grey blob.
    for side in (1.0, -1.0):
        q = rng.uniform(0.94, 1.06)
        # starting out at 0.15 s rather than 0.10 s, which leaves a span of
        # pale shell between the two of them for the 口 and 目 to read in.
        # Run in to the centre line they close into one dark bar across the
        # front and the face goes with it.
        #
        # The four nodes follow the fan: along the flat front edge, out to the
        # corner, round it, and a little way down the straight side — which is
        # where a 蟹壳's serrated margin actually runs. Measured as a fraction
        # of the rim they are 0.77, 0.82, 0.93, 0.93, so the stroke lies
        # inside the mass throughout and leans on the rim as it goes back.
        # Laid on the silhouette instead it would be a wire outline round a
        # grey blob; left where it was before the shell changed shape, it cut
        # across the new shoulder as a slot. The forward coordinates carry the
        # 0.62 → 0.66 deepening of `W` with them; the lateral ones barely move,
        # because `back` works on the dry end and the shoulder is at the wet.
        pts = [pt(side * s * 0.19, s * 0.303 * q),
               pt(side * s * 0.38, s * 0.250 * q),
               pt(side * s * 0.45, s * 0.128 * q),
               pt(side * s * 0.35, -s * 0.021 * q)]
        # Drawn at 0.70 alpha into the shell's own dark stop these came out
        # *heavier than the mass they sit in*, and two marks darker than
        # their surroundings are two marks — at thumbnail size a pair of
        # black dashes floating on the shell, which is a beetle's elytra.
        # An edge is a thickening of the mass, not an object on top of it,
        # so they are thinner, paler, and no longer carry the 顿笔 swell.
        sweep(cv, pts,
              [s * 0.008, s * 0.014, s * 0.012, s * 0.005],
              [0.40, 0.66, 0.56, 0.22], sh_lo, sh_hi,
              al * 0.52, feather=1.0 * wet, core=0.22,
              dry=air["dry"] * 0.6)


    # 口 — one dark dab at the front middle, where the mouthparts are.
    #
    # This replaces an arc drawn right across the front of the shell, which
    # is how this started and which was wrong for a reason worth keeping: a
    # hard line laid over three translucent presses has nothing to sit in,
    # so it floated clear of the ink as a wire hoop, and when it was shifted
    # inside the margin it became a crack across the shell instead. The
    # front edge is better left to the middle press's own loaded rim, with
    # one dark mark to tell the viewer which end is the front.
    # 0.289 s rather than 0.300: the notch took 0.032 s off the front edge at
    # the centre line and this dab's far end was already within a few pixels
    # of it, so it moved back by exactly what the edge did — then forward
    # again by most of it when `W` deepened. Its far tip sits at 0.923 of the
    # rim, which is the fraction it held on the old shape.
    mxp, myp = pt(0.0, s * 0.289)
    press(cv, mxp, myp, s * 0.18, s * 0.11, ang + rng.gauss(0, 0.2),
          lb_lo, lb_hi, load=0.86, grad=0.30, flank=1.0, rim=0.2,
          tip=0.3, tail=0.3, alpha=al * 0.86, feather=0.9 * wet)

    # 目 — two tiny dark uprights either side of it. A crab's eyes read as
    # two short strokes on stalks, not as dots; set wide apart they look
    # like nostrils, and set long they look like horns.
    for side in (1.0, -1.0):
        ex, ey = pt(side * s * 0.085, s * 0.289)
        tx, ty = pt(side * s * 0.098, s * 0.353)
        sweep(cv, [(ex, ey), ((ex + tx) * 0.5, (ey + ty) * 0.5), (tx, ty)],
              [s * 0.018, s * 0.015, s * 0.010],
              [0.80, 0.95, 1.0], lb_lo, dark, min(1.0, al * 1.05),
              feather=0.7, core=0.2)

    # --- 二螯 ----------------------------------------------------------
    #
    # 螯有大小 — one of them is the big one, by a quarter at least. And they
    # come forward and *in*: a claw held straight out to the side is a
    # semaphore, and two of them make the creature twice as wide as it is.
    big = rng.choice((1.0, -1.0))
    for side in (1.0, -1.0):
        f = 1.0 if side == big else rng.uniform(0.68, 0.82)
        # out at the shoulder, 0.44 s rather than 0.36: the arm is drawn dark
        # and the shell is translucent 淡墨, so every pixel of arm that starts
        # inside the mass stains straight through it. The fan put its shoulder
        # a long way further out, and the root went with it.
        bx, by = pt(side * s * 0.44, s * 0.18)
        # out to the side, then forward and in — a claw carried straight out
        # makes the creature twice as wide as it is and reads as semaphore
        # and the arm is carried well forward of the leg sector. Swung out at
        # a right angle it lands among the first pair of legs, and then the
        # claw is simply a ninth leg with a lump on the end.
        a1 = ang + side * rng.uniform(0.62, 0.90)
        a2 = ang + side * rng.uniform(0.04, 0.26)
        end, ea = limb(cv, rng, bx, by,
                       [(a1, s * 0.19 * f, side * 0.12),
                        (a2, s * 0.20 * f, -side * 0.16)],
                       s * 0.056 * f, s * 0.038 * f, lb_lo, lb_hi,
                       alpha=al, knuckle=0.26, d0=0.60, d1=0.92, taper=1.0,
                       dry=air["dry"] * 0.4, feather=0.85 * wet)
        pincer(cv, rng, end[0], end[1], ea, s * 0.50 * f, sh_lo, lb_hi,
               dark, alpha=al, gape=rng.uniform(0.34, 0.54), air=air, k=k)

    return s * 1.35


# --------------------------------------------------------------------------
# 虾 — 身透明, 须最长
# --------------------------------------------------------------------------

def shrimp(cv, rng, x, y, s, ang, st, air, depth=0.0, k=1.0, curl=None):
    """一只虾. `s` is the body length; (x, y) is the carapace, `ang` the way
    the head points. Returns the radius of the body's footprint — the
    whiskers reach half as far again and are allowed off the paper.
    """
    wet = air["wet"]
    al = (0.55 + 0.45 * (1.0 - depth)) * k
    bd_lo, bd_hi = _stops(st["shell"], depth, k)
    lb_lo, lb_hi = _stops(st["limb"], depth, k)
    dark = dilute(mix(I("jiao"), rgb(st["ink"]), 0.3), 1.0 - 0.5 * depth)
    ca, sa = math.cos(ang), math.sin(ang)

    def pt(f, o=0.0):
        return x + ca * f - sa * o, y + sa * f + ca * o

    # 弯 — a shrimp is never straight. The curl tightens toward the tail,
    # which is what makes the creature look like it is swimming rather than
    # lying on a plate.
    #
    # 虾有伸屈 — but it is not always bent the *same* amount, and that was
    # the duller fault: a range of 0.09–0.19 is a range of one shape, so
    # five shrimp on a sheet were five copies of one silhouette at five
    # sizes. So the draw is a mixture of two postures rather than one band.
    # 伸 — about a third of them are nearly straight, a shrimp *darting*,
    # which is the shape that carries speed. 屈 — the rest are bent harder
    # than before, a shrimp hanging in the water with its tail tucked. The
    # two read as two creatures doing two things.
    if curl is None:
        curl = (rng.uniform(0.03, 0.09) if rng.random() < 0.32
                else rng.uniform(0.13, 0.26)) * rng.choice((-1.0, 1.0))

    # --- 腹 — the abdomen: a mass, and then its joints ------------------
    #
    # The first version of this drew the abdomen as nothing but five or six
    # overlapping presses, trusting the overlaps to pool into segments. They
    # do not. They pool into a **pinecone of pale beads** — which is one of
    # the five faults that made fish.py's companion shrimp unusable at
    # this size, reproduced exactly, from different code. Worth knowing why:
    # a press darkens its *whole* outline (`rim`) and rounds off both ends
    # (`tip`, `tail`), so every plate arrives wearing its own complete
    # contour and no amount of overlapping will dissolve it. Nine presses
    # with their own edges are nine objects.
    #
    # So the abdomen is built as two different things:
    #
    #   mass   presses with no rim and no end-taper, overlapping by two
    #          thirds, all loaded on the *same* flank. Having no contour of
    #          their own they fuse into one tapering tube, dark along the
    #          back and open 留白 along the belly.
    #   节     one short dark stroke drawn *across* the body at each joint.
    #          Segments are a thing you draw, once the mass is a mass.
    #
    # Which is the general lesson of the whole family restated: 墨块 and
    # 骨线 are different jobs and want different tools.
    nseg = rng.randint(5, 6)
    a = ang + math.pi
    px, py = pt(s * 0.04)
    spine = []
    for i in range(nseg):
        ln = s * (0.108 - 0.008 * i)
        wd = s * (0.185 - 0.020 * i)
        spine.append((px, py, a, ln, wd))
        a += curl * (0.7 + 0.28 * i)
        px += math.cos(a) * ln
        py += math.sin(a) * ln
    lit = rng.choice((1.0, -1.0))
    for i, (sx, sy, aa, ln, wd) in enumerate(spine):
        mx, my = sx + math.cos(aa) * ln * 0.5, sy + math.sin(aa) * ln * 0.5
        press(cv, mx, my, ln * 3.8, wd, aa, bd_lo, bd_hi,
              load=0.36 + 0.05 * i, grad=0.62, flank=lit, rim=0.0,
              bare=0.30, streak=0.16 + air["dry"] * 0.5, tip=0.0, tail=0.0,
              ragged=0.03, phase=i * 2.3 + rng.random(),
              alpha=al * 0.44, feather=1.9 * wet)
    # 节 — the joints, each a short stroke across the back, and only the
    # first three. Marked at every joint and run out to the rim they become
    # bristles poking through the silhouette and the shrimp is a caterpillar;
    # kept inside the mass and only near the head, where a real shrimp's
    # plates actually overlap most visibly, they read as 节. They also stop
    # well short of the belly — a line all the way across is a rope ladder,
    # and the open side is what 虾身透明 means.
    for i, (sx, sy, aa, ln, wd) in enumerate(spine[1:4], 1):
        n1, n2 = -math.sin(aa) * lit, math.cos(aa) * lit
        # struck from the loaded flank *inward*, not the other way about:
        # the belly is bare paper, so a tick that starts there starts in
        # nothing and sticks out of the silhouette as a bristle
        pts = [(sx + n1 * wd * 0.32, sy + n2 * wd * 0.32),
               (sx + n1 * wd * 0.10, sy + n2 * wd * 0.10),
               (sx - n1 * wd * 0.12, sy - n2 * wd * 0.12)]
        sweep(cv, pts, [s * 0.015, s * 0.011, s * 0.005],
              [0.95, 0.72, 0.40], bd_lo, bd_hi, al * 0.50,
              feather=0.9 * wet, core=0.28)

    # --- 尾扇 — three or four flicks spread into a fan ------------------
    #
    # The roots are staggered back along the last segment rather than all
    # struck from one point. Struck from one point, three or four strokes
    # 7 px thick at the root multiply into a dark knob and the shrimp looks
    # like it has been tied off — which is what the first version did.
    nt = rng.randint(3, 4)
    for j in range(nt):
        f = (j - (nt - 1) * 0.5) / max(1.0, nt - 1.0)
        back = s * 0.035 * abs(f) * 2.0
        tx = px - math.cos(a) * back
        ty = py - math.sin(a) * back
        aa = a + f * rng.uniform(0.72, 1.00)
        ln = s * rng.uniform(0.20, 0.26) * (1.0 - 0.16 * abs(f) * 2.0)
        pts, _ = _chain(tx, ty, [(aa, ln * rng.uniform(0.86, 1.14),
                                  f * rng.uniform(0.5, 0.9))], 6)
        m = len(pts) - 1
        sweep(cv, pts, [max(0.4, s * (0.026 - 0.023 * (i / m) ** 0.8))
                        for i in range(m + 1)],
              [clamp01(0.70 - 0.38 * (i / m)) for i in range(m + 1)],
              bd_lo, bd_hi, al * 0.72, feather=1.1 * wet, core=0.3)

    # --- 头胸 — two presses making a wedge, blunt forward ---------------
    #
    # A third of the creature's length. Drawn any smaller the carapace
    # merges with the first abdominal plate and the shrimp has no head at
    # all, which is what happened the first time. And it is the *palest*
    # part of the body, because the next mark is the one that matters.
    hx, hy = pt(s * 0.15)
    press(cv, hx, hy, s * 0.42, s * 0.25, ang, bd_lo, bd_hi,
          load=0.34, grad=0.72, flank=lit, rim=0.0, bare=0.50,
          streak=0.16 + air["dry"] * 0.6, tip=0.26, tail=0.20, ragged=0.04,
          phase=rng.random() * 6.0, alpha=al * 0.50, feather=1.6 * wet)
    kx, ky = pt(s * 0.04, lit * s * 0.030)
    press(cv, kx, ky,
          s * 0.26, s * 0.16, ang + lit * 0.10, bd_lo, bd_hi,
          load=0.42, grad=0.62, flank=-lit, rim=0.0, bare=0.40,
          phase=rng.random() * 6.0, alpha=al * 0.44, feather=1.4 * wet)

    # 重墨一点 — the dab, and the single most important mark on the
    # creature. 齐白石's shrimp are transparent because of this one blot of
    # 浓墨 in the middle of the carapace: the eye reads the pale shell as
    # glass *because* something dark is visible through it. Without it the
    # body is a grey smear whatever else is done to it.
    dx, dy = pt(s * 0.155, lit * s * 0.012)
    disc(cv, dx, dy, s * rng.uniform(0.058, 0.072),
         dilute(mix(I("nong"), rgb(st["ink"]), 0.4), 1.0 - 0.45 * depth),
         min(1.0, al * rng.uniform(0.80, 0.96)),
         squash=rng.uniform(0.70, 0.88), soft=0.38)

    # 额剑 — the rostrum, one fine line forward over the eyes
    rx, ry = pt(s * 0.28)
    ex2, ey2 = pt(s * 0.46, -lit * s * 0.020)
    sweep(cv, [(rx, ry), ((rx + ex2) * 0.5, (ry + ey2) * 0.5),
               (ex2, ey2)],
          [s * 0.024, s * 0.014, s * 0.004], [0.90, 0.75, 0.50],
          lb_lo, lb_hi, al * 0.90, feather=0.8, core=0.25)

    # 目 — two dark dashes laid *across* the head. Round dots turn the head
    # into a face; 齐白石's are two commas at right angles to the body.
    for side in (1.0, -1.0):
        ox, oy = pt(s * 0.255, side * s * 0.055)
        aa = ang + math.pi * 0.5 * side
        pts = [(ox - math.cos(aa) * s * 0.012, oy - math.sin(aa) * s * 0.012),
               (ox + math.cos(aa) * s * 0.022, oy + math.sin(aa) * s * 0.022)]
        sweep(cv, pts, [s * 0.026, s * 0.014], [0.9, 1.0], lb_lo, dark,
              min(1.0, al * 1.1), feather=0.7, core=0.2)

    # --- 螯 — the long arms, and the little walking legs ---------------
    big = rng.choice((1.0, -1.0))
    for side in (1.0, -1.0):
        f = 1.0 if side == big else rng.uniform(0.74, 0.90)
        bx, by = pt(s * 0.10, side * s * 0.055)
        a1 = ang + side * rng.uniform(0.25, 0.55) + rng.gauss(0, 0.08)
        a2 = a1 + side * rng.uniform(0.35, 0.70)
        end, ea = limb(cv, rng, bx, by,
                       [(a1, s * 0.22 * f, side * 0.08),
                        (a2, s * 0.24 * f, side * 0.12)],
                       s * 0.015 * f, s * 0.009 * f, lb_lo, lb_hi,
                       alpha=al * 0.92, knuckle=0.45, d0=0.45, d1=0.80,
                       dry=air["dry"] * 0.3, feather=0.8 * wet)
        pincer(cv, rng, end[0], end[1], ea, s * 0.15 * f, bd_lo, lb_hi,
               dark, alpha=al * 0.95, gape=rng.uniform(0.30, 0.52),
               air=air, k=k)
    # 步足 — four or five hairs under the carapace, all swept back. They are
    # not drawn for themselves: they are what stops the body from looking
    # like it is floating with nothing underneath it.
    for i in range(rng.randint(3, 4)):
        side = 1.0 if i % 2 else -1.0
        bx, by = pt(s * (0.07 - 0.035 * i), side * s * 0.055)
        aa = ang + math.pi * (0.5 if side > 0 else -0.5) * rng.uniform(
            0.55, 0.95) + math.pi * (0.10 if side > 0 else -0.10)
        ln = s * rng.uniform(0.06, 0.10)
        pts, _ = _chain(bx, by, [(aa, ln, side * 0.6)], 4)
        m = len(pts) - 1
        sweep(cv, pts, [max(0.35, s * (0.011 - 0.008 * (i2 / m)))
                        for i2 in range(m + 1)],
              [clamp01(0.5 + 0.3 * (i2 / m)) for i2 in range(m + 1)],
              lb_lo, lb_hi, al * 0.6, feather=0.8, core=0.25)

    # --- 须 — the whiskers --------------------------------------------
    #
    # Longer than the shrimp, finer than anything else on the sheet, and all
    # curving the *same* way, because they are all being dragged by the same
    # water. The curvature grows as t², so each one leaves the head almost
    # straight and finishes in a real bend — and each is one `sweep` over
    # twenty points, which is the whole reason this module does not reuse
    # fish.py's shrimp: four-point polyline whiskers show their corners
    # the moment the creature is bigger than a thumbnail.
    sw = rng.choice((-1.0, 1.0))
    nw = rng.randint(4, 6)
    for i in range(nw):
        long_ = i < 2
        a1 = ang + rng.uniform(-0.55, 0.55)
        bx, by = pt(s * rng.uniform(0.20, 0.27), rng.uniform(-0.05, 0.05) * s)
        ln = s * (rng.uniform(1.00, 1.55) if long_ else rng.uniform(0.30, 0.60))
        bend = sw * rng.uniform(0.6, 1.3) * (1.0 if long_ else
                                             rng.uniform(0.5, 1.0))
        nn = 20 if long_ else 12
        # the hair is *integrated* along its own turning tangent rather than
        # laid out on an arc of fixed radius: the curvature grows as t², so
        # it leaves the head almost straight and finishes in a real bend
        pts = [(bx, by)]
        cx2, cy2 = bx, by
        for j in range(1, nn + 1):
            t = j / nn
            aa = a1 + bend * t * t
            cx2 += math.cos(aa) * ln / nn
            cy2 += math.sin(aa) * ln / nn
            pts.append((cx2, cy2))
        m = len(pts) - 1
        hw = s * (0.0085 if long_ else 0.0070)
        sweep(cv, pts, [max(0.32, hw * (1.0 - 0.72 * (j / m)))
                        for j in range(m + 1)],
              [clamp01(0.55 - 0.3 * (j / m)) for j in range(m + 1)],
              lb_lo, lb_hi,
              (0.30 if long_ else 0.38) * al * rng.uniform(0.75, 1.0),
              feather=0.75, core=0.4, dry=air["dry"] * 0.5,
              dry_ph=rng.random() * TAU)

    return s * 0.52


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

# The subject word has to agree with what actually got painted. Split three
# ways and chosen from the counts, because a sheet of two crabs and one shrimp
# titled 三虾写意 is not a stylistic liberty, it is a label contradicting the
# picture it is pasted on — and 双蟹 / 三虾 are checked against the counts
# exactly, since they name a number.
TITLE_XIE = ["墨蟹", "双蟹", "蟹趣", "秋蟹", "横行", "芦蟹"]
TITLE_XIA = ["群虾", "虾趣", "活虾", "三虾", "清波"]
TITLE_BOTH = ["虾蟹", "水族", "塘趣", "草间"]
TITLE_A = TITLE_XIE + TITLE_XIA + TITLE_BOTH
TITLE_B = ["图", "小景", "清趣", "写意", "图卷", "册页", "小品"]
ACTION = ["写生", "戏作", "写意", "拟古", "灯下戏墨", "南窗写虾", "并记",
          "点墨", "水边所见"]
SEAL_WORDS = ["墨缘", "写心", "横行", "秋水", "八足", "二螯", "知水", "芦塘",
              "活泼", "一爪", "白石", "清波"]


def _subject(rng, ns, nc):
    """The subject half of a title, agreeing with the creatures on the sheet."""
    if nc and ns:
        pool = TITLE_BOTH + (TITLE_XIE if nc >= ns else TITLE_XIA)
    elif nc:
        pool = TITLE_XIE + ["水族", "塘趣", "草间"]
    else:
        pool = TITLE_XIA + ["水族", "塘趣", "草间"]
    pool = [w for w in pool
            if (w != "双蟹" or nc == 2) and (w != "三虾" or ns == 3)]
    return pool[rng.randrange(len(pool))]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into crab_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
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
    for mod in ("crab_font", "fish_font", "grape_font", "orchid_font",
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
    """Measure the block, then let the 留白 decide where it goes.

    Same machinery as the other seven painters. What is particular here is
    that a 虾蟹 sheet is mostly empty paper by design, and the writing is
    doing real compositional work: it is the weight that keeps a shoal of
    pale creatures from drifting out of the picture.
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
    r0 = min(W, H) * 0.14
    for fy in (0.34, 0.52, 0.70):
        occ.add(W * 0.5, H * fy, r0)
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.5)
        paint_column(cv, rng, FONT, title, bx + gap + ss, by, gs, lead=lead)
        end = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, ss,
                           lead=lead_s, alpha=(0.60, 0.84))
        paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                   bx + (ss - seal_s) * 0.5, end + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
    if rng.random() < 0.5:                        # 压角章
        s2 = S * rng.uniform(0.036, 0.050)
        for cx, cy in ((W * 0.07, H * 0.88), (W * 0.88 - s2, H * 0.88),
                       (W * 0.07, H * 0.07), (W * 0.88 - s2, H * 0.07)):
            if occ.overlap(cx, cy, cx + s2, cy + s2, s2 * 0.5) == 0.0:
                paint_seal(cv, rng, FONT,
                           SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                           cx, cy, s2)
                break


# --------------------------------------------------------------------------
# 章法 — composition
# --------------------------------------------------------------------------

def _hold(p, lo, hi, k=4.0):
    """Squeeze a value toward the middle of [lo, hi] — bamboo.py's soft
    clamp. A hard clamp piles the outliers up exactly on the limit, which
    makes the thing it was meant to prevent perfectly visible."""
    m, r = (lo + hi) * 0.5, (hi - lo) * 0.5
    if r <= 0.0:
        return m
    t = (p - m) / r
    return m + r * math.tanh(t * k / 2.0) if abs(t) > 1.0 else p


def _hold2(p, lo, hi):
    """A hard clamp, for the one quantity that has no room to be soft about
    it: how far the whole group may be slid across the paper. `lo > hi` means
    the group is wider than the legal band, and then it does not move."""
    if lo >= hi:
        return 0.0
    return lo if p < lo else (hi if p > hi else p)


def creature_sites(rng, W, H, S, want, flow, spread, low):
    """Where each creature goes, how big it is and which way it points.

    `want` is a list of ("shrimp"|"crab", size) in the order they should get
    the room — biggest first, 主 before 从. Returns the same list with a
    position, a heading and a depth attached.

    A shoal is not a scatter. 群虾 came out messy for two reasons that
    compound, and both of them were here:

    * the positions were an **isotropic** Gaussian cloud, so the group had
      no direction of its own — a blob of shrimp rather than a stream of
      them; and
    * the headings were `flow + gauss(0, spread)` clamped to **±0.9 rad**,
      a 100° fan. A fan that wide has no current left in it: at ±50° two
      shrimp differ by a right angle and the eye reads each one separately.

    So the shoal now has a frame of its own — a 势 line at `axis`, a little
    oblique to the current the shrimp swim on — and every shrimp position is
    drawn in *that* frame, long along it and narrow across it. Oblique and
    not parallel: strung along their own heading they would be in each
    other's wake, which is 排队; at 0.3–0.8 rad off it they are 雁阵, a
    staggered column, which is the arrangement that reads as ordered without
    reading as ruled. The fan is then ±0.45 rad, about a quarter of what it
    was.

    The other rules are 忌, and all of them are about the fact that every
    creature on the sheet is the same creature:

    * **忌排队.** What keeps the staggered column from becoming a file is
      the across-axis spread (σ ≈ 0.072 S for the knot, against 0.145 S
      along it) and the fact that the gaps are random, not stepped — plus
      the stragglers, which are thrown 0.40–0.66 S out along the axis and
      alternate ends. This replaces the 三点一线
      rejection test borrowed from fish.py, which cannot survive here:
      the composition is now *deliberately* a line, and a test that refuses
      lines would refuse the composition.
    * **聚散.** Most of the shrimp are in one knot and the last one or two
      are set well off along the axis — one straggler behind, one scout
      ahead. Equal company is no company: 聚 needs 散 to be read as 聚.
    * **忌对称 / 忌对头.** Nose to nose needs nearly π and so cannot happen
      inside a ±0.45 fan, except for the one 回首 shrimp, which is checked.
    * **忌出界.** A creature may not be cut by the border. The extent tested
      is the footprint, which for a crab is twice its shell — the legs are
      most of the mark and a crab with two legs off the paper is a mistake,
      not a composition.

    Crabs are also kept low, and they get a cloud rather than a station:
    they are bottom-dwellers and they are not swimming, so they are not in
    the shoal's frame at all. A crab in the top third of the sheet is a crab
    falling out of the sky; the 浮萍 near the upper border is what tells the
    viewer which way is up, so everything else has to obey it.
    """
    cx0 = rng.uniform(W * 0.34, W * 0.66)
    cy0 = rng.uniform(H * 0.34, H * 0.62)
    sig = S * (0.24 if len(want) > 3 else 0.18)

    # 势 — the shoal's own line, oblique to the current by 0.3–0.8 rad.
    axis = flow + rng.choice((-1.0, 1.0)) * rng.uniform(0.34, 0.76)
    ux, uy = math.cos(axis), math.sin(axis)
    vx, vy = -uy, ux
    nsh = sum(1 for kind, _ in want if kind == "shrimp")
    apart = 0 if nsh < 3 else (1 if nsh < 5 else 2)
    knot = nsh - apart
    # The shoal's centre is its own, and it sits *above* the crabs when
    # there are any: 虾在水中, 蟹在底 is what the module's docstring claims
    # and sharing one centre with the crabs was quietly denying it.
    hasc = any(kind == "crab" for kind, _ in want)
    sx0 = rng.uniform(W * 0.30, W * 0.70)
    sy0 = (rng.uniform(H * 0.18, H * 0.44) if hasc
           else rng.uniform(H * 0.30, H * 0.58))
    si = 0

    out, sites = [], []
    rev = nsh >= 3 and rng.random() < 0.22
    for i, (kind, size) in enumerate(want):
        # the footprint first, because the room a creature needs decides
        # where it may stand
        # 0.62 rather than 0.56 for a shrimp: the body is `size` long and
        # centred on the carapace, but the two claw arms reach another half
        # again *forward* of it, and at 0.56 the leading shrimp of a group
        # had its claws cut off by the left border. Whiskers are allowed off
        # the paper — they trail away and the eye finishes them — but a claw
        # stopped dead at the edge is 出界.
        # 1.35 for a crab, which is the reach of the *claw* and not of the
        # legs: base at 0.44 s, arm 0.39 s, hand 0.50 s and a finger on the
        # end of that. At 1.05 the near crab of 双蟹 had its big claw cut off
        # by the left border. The constructive fallback below is what makes
        # the larger figure affordable — under pure rejection sampling a
        # footprint this generous is what dropped the second crab.
        #
        # 1.20 was the figure until the claw root moved out onto the shell's
        # new front corner, and it was already short: measured from pixels
        # rather than from the arithmetic — one crab alone on a blank canvas,
        # worst of forty headings — the drawn mark reached 1.30 s then and
        # 1.35 s now. A footprint shorter than the mark makes the 出界 test
        # of §5.2 vacuous: it certifies the reserve, not the ink.
        r = size * (1.35 if kind == "crab" else 0.62)
        pad = S * 0.02 + r
        if kind == "crab":
            ylo, yhi = max(pad, H * low), H - pad
        else:
            ylo, yhi = pad, H - pad
        if ylo > yhi:
            ylo = yhi = (ylo + yhi) * 0.5
        if kind == "crab":
            # 蟹无定向 — a crab's shell may face anywhere, and the pair of
            # them facing the same way is the one arrangement to avoid, so
            # this heading is drawn free of the current the shrimp share.
            # "Free" is not enough on its own, though: two draws from the
            # whole circle land within a radian of each other about a third
            # of the time, and when they do the two crabs read as two prints
            # of one block. So the second one is *refused* that heading.
            # The one heading a crab may not take is *up the sheet*. 蟹无定向
            # is about the compass, not about gravity: the 浮萍 and the weed
            # establish which way is up, and a crab aimed at the top border
            # with both claws raised is not facing away, it is climbing out
            # of the water. Sideways and downward are both kept — 横行介士,
            # a crab walks across the paper — so this costs the pose nothing
            # but about a third of the circle.
            a = rng.uniform(0.0, TAU)
            for _ in range(24):
                if (math.sin(a) > -0.62
                        and all(abs(math.atan2(math.sin(a - oa),
                                               math.cos(a - oa))) > 0.9
                                for ok, _ox, _oy, _os, oa in out
                                if ok == "crab")):
                    break
                a = rng.uniform(0.0, TAU)
        elif rev and i == len(want) - 1:
            a = flow + math.pi + rng.gauss(0.0, 0.22)
        else:
            # 一水之势 — the fan is a quarter of what it was. `spread` is
            # kept as the scene's own dial so 双蟹 is still steadier than
            # 群虾, but it is now scaled into a range where the group still
            # reads as one body of shrimp in one current.
            a = _hold(flow + rng.gauss(0.0, spread * 0.42),
                      flow - 0.45, flow + 0.45)
        slot, best = si, None
        if kind == "shrimp":
            si += 1
        # The spacing requirement is *relaxed* as the tries run out rather
        # than held fixed, because held fixed it does not fail by crowding —
        # it fails by dropping the creature, silently, and 双蟹 came out with
        # one crab. Two crabs at 0.2 S each want 0.92 (r + sr) ≈ 0.45 S
        # between them inside a band of free paper barely that wide, so the
        # strict gap is unsatisfiable about half the time. A pair of crabs
        # with their legs interleaved is a composition (相叠); a pair with one
        # crab missing is not.
        #
        # 疏密 — and the requirement is *different for two shrimp* than for
        # anything involving a crab, which is the fault that nine sheets at
        # thumbnail size made obvious: a single gap rule applied to every
        # pair does not merely permit even spacing, it **enforces** it. Nine
        # compositions came out with every creature the same distance from
        # every other, which is 匀, and 匀 has no 聚 in it and therefore no
        # 散 either. A shoal is a knot of shrimp close enough to overlap and
        # then a long reach of empty water. So two shrimp may close to 0.58
        # of their footprints — bodies crossing, which real shrimp do and
        # 齐白石 drew constantly — while a crab, which is a planted mass
        # rather than a swimmer, still keeps its 0.92.
        #
        # 0.66 and not 0.58, which was tried first and went too far: at 0.58,
        # with the knot's own σ tightened at the same time, three shrimp
        # interpenetrated until their whiskers made one scribble and the
        # bodies could not be counted. 聚 means 近, not 混 — the creatures
        # touch and overlap, and each one is still a shrimp.
        gap0 = 0.66 if kind == "shrimp" else 0.92
        for try_i in range(110):
            gap = gap0 if try_i < 70 else gap0 - 0.22 * (try_i - 70) / 40.0
            if kind == "shrimp":
                # drawn in the shoal's frame: `t` along the 势 line, `n`
                # across it. As the tries run out the station is pulled in
                # toward the knot rather than re-rolled somewhere else — a
                # shrimp that will not fit where the line wants it belongs
                # nearer its own group, which is the direction the whole
                # arrangement is already pointing.
                q = 1.0 - 0.05 * min(try_i, 10)
                if slot < knot:
                    # 聚 — tight. σ along the line is barely more than one
                    # shrimp, so the knot is a *bunch*: bodies crossing,
                    # heads near one another, read as one shape at ten paces.
                    t = rng.gauss(0.0, S * 0.145) * q
                    nn = rng.gauss(0.0, S * 0.072) * q
                else:
                    # 散 — one behind the knot, the next one ahead of it, and
                    # both of them *far*: the whole point of a knot is the
                    # water on either side of it, and at 0.30 S the straggler
                    # was merely the next shrimp along.
                    sgn = -1.0 if (slot - knot) % 2 == 0 else 1.0
                    t = sgn * S * rng.uniform(0.40, 0.66) * q
                    nn = rng.gauss(0.0, S * 0.085) * q
                px = sx0 + ux * t + vx * nn
                py = sy0 + uy * t + vy * nn
            elif i == 0 and try_i == 0:
                px, py = cx0, cy0
            else:
                px = cx0 + rng.gauss(0.0, sig)
                py = cy0 + rng.gauss(0.0, sig * 0.84)
            if not (pad < px < W - pad and ylo < py < yhi):
                continue
            # clear of the neighbours by the mean of the two footprints:
            # a flat gap in sheet units lets the two biggest creatures
            # overlap while holding the small ones far apart. The 0.58 of
            # 聚 is between two *shrimp* only — anything a crab is party to
            # keeps the full gap, whichever of the two is being placed.
            if not all(math.hypot(px - sx, py - sy)
                       > (gap if sk == "shrimp" else
                          max(gap, 0.92 - 0.22 * max(0, try_i - 70) / 40.0))
                       * (r + sr)
                       for sx, sy, sr, sk in sites):
                continue
            # 忌压头 — crossed bodies are allowed (相叠, and two shrimp in a
            # shoal do swim over one another), but a *head* laid on another
            # creature's back is not. The head is the one dark mass on a
            # shrimp, so it is where the eye goes, and when it goes there and
            # finds another animal's body the two shrimp read as one wreck.
            # Tested on the head point rather than on the body circle, since
            # that is the only part of the overlap that matters.
            # 忌并列 — two crabs at the same height are a mirrored pair,
            # and the pair is what the eye sees instead of either crab. The
            # band they are confined to is wide and shallow, so "wherever
            # there is most room" is almost always *beside*: left and right
            # at one level, which is the single arrangement to avoid. One
            # must sit back from the other.
            if (kind == "crab" and try_i < 84
                    and any(sk == "crab" and abs(py - sy) < S * 0.11
                            for _sx, sy, _sr, sk in sites)):
                continue
            if kind == "shrimp" and try_i < 84:
                hx = px + math.cos(a) * size * 0.38
                hy = py + math.sin(a) * size * 0.38
                if any(math.hypot(hx - sx, hy - sy) < 0.52 * sr
                       for sx, sy, sr, _sk in sites):
                    continue
            best = (px, py)
            break
        # Rejection sampling cannot find a gap it is not aimed at. Two crabs
        # want ~0.4 S between them, drawn from a cloud of σ ≈ 0.18 S about one
        # centre and confined to the lower band of the sheet — so the only
        # acceptable samples are two sigma out in a direction that also has to
        # be legal, and 110 tries miss nearly every time. (Measured: the
        # second crab of 双蟹 was dropped in 10 of 12 consecutive seeds.) So
        # when sampling fails, *construct*: stand the creature off a
        # neighbour at the distance required, in some direction, and let only
        # the border move it.
        #
        # And then take the best of those directions rather than the first
        # acceptable one, because "acceptable" is a threshold the geometry
        # sometimes cannot meet at all. Two crabs at 0.2 S of shell want
        # 0.44 S between their centres inside a legal box about 0.46 S square
        # — only the diagonal of that box is wide enough, so clamping a
        # stand-off into it fails the threshold about one pair in six, and
        # failing it dropped the crab. (Measured: 63 of 800.) Crossed legs
        # are a composition — 相叠 — and a 双蟹 sheet with one crab on it is
        # not, so this stage has no threshold and never returns nothing.
        if best is None and sites:
            nx, ny, nr = max(sites, key=lambda t: t[2])[:3]
            ph0 = rng.uniform(0.0, TAU)
            for d in (0.92 * (r + nr), 0.74 * (r + nr)):
                for j in range(24):
                    a0 = ph0 + j * TAU / 24.0
                    px = min(W - pad, max(pad, nx + math.cos(a0) * d))
                    py = min(yhi, max(ylo, ny + math.sin(a0) * d))
                    room = min(math.hypot(px - sx, py - sy) / (r + sr)
                               for sx, sy, sr, _sk in sites)
                    # ...and a thumb on the scale for standing off *up or
                    # down* rather than sideways, which is the same 忌并列
                    # as above: the legal band is wide and shallow, so room
                    # alone always answers "beside", and this stage is where
                    # the crowded pairs actually get placed.
                    room += 0.16 * abs(math.sin(a0))
                    if best is None or room > best[2]:
                        best = (px, py, room)
                if best is not None and best[2] > 0.80:
                    break
            best = (best[0], best[1])
        if best is None:
            # nothing placed yet and 110 samples all fell outside the legal
            # band — which can happen when a crab's band has collapsed to a
            # line. The middle of the band is always legal.
            best = (W * 0.5, (ylo + yhi) * 0.5)
        sites.append((best[0], best[1], r, kind))
        out.append((kind, best[0], best[1], size, a))

    # 回首 — the one creature turned back against the current is also the
    # one case where 忌对头 can actually happen, so it is the one case
    # checked: swing it aside rather than reversing it again.
    if rev and len(out) > 1 and out[-1][0] == "shrimp":
        kind, hx, hy, hs, ha = out[-1]
        for _k, ox, oy, _s, oa in out[:-1]:
            to = math.atan2(oy - hy, ox - hx)
            if (abs(math.atan2(math.sin(to - ha), math.cos(to - ha))) < 0.55
                    and math.hypot(ox - hx, oy - hy) < S * 0.40):
                ha += rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.1)
                out[-1] = (kind, hx, hy, hs, ha)
                break

    # 置阵 — and then the whole group is slid onto the sheet as one piece.
    #
    # Every creature was placed relative to a centre drawn before any of them
    # existed, and an elongated group hung off an unlucky centre puts six
    # shrimp in the top third with the bottom half of the paper empty. That
    # is not 留白, which is reserved space; it is a picture that missed.
    #
    # Translating is the right repair rather than re-drawing the centre
    # because the arrangement *inside* the group — the 势 line, the stagger,
    # the knot and its stragglers — is exactly what was wanted. Only its
    # position on the paper was wrong, and position is one subtraction.
    #
    # The shift is clamped by two things, in this order: no footprint may
    # leave the paper (忌出界 holds over the move), and no crab may rise out
    # of its band (a crab is not swimming).
    #
    # 取势 — the target is deliberately *off* centre. Aiming the group's box
    # at the middle of the paper is the mistake that reads worst at thumbnail
    # size: the mass lands dead centre and all four margins are equally
    # empty, which is four small voids instead of one big one. So one of the
    # four quarters is drawn and the group is pushed that way; the empty
    # paper then collects on the opposite diagonal, where it is a 留白 the
    # eye can cross. Only 0.80 of the wanted shift is taken, so the lean
    # stays a lean and does not become a corner.
    if out:
        mar = S * 0.02
        x0 = min(x - r for (x, _y, r, _k) in sites)
        x1 = max(x + r for (x, _y, r, _k) in sites)
        y0 = min(y - r for (_x, y, r, _k) in sites)
        y1 = max(y + r for (_x, y, r, _k) in sites)
        qx = rng.choice((-1.0, 1.0))
        # Down is the better lean, about two times in three. These are tall
        # sheets, the 题款 always takes a top corner and the 水草 always come
        # up off the bottom border, so a group leaning *up* puts the ink and
        # the writing in the same half and leaves the lower half of the
        # paper empty of everything — void below and clutter above is not
        # 留白, it is a picture that slid off the top of the mount.
        qy = -1.0 if rng.random() < 0.34 else 1.0
        tx = W * (0.50 + qx * rng.uniform(0.09, 0.15))
        ty = H * (0.50 + qy * rng.uniform(0.07, 0.12))
        dx = _hold2(tx - (x0 + x1) * 0.5, mar - x0, W - mar - x1)
        dlo, dhi = mar - y0, H - mar - y1
        for (kind, _x, y, _s, _a), (_sx, _sy, r, _k) in zip(out, sites):
            if kind == "crab":
                dlo = max(dlo, max(mar + r, H * low) - y)
        dy = _hold2(ty - (y0 + y1) * 0.5, dlo, dhi)
        dx, dy = dx * 0.80, dy * 0.80
        out = [(kind, x + dx, y + dy, s, a) for kind, x, y, s, a in out]
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
    pal = dict(weed=st["weed"])          # all fish.py's furniture needs

    if air.get("veil"):
        veil(cv, air["veil"][0], air["veil"][1])

    # 水势 — one current for the whole sheet, roughly horizontal, because
    # 虾 swim across a painting and not up and down it.
    flow = rng.choice((0.0, math.pi)) + rng.gauss(0.0, 0.30)

    # --- 水草 — behind everything, rooted off the bottom border ---------
    if rng.random() < sc["weed"]:
        # the scene's `weed` sets how many clumps as well as whether there
        # are any: 芦间 means "down among the weed" and one sheaf of five
        # blades in a corner is not among anything
        for _ in range(rng.randint(1, 2 + int(sc["weed"] * 3))):
            water_weed(cv, rng, pal, rng.uniform(W * 0.05, W * 0.95),
                       H + S * 0.02, S * rng.uniform(0.24, 0.56),
                       rng.randint(3, 7), rng.uniform(0.26, 0.50) * k,
                       rng.gauss(0.0, 0.14) + air["flow"] * 0.2)

    # --- the creatures, 大小相间 ---------------------------------------
    #
    # 大小相间 is about *ratio*. The sizes are drawn across the scene's
    # whole range and sorted big to small, so there is one unmistakable 主
    # and the rest are subordinate; six creatures within a tenth of each
    # other read as a specimen chart however well each one is painted.
    ns = rng.randint(*sc["shrimp"])
    nc = rng.randint(*sc["crab"])
    want = []
    for j in range(ns):
        # The shrimp need the same treatment as the crabs and for the same
        # reason: the scene's range spans about a third, and five draws from
        # it land in a heap. One shrimp is painted at the scene's size and
        # the others are pulled down from theirs, so the sheet has a 主虾 —
        # the one the eye lands on — and a 从 that falls away behind it.
        want.append(("shrimp", S * rng.uniform(*sc["ss"])
                     * (1.0 if j == 0 else rng.uniform(0.62, 0.90))))
    for j in range(nc):
        # 大小相间 applies hardest to the crabs, because there are only ever
        # one or two of them and two of a size is a pair rather than a
        # 主从. The scene's own range is narrow (0.18–0.23 S) and two draws
        # from it are within a tenth of each other half the time, so the
        # second crab is shrunk outright instead of being drawn again.
        want.append(("crab", S * rng.uniform(*sc["cs"])
                     * (1.0 if j == 0 else rng.uniform(0.70, 0.84))))
    # the crabs get the room first (they need most of it), then the shrimp
    # from biggest to smallest
    want.sort(key=lambda it: (-(it[0] == "crab"), -it[1]))
    placed = creature_sites(rng, W, H, S, want, flow, sc["spread"], sc["low"])

    # 远淡近浓 — depth by size, so the small ones are also the pale ones.
    # Small and dark reads as a small creature; small and pale reads as a
    # creature further off, which is the only depth available on a sheet
    # with no water painted in it.
    sizes = sorted((p[3] for p in placed), reverse=True)
    rank = {v: i for i, v in enumerate(sizes)}
    n = max(1, len(placed) - 1)
    order = []
    for kind, px, py, size, a in placed:
        # The ramp is convex rather than straight. Spread evenly, every
        # creature is a slightly different grey and the sheet has no 浓淡 —
        # just a gradient. Bent this way the big ones stay near full ink and
        # the pallor is spent on the last one or two, which is 墨分五色 used
        # as 主从: a dark anchor, a dark second, and something nearly gone.
        d = (rank[size] / n) ** 1.35 * rng.uniform(0.7, 1.0)
        # A crab is never let all the way back, however small it came out.
        # Depth here is ranked on size alone, and a crab is measured by its
        # shell while a shrimp is measured by its whole length — so on a 虾蟹
        # sheet the crab is routinely the "smallest" creature and was coming
        # out the palest thing on the paper. A pale crab is not a far crab,
        # it is a dead one: 蟹 is the planted, heavy, bottom-sitting mark
        # that the shrimp are swimming above, and it has to keep its ink.
        if kind == "crab":
            d = min(d, 0.40)
        order.append((d, kind, px, py, size, a))

    # 远 is worth one creature, not three. Size 主从 and the ink ramp are two
    # ways of saying "further off", and multiplied together they erase: a
    # shrimp at 0.62 of the scene's size *and* at 0.9 depth is a smudge, and
    # a 群虾 sheet was coming out with two of its four shrimp as smudges. So
    # the ramp is held back for everybody except the single furthest, which
    # keeps its pallor and becomes the one creature dissolving into the
    # water — 一笔虚, 余皆实.
    if len(order) > 2:
        far = max(range(len(order)), key=lambda i: order[i][0])
        for i, o in enumerate(order):
            cap = 0.66 if i == far else 0.46
            if o[0] > cap:
                order[i] = (cap,) + o[1:]
    for depth, kind, px, py, size, a in sorted(order, key=lambda o: -o[0]):
        if kind == "crab":
            r = crab(cv, rng, px, py, size, a, st, air, depth=depth, k=k)
        else:
            r = shrimp(cv, rng, px, py, size, a, st, air, depth=depth, k=k)
            if depth < 0.45 and rng.random() < sc["bub"]:
                mx, my = px + math.cos(a) * size * 0.34, \
                    py + math.sin(a) * size * 0.34
                paint_bubbles(cv, rng, mx, my, size * 0.20, a,
                              rng.randint(2, 4))
        occ.add(px, py, r * 0.85)
        if stat is not None:
            stat.setdefault(kind, 0)
            stat[kind] += 1

    # --- 浮萍 on the surface, in front of everything -------------------
    if rng.random() < sc["duck"]:
        duckweed(cv, rng, pal, W, H, rng.randint(4, 12), occ=occ)

    # counted off the sheet, not off the scene's wishes: siting can refuse a
    # creature, and the title has to describe the painting that exists
    got_s = sum(1 for p in placed if p[0] == "shrimp")
    got_c = sum(1 for p in placed if p[0] == "crab")
    head = _subject(rng, got_s, got_c) if rng.random() < 0.76 else air["cn"]
    # 虾趣清趣 — no stutter. Both halves are drawn from word lists and three
    # of the subjects end in 趣, which 清趣 then says again.
    tails = [w for w in TITLE_B if not any(c in head for c in w)]
    title = head + tails[rng.randrange(len(tails))]
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

def paint(path, size=(900, 1200), seed=None, scene=None, air=None,
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
        description="Generate random Chinese ink paintings of 虾蟹.")
    ap.add_argument("-o", "--out", default="crab.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="虾蟹|群虾|双蟹|芦间")
    ap.add_argument("--air", help="静水|水动|秋深|雨后")
    ap.add_argument("--style", help="水墨|赭墨|青墨|淡彩")
    ap.add_argument("--size", default="900x1200", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for key, v in SCENES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("风致 (water):")
        for key, v in AIR.items():
            print("  %-6s %-5s %s" % (key, v["cn"], v["en"]))
        print("设色 (colour):")
        for key, v in STYLES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("诀 (the rules): 壳用三笔, 八足二螯, 足分三节, 螯有大小, "
              "虾身透明, 虾贵在须 — 不画水")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1200")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("虾蟹 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
