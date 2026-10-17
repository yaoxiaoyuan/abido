#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fish.py — 随机生成中国画游鱼图 / Random Chinese fish paintings (游鱼).

The fourth painter in this set, and the first to attempt a living animal.
Its subject is 游鱼 — swimming fish — which in 国画 is governed by one rule
that overrides every other:

    **不画水.**  The water is not painted.

A Chinese fish painting is a fish in bare paper. There is no blue, no surface,
no bottom, no horizon; the 留白 *is* the water, and it only becomes water
because a fish is swimming in it. Everything else this module does follows from
that, because once you may not paint the medium, the fish has to carry the
whole illusion by itself:

    得势   the body is an S-curve. A straight fish is a dead fish, so the spine
           is a wave whose amplitude grows toward the tail — 摆尾
    分墨   背浓腹淡: ink loaded along the back, almost dry at the belly, which
           is what makes a flat silhouette read as a round body
    虚实   fins and tail are 淡墨 washes with no contour at all, so they read as
           translucent membrane; the body has weight, the fins have none
    点睛   the eye goes in last and off-centre. 八大山人's fish are famous for
           exactly one mark, and it is this one
    疏密   fish are grouped 大小相间 with a shared current, and the sheet is
           left mostly empty. A fish in every corner is an aquarium
    知鱼   the inscription. 濠梁观鱼 — Zhuangzi on the bridge over the Hao,
           arguing that he knows the fish are happy

The primitive each painter in this set had to find is different every time. In
gongbi.py it was `Blade`, the pressed brush stroke. In shanshui.py it was a
height field, because a mountain is not a stroke. In bamboo.py it was `Blade`
again, because a 竹叶 *is* one. Here it is `Spindle`: a body with a free-form
curved centreline, a width profile given as control points, and its shading
signed *across* the spine instead of inward from the rim. A fin, though, is a
`Blade` — and a very natural one, because `Blade`'s vein fan is 鳍条, fin rays.

Infrastructure (PNG writer, value noise, paper ground, the signed-field blade,
brush strokes, baked calligraphy outlines, seals, the occupancy map) is shared
with gongbi.py rather than copied — this module imports it.

Usage
-----
    python3 fish.py                             # one random painting
    python3 fish.py -n 6                        # six of them
    python3 fish.py --scene 群游 --species 金鱼
    python3 fish.py --palette 水墨 -s 7
    python3 fish.py --size 1200x800 -o scroll.png
    python3 fish.py --list                      # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Blade, Canvas, Noise, Occupancy, BrushFont, P, bell,
                    brush_path, brush_seg, clamp01, disc, hsv_mix, mix, rgb,
                    saturate, paint_column, paint_seal,
                    NU, NT, NS, SH0, SHK, TIANGAN, DIZHI)

TAU = math.pi * 2.0
NG = 64                  # 擦 — bristle-streak profile, a power of two


# --------------------------------------------------------------------------
# 墨色 — ink tones, and the four 设色 schools a fish painter may use
# --------------------------------------------------------------------------

INK = {
    "jiao":  "#1B1917",   # 焦墨  burnt — the pupil, and nothing else
    "nong":  "#2B2825",   # 浓墨  thick — the back, the gill, the mouth
    "zhong": "#4C453B",   # 重墨  heavy
    "dan":   "#837B6D",   # 淡墨  diluted — fins
    "qing":  "#AFA99B",   # 清墨  clear — the far fish, water weed
}


def I(name):
    return rgb(INK[name])


# Each 设色 gives the two stops of every ramp in the painting: the pale stop
# (which for a fish is the belly, and is near white because the body is a wash
# that multiplies into the paper, so the paper supplies the warmth) and the
# loaded stop (the back). 水墨 is the orthodox form — Bada Shanren and
# Qi Baishi both painted fish in nothing but ink. 朱砂 is the 金鱼 painter's
# palette, vermilion over white. 赭金 is the 鲤鱼's, ochre going to gold.
# 墨彩 is ink with one warm pigment allowed in, which is how most Qing album
# leaves of fish actually look.
# Note how pale the pale stops are. The bodies multiply into the paper, so the
# paper supplies all the warmth there is, and a pale stop with any colour of
# its own gets that colour multiplied *again* — two coats of cream. 水墨 drawn
# from a warm pale stop came out a tan fish: not wrong exactly, but no longer
# ink. The pigment palettes keep their hue, because 朱砂 on silk really is
# that colour and it is sitting on top rather than soaking in.
PALETTES = {
    "水墨": dict(cn="水墨", en="pure ink",
                 body=("#F0EEEA", "#2E2B27"), fin=("#EEECE7", "#6A645A"),
                 ink="#241F1C", weed=("#DCD9CF", "#6A6659"),
                 accent="#4C453B", lotus=("#DEDCD2", "#7C776A"),
                 piebald=False),
    "朱砂": dict(cn="朱砂", en="vermilion goldfish",
                 body=("#F6E6D4", "#A61E14"), fin=("#F2DCCB", "#C4583C"),
                 ink="#3A1A14", weed=("#C3C7A4", "#5E6B3C"),
                 accent="#D24A22", lotus=("#C8CDAC", "#6B7442"),
                 piebald=True),
    "赭金": dict(cn="赭金", en="ochre carp",
                 body=("#F2E3C4", "#8A5120"), fin=("#EEDCBE", "#B07C44"),
                 ink="#33251A", weed=("#C6C8A2", "#5F6B3E"),
                 accent="#C98A2E", lotus=("#CACDA8", "#6E7644"),
                 piebald=True),
    "墨彩": dict(cn="墨彩", en="ink with one colour",
                 body=("#EFECE4", "#3A332B"), fin=("#EDEAE1", "#756C5E"),
                 ink="#241F1C", weed=("#D2D5B8", "#5C6840"),
                 accent="#A4612C", lotus=("#C9CCAB", "#6C7544"),
                 piebald=False),
}


# --------------------------------------------------------------------------
# 鱼 — the repertoire.
#
# A species is almost entirely its *outline*, so that is what the table
# stores: `prof` is the body's half-width as a fraction of the maximum, at
# control points along the spine from snout (u=0) to tail root (u=1), and
# `wid` is the total depth of the fish as a fraction of its length. Read the
# numbers and you can see the fish — 金鱼 swells to full depth by a third of
# the way back and is more than half as deep as it is long; 鲤鱼 is a torpedo
# at 0.30 whose widest point is nearly at the middle.
#
# `wid` is deliberately the *full* depth and not the half-width that `Spindle`
# actually wants, even though that means a factor of two at the call site.
# Written as half-widths these numbers are unreadable, and the first draft of
# this table was wrong for every species at once because I fed depth ratios
# into a half-width slot: every fish came out twice as deep as it should be,
# which does not look like a fat fish, it looks like a turtle.
#
# Note every profile ends non-zero. The spine stops at the 尾柄, the caudal
# peduncle, which is a wrist and not a point — the tail fin is a separate,
# translucent thing hung off it. Running the body to zero is the single
# commonest way to draw a fish that looks like a leaf.
#
# Fins: `len` is a fraction of body length, `wid` the fin's own half-width as
# a fraction of its own length, and `sweep` how far it is raked back from the
# body's normal (0 = straight out, 1 = flat along the flank). A long-based
# fin — a carp's dorsal — is a high `sweep`, which lays the stroke down along
# the back instead of standing it up like a sail.
#
# Every `sweep` here is high, between 0.6 and 0.8, and that is the correction
# for the longest-running fault in the module. At 0.4–0.6 the fins stood out
# radially from the body like the pads of a cactus, and a fish bristling with
# eight short paddles reads as a toy. A swimming fish's fins *trail*: water is
# pushing them back along the flank, and the mark that says motion is a fin
# raked nearly flat. They are narrow for the same reason — a trailing veil,
# not an oar. The 背鳍 is the exception: a dorsal fin is held *up*, it is the
# fin that is not in the water's way, so its sweep stays near 0.5. Raked flat
# with the others it lay along the back and merged into the tail, and the fish
# came out with nothing on top of it at all.
# --------------------------------------------------------------------------

SPECIES = {
    "goldfish": dict(
        cn="金鱼", en="goldfish", season="夏",
        # Nearly as deep as it is long, and blunt at *both* ends. Tapering
        # from 1.00 at u=0.4 down to 0.18 made a teardrop, and a teardrop is
        # a tadpole — a 金鱼 is an egg with a fan on the back of it.
        prof=[(0.00, 0.15), (0.10, 0.62), (0.20, 0.89), (0.30, 0.99),
              (0.40, 1.00), (0.50, 0.98), (0.60, 0.93), (0.70, 0.84),
              (0.80, 0.70), (0.90, 0.50), (1.00, 0.30)],
        # The fin lengths below are fractions of the *body length*, and the
        # first set were all roughly double what they are now: a dorsal fin
        # longer than the body was deep, a pectoral two thirds of that. Every
        # fish came out trailing three pale plumes, and because they were pale,
        # soft and bigger than the body, they read not as fins but as the
        # fish's shadow. A fin scales with the body's *depth*, not its length.
        wid=0.64, swing=(0.05, 0.13), freq=(2.0, 3.0),
        head=0.24, rim=0.18, line=0.0,
        eye=0.085, eye_u=0.090, eye_v=0.30, bulge=1.12,
        # 金鱼's tail is the whole point of a goldfish: a fan of long frilled
        # strokes, longer than the body itself, and so thin that whatever is
        # behind it shows through.
        tail=dict(n=6, len=0.74, wid=0.15, splay=0.30, frill=0.06,
                  soft=2.2, alpha=(0.30, 0.46), rays=0.10),
        # 金鱼's are veils and are allowed to be long — a third of the body
        # for the paired fins, where 鲤鱼's are a sixth. This is most of what
        # separates the two fish once the outline is right.
        # the dorsal is the one fin a 金鱼 does not get extra length: three
        # long strokes standing up off the back read as antlers
        dorsal=dict(u=0.30, len=0.25, wid=0.38, sweep=0.58, frill=0.10),
        pect=dict(u=0.23, len=0.27, wid=0.40, sweep=0.74, frill=0.08),
        vent=dict(u=0.58, len=0.20, wid=0.38, sweep=0.80, frill=0.10),
        anal=dict(u=0.80, len=0.27, wid=0.38, sweep=0.72, frill=0.10),
        scales=0.0, barbel=0.0, mottle=0.0, spine=0.0, patch=1.0,
    ),
    "carp": dict(
        cn="鲤鱼", en="carp", season="春",
        prof=[(0.00, 0.14), (0.10, 0.56), (0.20, 0.82), (0.30, 0.95),
              (0.40, 1.00), (0.50, 0.98), (0.60, 0.90), (0.70, 0.78),
              (0.80, 0.62), (0.90, 0.42), (1.00, 0.20)],
        wid=0.30, swing=(0.07, 0.17), freq=(2.2, 3.4),
        head=0.28, rim=0.20, line=0.24,
        eye=0.060, eye_u=0.085, eye_v=0.30, bulge=1.05,
        tail=dict(n=6, len=0.30, wid=0.17, splay=0.34, frill=0.04,
                  soft=1.3, alpha=(0.46, 0.66), rays=0.10),
        dorsal=dict(u=0.34, len=0.17, wid=0.32, sweep=0.60, frill=0.05),
        pect=dict(u=0.20, len=0.17, wid=0.34, sweep=0.74, frill=0.04),
        vent=dict(u=0.54, len=0.11, wid=0.34, sweep=0.82, frill=0.04),
        anal=dict(u=0.80, len=0.13, wid=0.34, sweep=0.74, frill=0.04),
        scales=0.9, barbel=1.0, mottle=0.0, spine=0.0, patch=0.0,
    ),
    "mandarin": dict(
        cn="鳜鱼", en="mandarin fish", season="秋",
        prof=[(0.00, 0.18), (0.10, 0.60), (0.20, 0.86), (0.30, 0.98),
              (0.40, 1.00), (0.50, 0.94), (0.60, 0.84), (0.70, 0.70),
              (0.80, 0.54), (0.90, 0.34), (1.00, 0.16)],
        wid=0.34, swing=(0.06, 0.15), freq=(2.0, 3.2),
        head=0.38, rim=0.22, line=0.12,
        eye=0.068, eye_u=0.100, eye_v=0.32, bulge=1.12,
        tail=dict(n=5, len=0.28, wid=0.18, splay=0.28, frill=0.05,
                  soft=1.5, alpha=(0.44, 0.64), rays=0.10),
        # 鳜鱼's dorsal is a row of spines. 齐白石 drew it as a saw, and it is
        # the mark that names the fish.
        dorsal=dict(u=0.32, len=0.20, wid=0.44, sweep=0.50, frill=0.0),
        pect=dict(u=0.22, len=0.17, wid=0.34, sweep=0.74, frill=0.04),
        vent=dict(u=0.52, len=0.11, wid=0.34, sweep=0.82, frill=0.04),
        anal=dict(u=0.78, len=0.13, wid=0.34, sweep=0.72, frill=0.04),
        scales=0.0, barbel=0.0, mottle=1.0, spine=1.0, patch=0.0,
    ),
    "crucian": dict(
        cn="鲫鱼", en="crucian carp", season="冬",
        prof=[(0.00, 0.13), (0.10, 0.54), (0.20, 0.84), (0.30, 0.97),
              (0.40, 1.00), (0.50, 0.97), (0.60, 0.90), (0.70, 0.76),
              (0.80, 0.58), (0.90, 0.38), (1.00, 0.16)],
        wid=0.40, swing=(0.06, 0.16), freq=(2.1, 3.3),
        head=0.26, rim=0.19, line=0.0,
        eye=0.064, eye_u=0.085, eye_v=0.30, bulge=1.05,
        tail=dict(n=5, len=0.30, wid=0.17, splay=0.30, frill=0.05,
                  soft=1.6, alpha=(0.42, 0.60), rays=0.10),
        dorsal=dict(u=0.33, len=0.17, wid=0.34, sweep=0.56, frill=0.05),
        pect=dict(u=0.21, len=0.17, wid=0.34, sweep=0.74, frill=0.05),
        vent=dict(u=0.54, len=0.11, wid=0.34, sweep=0.82, frill=0.05),
        anal=dict(u=0.79, len=0.13, wid=0.34, sweep=0.74, frill=0.05),
        scales=0.4, barbel=0.0, mottle=0.0, spine=0.0, patch=0.0,
    ),
}


# 章法 — what is on the sheet besides the fish. The counts are deliberate:
# 独游 is one fish in an entire sheet of nothing, which is the hardest and the
# most Chinese of the five.
SCENES = {
    "群游": dict(cn="群游", en="a school, 大小相间",
                 n=(5, 8), size=(0.42, 1.00), weed=0.55, duck=0.45,
                 lotus=0.15, spread=0.42, back=0.30),
    "双鱼": dict(cn="双鱼", en="a pair, 呼应",
                 n=(2, 2), size=(0.62, 1.00), weed=0.50, duck=0.40,
                 lotus=0.20, spread=0.30, back=0.45),
    "独游": dict(cn="独游", en="one fish, 留白",
                 n=(1, 1), size=(1.00, 1.00), weed=0.30, duck=0.30,
                 lotus=0.10, spread=0.0, back=0.0),
    "荷塘": dict(cn="荷塘", en="under lotus",
                 n=(3, 6), size=(0.46, 1.00), weed=0.35, duck=0.60,
                 lotus=1.00, spread=0.40, back=0.30),
    "藻荇": dict(cn="藻荇", en="among water weed",
                 n=(3, 6), size=(0.48, 1.00), weed=1.00, duck=0.35,
                 lotus=0.0, spread=0.44, back=0.30),
}


# --------------------------------------------------------------------------
# 身 — the body: a field around a curved spine
# --------------------------------------------------------------------------

def _tabulate(ctrl, n):
    """Resample control points [(u, v), ...] onto n+1 even samples, with a
    Catmull-Rom spline through them.

    The first version of this smoothstepped between neighbours instead, on the
    reasoning that a spline can overshoot and put a bulge on the fish's back
    while smoothstep provably cannot. That was true and beside the point.
    Smoothstep is flat at *every* control point, so a profile given as eleven
    dense samples came out visibly scalloped — one ripple per control point
    along the whole dorsal silhouette, which on a 450-pixel fish looks like a
    row of blisters. Catmull-Rom is only C1, but it is C1, and with samples
    this dense and this smooth there is no overshoot to speak of.

    The lesson generalises: the guarantee an interpolant gives you is not
    automatically the property you need. Monotonicity was never the problem.
    Curvature continuity was.
    """
    out = [0.0] * (n + 1)
    m = len(ctrl)
    j = 0
    for i in range(n + 1):
        u = i / n
        while j < m - 2 and ctrl[j + 1][0] < u:
            j += 1
        u0, v0 = ctrl[j]
        u1, v1 = ctrl[j + 1]
        t = 0.0 if u1 <= u0 else (u - u0) / (u1 - u0)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        # the two outer points, clamped at the ends
        v_a = ctrl[j - 1][1] if j > 0 else v0
        v_b = ctrl[j + 2][1] if j + 2 < m else v1
        t2 = t * t
        t3 = t2 * t
        out[i] = (v0
                  + 0.5 * (v1 - v_a) * t
                  + 0.5 * (2.0 * v_a - 5.0 * v0 + 4.0 * v1 - v_b) * t2
                  + 0.5 * (-v_a + 3.0 * v0 - 3.0 * v1 + v_b) * t3)
    return out


class Spindle:
    """A fish body: a signed field whose centreline is a curve.

    The frame is `Blade`'s — `u` runs 0→1 along the axis from the snout, `v`
    is the signed offset across it — but three things differ, and each of them
    is a fish rather than a petal:

    * **the centreline is free.** `Blade` offsets its axis by `bend·L·u²`, one
      parabola, which is a bent leaf. A swimming fish is a travelling wave
      whose amplitude grows toward the tail, so the offset is tabulated from
      `swing`, `phase` and `freq` and can go both ways. Like `Blade` this is
      a *shear* — cross-sections stay parallel to the local y axis instead of
      standing perpendicular to the curve. At the curvature a fish actually
      has, nobody can see the difference, and it keeps the renderer O(area)
      with no stitching between segments.
    * **the width is a profile, not a formula.** Four exponents cannot
      describe a snout, a shoulder, a long taper and a wrist at the same time,
      so `prof` is control points and the species table is readable as a
      drawing.
    * **the shading is signed across the spine, not inward from the rim.**
      This is the whole reason the class exists. A petal is shaded from its
      edge inward because that is where a wash pools. A fish is shaded
      背浓腹淡 — loaded along the top edge, nearly dry along the bottom — and
      that gradient across the body is the only thing making a flat shape
      read as a round one. Get it inward-from-the-rim and you have painted a
      fish-shaped leaf.
    """

    __slots__ = ("cx", "cy", "ang", "L", "W", "dorsal",
                 "light", "deep", "ink", "line", "alpha",
                 "w_back", "p_back", "w_head", "p_head", "w_rim", "p_rim",
                 "hl", "p_hl", "lit", "soft", "blend", "bow",
                 "vfade", "p_vfade", "vlo", "vhi", "vfe", "dry", "dfreq",
                 "ulo", "uhi", "ufe", "samp", "sfreq", "resv",
                 "eamp", "efreq", "ephase", "mamp", "mfreq", "bamp",
                 "hw_t", "of_t", "hwmax", "ofmax")

    def __init__(self, cx, cy, ang, L, W, prof, swing=0.0, phase=0.0,
                 freq=2.4, **kw):
        """`(cx, cy)` is the snout and `ang` is the **axis** — the direction
        from head to tail, which is the opposite of the way the fish faces.
        `paint_fish` takes the facing angle and flips it here, because facing
        is what a composition wants to talk about and the axis is what the
        field needs."""
        d = dict(
            light=(224, 216, 198), deep=(52, 48, 42), ink=I("nong"),
            line=0.0, alpha=1.0,
            w_back=0.96, p_back=1.65,     # 背浓 — ink along the top edge
            w_head=0.28, p_head=3.2,      # 头重 — the skull takes more ink
            w_rim=0.20, p_rim=3.4,        # the edge where a wet stroke pools
            hl=0.34, p_hl=2.4,            # 腹浅 — belly lifted toward white
            vfade=0.0, p_vfade=3.4,       # 腹虚 — belly edge dissolved
            # 一笔 — which band of the flank this pass is. A body is not one
            # shape, it is two or three strokes of a loaded brush laid side by
            # side, and `(vlo, vhi, vfe)` is how one of them is cut out of the
            # envelope: a smoothstep window over the *signed* flank, feathered
            # by `vfe`, multiplied into coverage. The default window is wider
            # than the body, so a Spindle with no window set is the whole fish.
            vlo=-1.2, vhi=1.2, vfe=0.30,
            # ...and where along the body it starts and stops. Same window,
            # same feather, over `u` instead of `v`. A stroke that runs the
            # full length of the envelope has to be the *whole* contour; one
            # that stops short leaves the rest of the silhouette to the paper.
            ulo=-0.1, uhi=1.1, ufe=0.16,
            dry=0.0, dfreq=7.0,           # 飞白 — see render()
            samp=0.0, sfreq=2.6,          # 擦 — see render()
            resv=(),                      # 留白 — see render()
            lit=0.0, soft=0.0, blend="over", bow=0.07,
            eamp=0.045, efreq=3.4, ephase=0.0,   # 笔痕 — see below
            mamp=0.30, mfreq=2.1,                # 墨色不匀 — see render()
            bamp=0.0,                            # 墨渍 — see render()
            dorsal=None,
        )
        d.update(kw)
        self.cx, self.cy, self.ang, self.L, self.W = cx, cy, ang, L, W
        for k, v in d.items():
            setattr(self, k, v)

        # Which local flank is the fish's back? Local +v maps to the world
        # direction (-sin a, cos a), so its screen-y component is cos a: when
        # that is positive, +v points *down* the page and the back is -v.
        # Without this a fish swimming leftward is painted upside down — the
        # ink ends up along its belly — and because the silhouette is almost
        # symmetric, it reads not as a mistake but as a fish that is somehow
        # wrong.
        if self.dorsal is None:
            self.dorsal = -1.0 if math.cos(ang) > 0.0 else 1.0

        # 笔痕 — a slow wobble multiplied onto the width profile.
        #
        # Without it the silhouette is a mathematically perfect envelope, and
        # that one quality was enough to make every fish read as airbrushed
        # plastic however good the ink gradient inside it was. No brush can
        # draw a curve that clean; a body stroke swells where the hair is
        # loaded and pinches where it lifts. Two or three cycles over the
        # whole length at four per cent is below the threshold of looking
        # *wavy* and well above the threshold of looking *drawn*.
        # Three scales of it, and the fast ones are what matter. One slow swell
        # over the whole length changes the *proportions* of the envelope but
        # leaves it as smooth a curve as it was before, and a long smooth arc
        # from the snout over the back to the wrist was the last thing on these
        # fish still reading as vector art. The two fast terms are at a tenth
        # and a twentieth of the body length, which is the scale at which a
        # brush actually rides up over the paper's tooth and drops back.
        # The two fast terms are counted in cycles over the whole stroke, and
        # that has to be bounded in *pixels*, which is the one thing the frame
        # does not know about. 9.7 × `efreq` cycles is a brush riding the
        # paper's tooth on a 600-pixel body and a sawtooth on a 90-pixel fin —
        # and once the fins became strokes too, every one of them came out as a
        # feather. Nothing here gets a wavelength shorter than about twelve
        # pixels, which is roughly the finest thing a brush can leave.
        pt = _tabulate(prof, NU)
        ea, ef, ep = self.eamp, self.efreq, self.ephase
        cmax = max(1.0, L / 12.0)
        ef = min(ef, cmax * 0.30)
        f2 = min(ef * 4.6, cmax * 0.55)
        f3 = min(ef * 9.7, cmax)
        hw = []
        for i, v in enumerate(pt):
            u = i / NU
            w = (math.sin(ep + ef * TAU * u
                          + 0.7 * math.sin(ep * 1.7 + 2.1 * u))
                 + 0.30 * math.sin(ep * 3.3 + f2 * TAU * u)
                 + 0.18 * math.sin(ep * 5.1 + f3 * TAU * u + 1.3))
            hw.append(W * v * (1.0 + ea * w))
        self.hw_t = hw
        self.hwmax = max(self.hw_t)
        # 摆尾 — the swimming wave. Zeroed at the snout so (cx, cy) is the
        # nose, and growing toward the tail, because that is which end of a
        # fish does the moving.
        s0 = math.sin(phase)
        of = [0.0] * (NU + 1)
        for i in range(NU + 1):
            u = i / NU
            of[i] = (swing * L * (math.sin(phase + u * freq) - s0)
                     * (0.18 + 0.82 * u))
        self.of_t = of
        self.ofmax = max(abs(v) for v in of)

    # ---- geometry, for hanging fins and eyes on ------------------------
    def hw_at(self, u):
        return self.hw_t[int(clamp01(u) * NU)]

    def of_at(self, u):
        return self.of_t[int(clamp01(u) * NU)]

    def pos(self, u, side=0.0):
        """World point at `u`, `side` of the local half-width off the spine.
        `side` is in dorsal-signed units: +1 is the back edge, -1 the belly."""
        u = clamp01(u)
        x = u * self.L
        y = self.of_at(u) + side * self.dorsal * self.hw_at(u)
        ca, sa = math.cos(self.ang), math.sin(self.ang)
        return self.cx + x * ca - y * sa, self.cy + x * sa + y * ca

    def axis(self, u):
        """World angle of the centreline tangent at `u`, pointing **toward the
        tail** — `u` grows from the snout, so this is downstream along the
        body, *not* the way the fish is looking.

        The two are a `math.pi` apart and getting them confused is invisible
        in the code and glaring in the render: it rakes every fin forward, so
        the fish reads as being dragged backwards through the water, and it
        splays the caudal fan over the fish's own back. `face()` is the other
        one. Both exist so that neither has to be remembered.
        """
        u = clamp01(u)
        i = int(u * NU)
        j = min(NU, i + 8)
        k = max(0, i - 8)
        dy = (self.of_t[j] - self.of_t[k])
        dx = (j - k) / NU * self.L
        return self.ang + math.atan2(dy, dx if dx else 1.0)

    def face(self, u=0.0):
        """World angle the fish is looking along at `u`."""
        return self.axis(u) + math.pi

    def out_ang(self, u, side):
        """World angle pointing away from the spine on `side`."""
        return self.axis(u) + (math.pi * 0.5) * (1.0 if side * self.dorsal
                                                 > 0 else -1.0)

    # ---- render ---------------------------------------------------------
    def render(self, cv):
        L, W = self.L, self.W
        if L < 2.0 or W < 0.6:
            return
        ang = self.ang
        ca, sa = math.cos(ang), math.sin(ang)
        hw_t, of_t = self.hw_t, self.of_t
        dorsal = self.dorsal

        # --- shade tables ------------------------------------------------
        # One over u (the head), one over the *signed* flank. The signed one
        # is the departure from Blade, which only ever needs |v|.
        su_t = [0.0] * (NU + 1)
        # 墨色不匀 — how much ink the brush has left, along the body.
        #
        # The flank's tone was a single exact power curve from back to belly,
        # the same at every point along the fish, and that one fact was what
        # kept these bodies looking airbrushed after the silhouette had
        # already been fixed. No brush unloads evenly: it is wet at the
        # shoulder and dry by the tail, it catches on the paper, and the tone
        # of a single stroke varies along its length by much more than any
        # painter intends. A quarter either way over two cycles is enough —
        # it never reads as banding, it reads as ink.
        mu_t = [0.0] * (NU + 1)
        ma, mf = self.mamp, self.mfreq
        for i in range(NU + 1):
            u = i / NU
            su_t[i] = self.w_head * (1.0 - u) ** self.p_head
            mu_t[i] = (1.0 + ma * math.sin(self.ephase * 2.3
                                           + mf * TAU * u)
                       * (0.45 + 0.55 * u))
        # 墨渍 — a coarse two-dimensional blotch field over the whole flank.
        #
        # `mamp` varies the load along the body and the cross-spine ramp varies
        # it across, but their product is still a *separable* function, and a
        # separable field has no blotches in it: every iso-tone line runs either
        # along the fish or across it, which is why the flank kept coming out as
        # a smooth sheet of airbrushed grey however much the two profiles were
        # bent. Ink pools in patches. Three sinusoids at incommensurate
        # frequencies in u and v, sampled off a nine-band table so the inner
        # loop stays two integer ops, are enough to break the separability, and
        # breaking it is the entire point — the amplitude barely matters.
        nb = 9
        bl_t = None
        if self.bamp:
            ep = self.ephase
            bl_t = []
            for i in range(NU + 1):
                u = i / NU
                row_b = []
                for j in range(nb + 1):
                    vv = j / nb * 2.0 - 1.0
                    w = (math.sin(ep * 3.1 + 5.2 * u + 1.8 * vv)
                         + 0.70 * math.sin(ep * 5.7 + 11.0 * u - 2.6 * vv)
                         + 0.50 * math.sin(ep * 2.3 + 3.0 * u + 5.0 * vv))
                    row_b.append(1.0 + self.bamp * w * 0.46)
                bl_t.append(row_b)

        sv_t = [0.0] * (NT + 1)
        for j in range(NT + 1):
            vs = j / NT * 2.0 - 1.0           # -1 belly .. +1 back
            db = 0.5 + 0.5 * vs
            s = self.w_back * db ** self.p_back
            if self.w_rim:
                # Back edge only, or nearly. A wet stroke pools against the
                # edge the brush was *pressed* to, and on a fish that is the
                # back. Applied symmetrically — which is what `Blade` does,
                # because a petal has no back — it drew a definite grey line
                # along the belly from the chin to the wrist, and that one line
                # was what kept every fish looking like a 3-D model: a closed
                # outline with a lifted white underside inside it. 没骨 means
                # the ink simply stops.
                av = vs if vs >= 0.0 else -vs
                s += (self.w_rim * av ** self.p_rim
                      * (1.0 if vs >= 0.0 else 0.22))
            if self.hl:
                s -= self.hl * (1.0 - db) ** self.p_hl
            sv_t[j] = s

        # 腹虚 — the belly edge dissolved, as a multiplier on coverage.
        #
        # `soft` feathers the whole silhouette evenly, and an evenly feathered
        # closed outline is still a closed outline: with a lifted belly inside
        # it, every fish read as an airbrushed 3-D model of a fish — a smooth
        # pale underside meeting the paper along a definite line all the way
        # from the chin to the wrist. No 写意 painter closes that line. The
        # back of the stroke is where the brush was loaded and it has an edge;
        # the belly is where the hair ran dry and it has none, so the ink just
        # stops somewhere and the paper is allowed to be both water and fish.
        # Only the belly *rim* goes — the interior keeps its tone — so the
        # silhouette is lost without the volume going with it.
        af_t = [1.0] * (NT + 1)
        if self.vfade:
            for j in range(NT + 1):
                vs = j / NT * 2.0 - 1.0
                if vs < 0.0:
                    af_t[j] = 1.0 - self.vfade * (-vs) ** self.p_vfade
        # 一笔 — and the band this stroke covers, folded into the same table.
        vlo, vhi, vfe = self.vlo, self.vhi, max(1e-3, self.vfe)
        banded = vlo > -1.1 or vhi < 1.1
        if banded:
            for j in range(NT + 1):
                vs = j / NT * 2.0 - 1.0
                w = (vs - vlo) / vfe
                w2 = (vhi - vs) / vfe
                if w2 < w:
                    w = w2
                if w <= 0.0:
                    af_t[j] = 0.0
                elif w < 1.0:
                    af_t[j] *= w * w * (3.0 - 2.0 * w)
        use_af = bool(self.vfade) or banded

        # 飞白 — the dry edge.
        #
        # `soft` feathers the rim by a fixed amount everywhere, and a rim of
        # constant feather is an airbrushed edge however wide it is: the one
        # quality it can never have is *variation*, and variation along the edge
        # is the whole difference between a shape and a stroke. Real hair does
        # not deposit evenly — in one stretch the ink reaches the contour, in
        # the next the brush is riding on the paper's tooth and falls short of
        # it by a hair's width, and the silhouette comes out broken rather than
        # drawn. Three incommensurate sinusoids over u, biased low so that most
        # of the edge is intact and only some of it is eaten away.
        dry_t = None
        if self.dry:
            ep, df = self.ephase, self.dfreq
            dry_t = []
            for i in range(NU + 1):
                u = i / NU
                w = (math.sin(ep * 4.7 + df * TAU * u)
                     + 0.80 * math.sin(ep * 2.9 + df * 2.7 * TAU * u + 1.1)
                     + 0.60 * math.sin(ep * 8.3 + df * 0.6 * TAU * u + 2.2))
                # ×(0.42 + 0.92u) — a stroke is wet where it started and dry
                # where it ended, so the head end of the body keeps a clean
                # edge and the breaking up happens back toward the wrist.
                # Uniform dryness along the whole length is just a different
                # texture applied evenly, which is the fault it was added to
                # fix.
                dry_t.append(clamp01((w + 2.4) / 4.8) ** 1.7
                             * (0.42 + 0.92 * u))

        # 一笔 (along) — the u window, as a coverage multiplier.
        ulo, uhi, ufe = self.ulo, self.uhi, max(1e-3, self.ufe)
        u_win = ulo > -0.02 or uhi < 1.02
        au_t = None
        if u_win:
            au_t = [1.0] * (NU + 1)
            for i in range(NU + 1):
                u = i / NU
                w = (u - ulo) / ufe
                w2 = (uhi - u) / ufe
                if w2 < w:
                    w = w2
                if w <= 0.0:
                    au_t[i] = 0.0
                elif w < 1.0:
                    au_t[i] = w * w * (3.0 - 2.0 * w)

        # 留白 — patches the brush was taken *around*.
        #
        # A 红白金鱼's white markings were painted as soft discs of 白粉 over
        # the finished body, and that is the single most airbrushed-looking
        # thing a painter can do: an opaque pale cloud on a saturated red flank
        # is a specular highlight, so every goldfish came out with the shine of
        # a balloon on its shoulder. The white of a white patch is not paint at
        # all — it is the paper, and the way you get it is by not putting the
        # brush there. So a reserve is a hole in the body wash's *coverage*,
        # which under `mul` leaves exactly the untouched sheet, with the body's
        # own dry edge and bristle streaks running into it.
        #
        # Tabulated at the same resolution as everything else across the
        # flank, and `None` for every `u` no reserve reaches. A first attempt
        # used 25 bands across to keep the table small and the soft edge of the
        # patch came out as a flight of horizontal stairs — on a 300-pixel-deep
        # goldfish each band is a dozen pixels tall, and a quantised gradient
        # is a worse artefact than the highlight it replaced. Skipping the
        # untouched rows costs a `None` test in the loop and buys the full `NT`
        # resolution for about a quarter of the memory the dense table wanted.
        rs_t = None
        if self.resv:
            rs_t = [None] * (NU + 1)
            for i in range(NU + 1):
                u = i / NU
                near = [r for r in self.resv if abs(u - r[0]) < r[2]]
                if not near:
                    continue
                row_r = [1.0] * (NT + 1)
                for j in range(NT + 1):
                    vv = j / NT * 2.0 - 1.0
                    m = 1.0
                    for (ru, rv, au, av, st) in near:
                        du = (u - ru) / au
                        dv = (vv - rv) / av
                        q = du * du + dv * dv
                        if q < 1.0:
                            w = 1.0 - q
                            m *= 1.0 - st * w * w
                    row_r[j] = m
                rs_t[i] = row_r

        # 擦 — the dragged bristle, and the last thing standing between these
        # bodies and an airbrush.
        #
        # Everything else in this class varies the ink *smoothly*: `mamp` along
        # the stroke, the ramp across it, `bamp` in coarse patches, `dry` at
        # the rim. All of them are fields over a region, and a field over a
        # region is a wash. What a brush actually leaves is none of those — it
        # is a bundle of hairs, so a side stroke is a *bundle of lines*: narrow
        # streaks running the length of the mark where a hair stayed down,
        # separated by streaks where the paper was never touched at all.
        # Nothing else in the module makes a mark that is directional, and
        # direction is how a stroke says which way the hand went.
        #
        # So: a periodic ridge profile sampled *across* the stroke, drifting
        # slowly along it, multiplied into coverage rather than into tone. The
        # distinction matters and it is the same one the `mul` blend turned on:
        # taking coverage away lets the paper through, which is what a gap
        # between hairs is. Taking tone away would paint a pale line, and a
        # pale line inside a mark is a highlight — a different and much worse
        # thing, since it is the airbrush this was added to kill.
        #
        # A table of 64 so the inner loop is a multiply, a truncation and a
        # mask; the profile is four incommensurate harmonics so the streaks
        # come out unequal in width and weight. Evenly spaced streaks of equal
        # strength are a hatching pattern, which reads as engraving.
        #
        # Two things about it are not obvious and both were wrong first time.
        #
        # The pitch is in **pixels**, not in fractions of the local width. Read
        # off the normalised flank the streaks stretch and squeeze with the
        # envelope, so they run parallel to the silhouette and converge at both
        # ends — which is the grain of a plank, not the track of a brush. Hairs
        # are a fixed distance apart; the number of streaks across a stroke is
        # whatever the stroke's thickness divides into that, and at the thin
        # ends it is one or two.
        #
        # And the strength is broken up along the length as well as rising
        # toward the tail. A streak that runs unbroken from the head to the
        # wrist is a groove in the surface; a brush deposits in one stretch and
        # skips in the next, so the amplitude carries its own slow wobble and
        # the streaking comes and goes.
        gk_t = dr_t = sa_t = None
        if self.samp:
            ep = self.ephase
            gk_t = []
            for j in range(NG):
                t = j / NG * TAU
                w = (math.sin(t + ep)
                     + 0.64 * math.sin(2.0 * t + ep * 2.1)
                     + 0.41 * math.sin(3.0 * t + ep * 0.7)
                     + 0.27 * math.sin(5.0 * t + ep * 3.3))
                gk_t.append(clamp01((w + 1.55) / 2.6))
            dr_t = []
            sa_t = []
            for i in range(NU + 1):
                u = i / NU
                dr_t.append(NG * (0.42 * math.sin(ep * 1.3 + 2.3 * u)
                                  + 0.19 * math.sin(ep * 2.7 + 7.1 * u)))
                sa_t.append(self.samp * (0.26 + 1.08 * u)
                            * (0.56 + 0.44 * math.sin(ep * 3.9 + 5.3 * u)))

        # --- 分染 colour ramp, in HSV, exactly as Blade builds it --------
        light, deep = self.light, self.deep
        white = P("baifen")
        cr_t = [0.0] * (NS + 1)
        cg_t = [0.0] * (NS + 1)
        cb_t = [0.0] * (NS + 1)
        for j in range(NS + 1):
            sh = SH0 + j / SHK
            if sh < 0.0:
                c = hsv_mix(light, white, (sh / SH0) * 0.70)
            else:
                c = hsv_mix(light, deep, sh, bow=self.bow)
            cr_t[j], cg_t[j], cb_t[j] = c

        # --- bounding box -------------------------------------------------
        # A banded stroke covers one flank, so its box is lopsided: taking the
        # symmetric one costs a whole second body's worth of pixels that are
        # culled on the first table lookup, and the body is two thirds of this
        # module's render time.
        pad = 3.0 + 2.0 * self.soft
        t_a = max(-1.0, vlo - vfe) * dorsal
        t_b = min(1.0, vhi + vfe) * dorsal
        if t_b < t_a:
            t_a, t_b = t_b, t_a
        e_lo = t_a * self.hwmax - self.ofmax - pad
        e_hi = t_b * self.hwmax + self.ofmax + pad
        ux0 = max(-pad, (ulo - ufe) * L - pad)
        ux1 = min(L + pad, (uhi + ufe) * L + pad)
        if ux1 <= ux0:
            return
        xs, ys = [], []
        for ux, vy in ((ux0, e_lo), (ux1, e_lo), (ux1, e_hi),
                       (ux0, e_hi)):
            xs.append(self.cx + ux * ca - vy * sa)
            ys.append(self.cy + ux * sa + vy * ca)
        x0, y0, x1, y1 = cv.clip(min(xs), min(ys), max(xs), max(ys))
        if x0 >= x1 or y0 >= y1:
            return

        ir, ig, ib = self.ink
        line_k = self.line
        alpha = self.alpha
        isf = 1.0 / (1.0 + 2.0 * self.soft)
        dmin = -1.2 if self.soft <= 0.0 else -0.5 / isf - 0.3
        lw = 1.1 + W * 0.010
        inv_lw = 1.0 / lw
        lk = -self.lit * (gb.LIGHT[0] * -sa + gb.LIGHT[1] * ca)
        mul = self.blend == "mul"
        nbh = nb * 0.5
        dry = self.dry
        dw = W * 0.26 + 1.5
        inv_dw = 1.0 / dw
        # cycles per *pixel* across the stroke: `sfreq` is a count over the
        # half-width at the body's thickest, and everything thinner gets
        # proportionally fewer streaks.
        sfq = self.sfreq * NG / max(3.0, self.hwmax)
        ngm = NG - 1
        buf = cv.buf
        cw = cv.w
        cx, cy = self.cx, self.cy
        Lx = L + 1.5
        nt2 = NT * 0.5

        for py in range(y0, y1):
            dy = py + 0.5 - cy
            row = py * cw
            dyc, dys = dy * ca, dy * sa
            for px in range(x0, x1):
                dx = px + 0.5 - cx
                x = dx * ca + dys
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
                yv = y - of_t[iu]
                ay = yv if yv >= 0.0 else -yv
                d = hw - ay
                if d < dmin:
                    continue
                # 收口 — close both ends. The snout is a blunt cap and the
                # wrist is a cut, so both are just the axial distance.
                d2 = L - x
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
                if d < dw:
                    # The streaks stop short of the rim. They are a thing the
                    # middle of a stroke does; run out to the contour they
                    # chop it into horizontal steps, and a terraced back is a
                    # worse artefact than the smooth one it replaced. 飞白 is
                    # what breaks the edge, and it lives in the same band.
                    srim = d * inv_dw
                    if dry_t is not None:
                        cov *= 1.0 - dry * dry_t[iu] * (1.0 - srim)
                else:
                    srim = 1.0
                if au_t is not None:
                    cov *= au_t[iu]
                    if cov <= 0.0:
                        continue
                a = cov * alpha
                if a <= 0.004:
                    continue

                ts = yv / hw
                if ts > 1.0:
                    ts = 1.0
                elif ts < -1.0:
                    ts = -1.0
                vs = ts * dorsal
                ivs = int((vs + 1.0) * nt2)
                if use_af:
                    a *= af_t[ivs]
                    if a <= 0.004:
                        continue
                if gk_t is not None:
                    a *= 1.0 - sa_t[iu] * srim * (
                        1.0 - gk_t[int(yv * sfq + dr_t[iu]) & ngm])
                if rs_t is not None:
                    rr = rs_t[iu]
                    if rr is not None:
                        a *= rr[ivs]
                        if a <= 0.004:
                            continue
                mk = mu_t[iu]
                if bl_t is not None:
                    mk *= bl_t[iu][int((vs + 1.0) * nbh)]
                sh = su_t[iu] + sv_t[ivs] * mk + lk * ts
                if sh > 1.0:
                    sh = 1.0
                elif sh < SH0:
                    sh = SH0
                isx = int((sh - SH0) * SHK)
                r = cr_t[isx]
                g = cg_t[isx]
                b = cb_t[isx]

                if line_k and d < lw:
                    k = (1.0 - d * inv_lw)
                    k = k * k * line_k
                    r += (ir - r) * k
                    g += (ig - g) * k
                    b += (ib - b) * k

                i = (row + px) * 3
                if mul:
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
# 线 — one line, drawn densely
# --------------------------------------------------------------------------

def _line(cv, pts, w0, w1, color, alpha):
    """`brush_path`, resampled so the capsules overlap.

    `brush_path` lays one tapered capsule per segment of the polyline it is
    given, and a polyline with a dozen points in it has two visible faults at
    any size worth looking at. The capsules are straight, so a curve comes out
    as a polygon — the eye ring was a visible dodecagon — and consecutive
    capsules overlap only in a small lens near each shared endpoint, so a
    semi-transparent stroke double-darkens there and the line arrives as a
    string of beads. Both are artefacts of sampling, not of the brush, and
    both go away if the step is cut to about the stroke's own width: the
    facets fall below a pixel and the overlaps become continuous, so the
    accumulated alpha is flat along the whole length instead of pulsing.

    Resampling is linear, which is enough — the input points already describe
    the curve, this only subdivides it.
    """
    n = len(pts) - 1
    if n < 1:
        return
    step = max(1.1, min(w0, w1) * 0.9)
    out = [pts[0]]
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        m = max(1, int(math.hypot(dx, dy) / step))
        for j in range(1, m + 1):
            t = j / m
            out.append((x0 + dx * t, y0 + dy * t))
    brush_path(cv, out, w0, w1, color, alpha)


# --------------------------------------------------------------------------
# small shared marks
# --------------------------------------------------------------------------

def arc(cv, pts_fn, n, w0, w1, color, alpha):
    """A brush arc from a parametric function — gill covers, scale rows,
    ripples. Kept as one `brush_path` and not a chain of segments, because
    every `brush_seg` joint is a round cap composited twice and shows as a
    bead of ink."""
    pts = [pts_fn(i / (n - 1.0)) for i in range(n)]
    brush_path(cv, pts, w0, w1, color, alpha)


def ellipse_arc(cv, cx, cy, rx, ry, rot, a0, a1, w0, w1, color, alpha):
    c, s = math.cos(rot), math.sin(rot)
    n = max(4, int(abs(a1 - a0) * max(rx, ry) / 3.0) + 4)

    def f(t):
        a = a0 + (a1 - a0) * t
        px, py = math.cos(a) * rx, math.sin(a) * ry
        return cx + px * c - py * s, cy + px * s + py * c

    arc(cv, f, n, w0, w1, color, alpha)


# --------------------------------------------------------------------------
# 鳍 — fins. A fin is a Spindle, which is to say a stroke.
#
# It was a `Blade` for a long time, on the reasoning that `Blade`'s vein fan is
# literally 鳍条 and a fin is literally a blade shape. Both true, and it was
# still the wrong primitive, for the same reason the body could not be one:
# `Blade` is a *shape* with a shaded interior and an even rim. Six versions of
# the fins were spent making that shape less leaf-like — narrower, blunter,
# flatter, fanned, in threes — and they still came out as pale translucent
# feathers set against the fish rather than marks made by the same brush that
# made the body.
#
# A `Spindle` is a stroke: ink where the hair landed (`w_head`, which is the
# *root* of a fin, since u runs root → tip), running out toward the free end,
# a centreline that can curve, and a dry broken edge. The last of those is what
# a fin actually needs most — a membrane's outline is *interrupted*, and no
# amount of softening an uninterrupted one gets there.
# --------------------------------------------------------------------------

# Pinched where it leaves the body, opening out, blunt and dissolving at the
# free end. The reverse of a leaf, which is why `Blade`'s defaults always had
# to be fought. One profile does every fin and every tail lobe on all four
# species: a fin is a fin.
FIN_PROF = [(0.00, 0.20), (0.12, 0.62), (0.30, 0.90), (0.50, 1.00),
            (0.70, 0.98), (0.86, 0.88), (1.00, 0.70)]


def _fin_stroke(cv, rng, bx, by, a, L, W, pal, alpha, soft, frill, ink,
                dry=0.62, wob=0.09, load=0.94):
    """One stroke of a fin or a tail lobe.

    `w_head` is the whole idea: it is the ink laid down at `u = 0`, and `u = 0`
    is where the brush touched the body. So the mark is loaded at the root and
    dries toward the tip, with a bit of cross-stroke weight on one side so it
    is not uniform across its width, and no 腹虚 or pooled rim at all — those
    are a body's business.

    The alpha is pushed *up*, not down. These overlap and multiply, and the
    fins' long-standing fault was being too faint to read as marks: the ink has
    to be in the stroke before any of its shape can matter.
    """
    if L < 1.5 or W < 0.4:
        return
    lt, dp = pal["fin"]
    Spindle(bx, by, a, L, W, FIN_PROF,
            swing=wob * rng.choice((-1.0, 1.0)),
            phase=rng.uniform(0.3, 1.0), freq=rng.uniform(1.2, 2.0),
            light=rgb(lt), deep=mix(rgb(dp), rgb(pal["ink"]), 0.34),
            ink=ink or rgb(pal["ink"]), line=0.0, dorsal=1.0,
            w_head=load, p_head=1.15,
            w_back=0.20, p_back=1.0, w_rim=0.0, hl=0.0, vfade=0.0,
            # A fin is a tenth the length of the body, so the same `dfreq` that
            # breaks a body edge into a few patches cuts a fin into teeth: at
            # 7–13 cycles over a stroke this short every fin came out as a
            # sawtooth and the fish grew feathers. Two or three undulations over
            # the whole stroke is a dry edge; ten is a comb.
            dry=dry, dfreq=rng.uniform(2.2, 4.0),
            # 鳍条 come out of the overlaps, not out of a vein fan: three
            # strokes multiplied over each other already give a dark root and
            # a frayed free edge. Drawn as actual ribs they read as a
            # caterpillar, and 写意 fish painters do not draw them at all.
            # `eamp` here is the frill of a 金鱼's veil, so it is wound right
            # up for those and left as ordinary brush wobble for everyone else.
            eamp=0.055 + frill * 0.55, efreq=rng.uniform(1.4, 2.6),
            ephase=rng.uniform(0, TAU),
            # 鳍 is the one place the bristle streaks are literal: a fin
            # really is a row of rays, and a dry brush dragged out from the
            # root leaves exactly that without a single ray having to be
            # drawn. Stronger than on the body, and pitched to give four or
            # five streaks across a fin rather than ten.
            samp=0.46, sfreq=rng.uniform(1.6, 2.6),
            mamp=0.34, mfreq=rng.uniform(1.5, 2.6), bamp=0.0,
            soft=soft, bow=0.05, blend="mul",
            alpha=min(1.0, alpha)).render(cv)


def _fin(cv, rng, body, fin, pal, scale, alpha, soft, rays, side=1.0,
         ink=None, frill_k=1.0):
    """One fin, hung on the body at `fin["u"]` on `side`.

    A fin is *two or three narrow strokes fanned from one base*, not one
    shape. The first version drew it as a single wide `Blade` and the result
    was the module's most persistent failure: every fish carried a row of
    three grey oak leaves under its belly, clearly separate objects that
    happened to be touching it. Four things were wrong at once and they all
    have the same cause — a fin is a translucent membrane stretched on rays,
    so anything with a single smooth closed outline reads as a solid.

    The tail was always the convincing fin on these fish, and the tail was
    the one drawn as a fan of lobes. So this does the same. Narrow strokes
    overlapping at the root give a dark base and a frayed free edge for free,
    out of the multiply, without drawing a ray or a contour.

    The direction is the outward normal rotated *toward the tail* by `sweep`,
    interpolating along the shorter arc so the same number works on either
    flank and whichever way the fish faces. A fin raked forward is the single
    most reliable way to make a fish look like it is being dragged backwards.

    `Blade`'s defaults are exactly wrong for this shape. A `Blade` is narrow
    at the base, broadest around the middle and drawn to a point — which is a
    leaf. A fin ray is pinched where it meets the body and *opens* outward, so
    the profile widens as fast as the field can manage (low `k1`, low `e1`)
    and holds that width out to a soft blunt end (high `k2`, low `e2`).
    """
    u = fin["u"]
    nrm = body.out_ang(u, side)
    tgt = body.axis(u)
    dlt = math.atan2(math.sin(tgt - nrm), math.cos(tgt - nrm))
    base = nrm + dlt * fin["sweep"] * rng.uniform(0.94, 1.06)
    L0 = scale * fin["len"] * rng.uniform(0.92, 1.08)
    if L0 < 1.5:
        return
    # How many strokes: a big fin on a near fish gets three, a small or far
    # one gets two. One is never enough — one stroke is a leaf again.
    n = 3 if L0 > scale * 0.22 and rays > 0.0 else 2
    frill = fin.get("frill", 0.0) * frill_k
    # How wide the fan opens. At wid×1.5 — about 0.8 rad for a pectoral —
    # the three strokes separated into three distinct pointed lobes and every
    # fin read as a little maple leaf. A fin fans by a few degrees, not by
    # fifty: the strokes have to stay mostly on top of each other so what
    # shows is one membrane with a frayed edge.
    span = fin["wid"] * 0.55
    for k in range(n):
        f = (k / (n - 1.0)) * 2.0 - 1.0                      # -1 .. +1
        a = base + f * span * rng.uniform(0.80, 1.15)
        # the leading stroke is the long one, so the fin's free edge rakes
        # back instead of sitting square to the body
        L = L0 * (0.80 + 0.20 * (1.0 - f) * 0.5) * rng.uniform(0.88, 1.06)
        # Rooted well inside the silhouette, and at a slightly different
        # point per stroke so the base is a base and not a pivot. Burial is
        # free here because `paint_fish` lays the fins down before the body
        # wash — see the note there.
        bu = u + (0.0 if n == 1 else f * fin["len"] * 0.10)
        # Rooted *at* the silhouette, not well inside it.
        #
        # At 0.6–0.8 of the half-width the root was a fifth of the body's depth
        # in from the edge, and since these fins are raked nearly flat along the
        # flank, the first third of every stroke ran inside the silhouette. That
        # was free while the body was opaque paint laid down afterwards. The
        # body is a transparent wash now, so it hides nothing: every buried root
        # showed through the flank as a dark streak, and a fish with four grey
        # smears inside it reads as a fish painted on blotting paper.
        bx, by = body.pos(bu, side * rng.uniform(0.90, 1.02))
        # Softer than the body — a fin has almost no contour, and one with a
        # crisp outline is a petal. But only a little softer. At 3× the fins
        # went so diffuse that a pale wide blur hanging off the far side of the
        # fish read as the fish's *shadow*, which is the one thing a painting
        # with no water in it must not have.
        _fin_stroke(cv, rng, bx, by, a, L, L * fin["wid"] * 0.40,
                    pal, alpha * rng.uniform(1.04, 1.26), soft * 1.5,
                    frill, ink, dry=0.28,
                    wob=rng.uniform(0.05, 0.13))


def _tail(cv, rng, body, spec, pal, scale, alpha, soft):
    """尾鳍 — the caudal fin, in lobes, straight off the wrist.

    Every lobe starts at the same point and is splayed about the backward
    axis, so the tail opens like a fan rather than sprouting from several
    different places. The lobes are drawn outermost-first, which puts the
    heaviest overlap — and so the darkest ink, since these multiply — down the
    middle where the fin is actually thickest.

    A tail is *strokes*, not a shape. The first draft drew two broad lobes and
    got two leaves pinned to the fish's back end; the fix was to make each
    lobe narrow (a sixth of its own length) and use five or six of them, which
    is both how a brush does it and how the fin's rays come out for free, out
    of the overlaps, without drawing a single one.
    """
    t = spec["tail"]
    back = body.axis(1.0)
    n = t["n"]
    lt, dp = pal["fin"]
    order = sorted(range(n), key=lambda k: -abs(k - (n - 1) * 0.5))
    for k in order:
        f = 0.0 if n == 1 else (k / (n - 1.0)) * 2.0 - 1.0     # -1 .. +1
        a = back + f * t["splay"] * rng.uniform(0.88, 1.12)
        # 燕尾 — the middle lobes are the short ones, so the fin's free edge
        # is forked rather than round. Every forked tail in nature works this
        # way and a round one reads as a frayed paintbrush.
        L = (scale * t["len"] * rng.uniform(0.88, 1.12)
             * (0.76 + 0.24 * abs(f)))
        if L < 1.5:
            continue
        # Rooted at a slightly different point per lobe. Started from one
        # shared point they all multiplied over each other there and left a
        # hard dark triangle at the wrist, which is a stain, not a joint.
        bx, by = body.pos(rng.uniform(0.90, 1.0), f * rng.uniform(0.18, 0.86))
        # A lobe is the same stroke as a fin, loaded at the wrist and dry at
        # the free end — which is what a tail is, and why both now go through
        # one function. The one thing it needs more of is ink: a 写意 fish is
        # read by its tail before its head, and at the paired fins' weight this
        # was a grey ghost trailing a solid body.
        # `load` well below a fin's. Six lobes all loaded at the root and all
        # rooted within a twentieth of the body of each other multiply into a
        # near-black wedge at the wrist, which is the oldest fault in the tail
        # and came straight back the moment the lobes stopped being pale. A
        # tail is dark *down the middle of the fan*, from the overlaps, not at
        # the single point where it meets the fish.
        _fin_stroke(cv, rng, bx, by, a, L, L * t["wid"] * 0.54,
                    pal, alpha * rng.uniform(0.92, 1.08),
                    soft * t["soft"] * 1.15, t["frill"], None,
                    dry=0.26, wob=rng.uniform(0.07, 0.18), load=0.62)


# --------------------------------------------------------------------------
# 鱼 — one fish, painted in the order a brush would
# --------------------------------------------------------------------------

def _hex(c):
    """A colour triple back to "#rrggbb". The palettes are stored as hex pairs
    and every call site turns them into triples with `rgb()`, so a derived
    palette has to go back the way it came."""
    return "#%02X%02X%02X" % tuple(
        max(0, min(255, int(v + 0.5))) for v in c)


def _fade(pal, depth):
    """A copy of the palette with every loaded stop pulled toward its own pale
    stop, by `depth`.

    The first version of `paint_fish` spent depth on `alpha` alone, and that is
    not 远淡近浓, it is translucency. A fish at alpha 0.4 over bare paper still
    arrives at very nearly the ink value it was given, because the paper it is
    thinning into is nearly white and the fish's own dark stop is nearly black
    — the contrast *within* the mark barely moves, and contrast is what the eye
    reads distance from. Washing out the stops themselves is what a painter
    actually does with a far fish: less ink on the brush, not more water in
    front of it. Keep both, they do different jobs.
    """
    if depth <= 0.0:
        return pal
    f = min(0.46, depth * 0.56)
    out = dict(pal)
    for key in ("body", "fin", "weed", "lotus"):
        lt, dp = pal[key]
        out[key] = (lt, _hex(mix(rgb(dp), rgb(lt), f)))
    out["ink"] = _hex(mix(rgb(pal["ink"]), rgb(pal["body"][0]), f))
    return out


def paint_fish(cv, rng, spec, pal, x, y, L, ang, depth=0.0, occ=None):
    """One fish. (x, y) is the snout, `ang` the way it *faces* — not the body
    axis, which is `ang + pi` — and `L` the body length from snout to the
    wrist of the tail, so the whole mark, tail included, is longer than `L`.

    `depth` is 0 for the nearest fish and 1 for the farthest, and it is spent
    on 远淡近浓: the far fish lose ink, gain a soft edge, and lose the small
    marks entirely. 分墨 is how a flat sheet gets three planes, and it matters
    more here than in a flower painting, because there is no ground, no
    horizon and no overlap to say which fish is in front.

    The order is the painter's, which is also the occlusion order: 远鳍 (the
    far-side pectoral) → 身 → 尾 → 近鳍 → 鳃 → 鳞 → 口 → 须 → 点睛. The eye
    is last in every manual on the subject.
    """
    k = 1.0 - depth
    pal = _fade(pal, depth)
    alpha = 0.52 + 0.48 * k
    soft = (0.35 + 1.35 * depth) * max(0.5, L * 0.006)
    scale = L
    swing = rng.uniform(*spec["swing"]) * rng.choice((-1.0, 1.0))

    # 两笔成鱼 — the body is two strokes, not one shape.
    #
    # One Spindle over the whole envelope gives a closed convex silhouette
    # filled with a single smooth ramp, and that is an airbrushed solid no
    # matter how well the ramp is tuned or how finely the rim is feathered:
    # the thing it has no way to show is 笔. A 写意 painter does not draw a
    # fish and shade it. They load the brush, lay it on its side and pull one
    # stroke from the head along the back to the wrist — that stroke *is* the
    # back — and then take what is left on the hair and drag a second, drier,
    # paler one along underneath. What makes the result read as painted is
    # everything that follows from there being two of them: the seam where
    # they overlap, the two edges having different qualities, the two
    # silhouette wobbles not matching, and the paper left showing between
    # them where the second stroke ran out.
    #
    # Both strokes share the envelope, the spine wave and the profile, so they
    # are the same fish; each is cut out of it by a band over the signed flank,
    # and each gets its own brush — its own `eamp` wobble, its own ink load,
    # its own dry patches, its own edge.
    bw = L * spec["wid"] * 0.5 * rng.uniform(0.90, 1.12)
    ph = rng.uniform(0, TAU)
    fq = rng.uniform(*spec["freq"])
    b_lt, b_dp = rgb(pal["body"][0]), rgb(pal["body"][1])

    # 白斑 — the white markings of a 红白金鱼, as reserves rather than paint.
    # Chosen here because both body strokes have to know about them: a patch
    # the back stroke goes round and the belly stroke paints over is not a
    # patch. Two or three overlapping ellipses each, so the edge of the
    # reserve is as irregular as a brush going round something.
    resv = []
    if spec.get("patch") and pal.get("piebald") and depth < 0.6:
        for _ in range(1 if rng.random() < 0.6 else 2):
            pu = rng.uniform(0.24, 0.70)
            pv = rng.uniform(0.30, 0.92)
            pr = rng.uniform(0.085, 0.145)
            for _ in range(rng.randint(2, 3)):
                resv.append((pu + rng.gauss(0.0, 0.045),
                             pv + rng.gauss(0.0, 0.22),
                             pr * rng.uniform(0.70, 1.15),
                             pr * rng.uniform(2.4, 4.2) / spec["wid"],
                             rng.uniform(0.62, 0.88)))
    resv = tuple(resv)

    def stroke(**kw):
        """One pass of the body brush. `ang + pi`: the caller says which way
        the fish looks, Spindle wants the axis. `bw` is already a half-width —
        the species table gives depth, because depth is what is readable.

        忌同形 — the envelope is jittered by a tenth per fish. Scaled copies
        of one outline are as monotonous as equal spacing: five goldfish of
        identical proportion read as five impressions of one woodblock, and
        the eye catches that faster than it catches equal gaps.
        """
        d = dict(
            light=b_lt, deep=b_dp, ink=rgb(pal["ink"]),
            w_head=spec["head"], w_rim=spec["rim"], line=0.0,
            # 没骨 — the body is a *wash*, so it multiplies into the paper.
            #
            # Painted `over`, the pale end of the ramp is opaque pigment a
            # little lighter than the paper, so the belly arrived as a pale
            # shape with a definite outline all the way from the chin to the
            # wrist, and no amount of feathering or dissolving that edge
            # helped: the tone difference *was* the edge. Multiplied, the pale
            # end is the paper — 1.0 × paper — so the belly has nowhere to
            # stop, and the same table that drew a plastic 3-D model reads as
            # ink soaking in. It is also what lets two strokes overlap into a
            # seam instead of the upper one simply hiding the lower.
            blend="mul", hl=0.10,
            # 背浓腹淡, pushed hard because the wash is transparent. At the
            # default 1.65 the spine itself carried a third of the load and
            # the tone then crawled down to the belly, so the whole flank was
            # a middling warm grey — ink everywhere is the same as ink
            # nowhere. At 2.5 the load sits in the top third of the stroke,
            # which is what a loaded brush laid on its side actually leaves.
            p_back=2.5,
            # A quarter, not a third, and at a short wavelength: a broad
            # gentle lightening in the middle of a round red body does not
            # read as 墨渍, it reads as the shine on a balloon.
            bamp=rng.uniform(0.14, 0.24),
            eamp=rng.uniform(0.050, 0.090), efreq=rng.uniform(2.2, 4.2),
            ephase=rng.uniform(0, TAU),
            samp=rng.uniform(0.15, 0.26), sfreq=rng.uniform(3.6, 5.8),
            lit=0.13 * k, alpha=alpha, resv=resv,
        )
        d.update(kw)
        return Spindle(x, y, ang + math.pi, L, bw, spec["prof"],
                       swing=swing, phase=ph, freq=fq, **d)

    # 腹笔 — the second stroke, and the one the brush had almost nothing left
    # for. A weak even wash (no pooled edge, no lifted belly, a shallow ramp),
    # badly broken along the bottom, and reaching a little past the spine so
    # that it has something to overlap. Its job is not to colour the belly in:
    # it is to leave most of the belly as paper and put a *little* weight under
    # the fish so the back stroke is not floating.
    # The two ways of not closing a silhouette are not interchangeable, and
    # getting that wrong cost a pass. `vfade` takes the belly rim down by the
    # same fraction along the whole length, so the contour is uniformly weak —
    # present everywhere, convincing nowhere, and at 0.70 the underside of the
    # fish vanished altogether and the ventral fins were left attached to bare
    # paper. `dry` takes it away in *patches*, so the belly edge is solid for a
    # stretch, gone for a stretch, and solid again. That is what a brush leaves
    # and it is also the only version that still gives the fins something to
    # come out of. Most of the dissolve is now the dry field's.
    # 不到尾 — and it stops before the wrist. A second stroke that runs the
    # full length of the first one makes a closed shape again, however dry its
    # edge is, because between them the two still describe every part of the
    # contour. Lifting the belly brush at about three quarters leaves the whole
    # lower rear quadrant as paper, and that is the single opening that keeps
    # the fish from being a filled outline: the back stroke alone carries on to
    # the tail, so the body *runs out* into the wrist instead of closing round
    # it. This is also just what the stroke order gives you — the belly is the
    # short second pull of a brush that is already nearly dry.
    belly = stroke(
        uhi=rng.uniform(0.68, 0.80), ufe=0.26,
        samp=rng.uniform(0.24, 0.38), sfreq=rng.uniform(2.6, 4.2),
        vlo=-1.30, vhi=0.02, vfe=0.72,
        deep=mix(b_lt, b_dp, 0.52),
        # p_back well *under* one, which looks wrong and is the only thing that
        # works. The ramp is a power of the distance up from the belly and it
        # is indexed over the whole flank, not over the band, so inside the
        # belly stroke's own band every value of it is down at the pale end —
        # and the pale end of a wash that multiplies is the paper. At 0.92 this
        # stroke was laying near-white onto cream and doing nothing at all,
        # which is exactly how it looked: two passes and still no underside to
        # the fish. Below one, the curve lifts off its floor immediately and
        # the stroke has ink in it from its bottom edge up.
        w_back=0.72, p_back=0.34, w_rim=0.0, hl=0.0,
        vfade=0.30 + 0.16 * depth, p_vfade=1.4,
        dry=0.74, dfreq=rng.uniform(5.0, 9.0),
        soft=soft * 2.6, alpha=alpha * 0.80)
    # 背笔 — the loaded one, and the fish's whole structure. Its lower edge is
    # where the hair lifted off the paper, so the band is feathered wide and
    # there is no 腹虚 term: below the band there is simply no stroke, which is
    # a cleaner way to not close a silhouette than fading one out. Its upper
    # edge is the pressed one and stays the crispest thing on the fish —
    # soft×1.15, where the single-stroke body needed ×1.5 to stop looking like
    # vector art, because a stroke with a visible dry edge no longer has to
    # prove it is not one.
    #
    # `p_back` drops from 2.5 to 1.05 (through 1.45 and 1.25), and that is not
    # a tweak — it is the
    # correction for the first thing the split broke. The ramp runs over the
    # whole flank while the band only keeps the top of it, so the two biases
    # compound: at 2.5 the mid-flank was down to a sixth of the load and the
    # stroke had collapsed into a narrow dark ribbon along the dorsal contour.
    # A fish with that on it does not read as a fish painted in two strokes,
    # it reads as a fish with an eyebrow. The band is what decides where the
    # stroke *ends*; the ramp only has to shade it.
    # 墨骨 — the loaded stroke's dark end is pulled a third of the way into
    # the ink. This is the order a painter works in: the back is laid in with
    # 墨 and the colour goes on over it, so what is dark on a 朱砂 goldfish is
    # dark *because of ink*, not because of more vermilion. Drawn from the
    # pigment's own loaded stop the coloured palettes came out as a smooth
    # salmon gradient from cream to bright red, which is a plastic fish of a
    # different colour. 水墨 is unaffected — its loaded stop is already ink.
    body = stroke(
        vlo=-0.80, vhi=1.30, vfe=0.96, p_back=1.05, w_back=1.30,
        deep=mix(b_dp, rgb(pal["ink"]), 0.44),
        # 色不可灰 — bow the ramp's saturation out at the middle.
        #
        # The two stops of a coloured palette are a cream and a loaded pigment,
        # and interpolating between them in HSV walks saturation linearly, so
        # the whole mid-flank — which is most of the fish — sits at half
        # chroma. On 朱砂 that is not vermilion thinned with water, it is
        # salmon: a 金鱼 painted in it came out the colour of a plastic toy
        # however much ink went into the back. Real 朱砂 thinned on the palette
        # stays red a long way down; it loses *value* faster than it loses
        # chroma. Bowing saturation up at mid-ramp is that curve, and on the
        # two ink palettes it costs nothing, because a neutral has no chroma to
        # bow.
        bow=0.32,
        line=spec["line"] * k, vfade=0.0,
        dry=0.52 + 0.12 * depth, dfreq=rng.uniform(4.0, 7.0),
        soft=soft * 1.15)

    fin_a = (0.30 + 0.34 * k)
    # 鳍条 — a hint only. At 0.26 the rays came out as a row of regular
    # ribs and every fin looked like a caterpillar; 写意 fish painters do not
    # draw them at all.
    rays = 0.0 if depth > 0.55 else 0.11 * k

    # 鳍在身前 — every fin goes down *before* the body wash.
    #
    # This is the single change that made the fins work, and it took three
    # earlier attempts to find, because the symptom looked like two unrelated
    # bugs. Painted after the body, each fin's buried root showed *through* the
    # flank as a pale lozenge — the fins multiply, so the third of each stroke
    # that is inside the silhouette stained the belly — and at the same time
    # the outer part read as a separate leaf lying against the fish, because
    # there was no point anywhere along the join where fin and body were the
    # same mark.
    #
    # Underneath, the body wash hides every root and the fin emerges from the
    # silhouette edge, which is both what an anatomy does and what a brush
    # does: 写意 painters lay the fins in first and drop the body over them.
    # It also fixes the tail's wrist for free, where the lobes used to pile up
    # into a hard dark triangle.
    #
    # 远鳍 — the far-side pectoral, shorter and paler than the near one and on
    # the same (belly) side; a pectoral fin high on the back is a second
    # dorsal. It is the cheapest depth cue a fish has, and the only way a
    # sheared 2-D field gets to say anything at all about the far flank.
    if depth < 0.75:
        _fin(cv, rng, body, spec["pect"], pal, scale * 0.78,
             fin_a * 0.60, soft * 1.8, 0.0, side=-1.0,
             frill_k=0.6)
    # The caudal gets half again the paired fins' ink. It is the fin that
    # carries the 势 — a fish is read by its tail before its head — and at the
    # paired fins' alpha it came out a pale grey ghost trailing behind a solid
    # body, which looks less like a tail than like motion blur.
    _tail(cv, rng, body, spec, pal, scale, min(1.0, fin_a * 1.55), soft)
    # 近鳍 — dorsal on the back, the rest on the belly.
    # the dorsal gets more ink than the paired fins, which is both how it
    # looks — it is out of the water's way, so it is the fin a painter can
    # actually see — and necessary, since it has the paper behind it rather
    # than the body and so nothing to tell against
    _fin(cv, rng, body, spec["dorsal"], pal, scale, fin_a * 1.25, soft, rays,
         side=1.0)
    for key in ("vent", "anal"):
        _fin(cv, rng, body, spec[key], pal, scale, fin_a, soft, rays,
             side=-1.0)
    # the near pectoral roots a little inside the edge, which is the only
    # thing in a sheared 2-D field that says it is on the near flank
    _fin(cv, rng, body, spec["pect"], pal, scale, fin_a * 0.95, soft, rays,
         side=-0.86)

    belly.render(cv)
    body.render(cv)

    ink = rgb(pal["ink"])
    nong = mix(ink, I("nong"), 0.4)

    # 背鳍硬棘 — 鳜鱼's saw. Drawn as separate spines over the dorsal wash,
    # because a serrated silhouette alone does not read as *spines*; what reads
    # is a row of stiff lines all leaning the same way.
    #
    # They have to stay *inside* the dorsal wash and they have to arch. Eight
    # of them, all the same length, standing straight up and projecting well
    # past the fin, is a picket fence — and a picket fence on a fish's back
    # does not read as spiny, it reads as a mistake. The length is therefore
    # scaled by a hump across the fin's base and capped below the wash's own
    # length, and they rake back with it.
    if spec["spine"] and depth < 0.6:
        d = spec["dorsal"]
        n_sp = rng.randint(4, 6)
        # Unevenly spaced, unequal, curved, and rooted *under* the wash.
        #
        # Evenly spaced straight bars of one length at one angle are a comb,
        # and that is what this was — eight identical grey tally marks over the
        # fish's shoulder, easily the most mechanical thing left in the module.
        # Three separate properties had to go: the spacing (a real fin's rays
        # crowd at the front and open out behind), the straightness (a spine
        # that carries a membrane bows back under it), and the root, which was
        # out at 0.84 of the flank where the wash has already thinned, so every
        # spine began in mid-air. Started at 0.52 the first third of each one is
        # buried and the fin reads as stiffened rather than decorated.
        fs = sorted(rng.uniform(0.0, 1.0) for _ in range(n_sp))
        for i, f in enumerate(fs):
            t = d["u"] + 0.05 + f * 0.24
            px, py = body.pos(t, 0.52)
            a = body.out_ang(t, 1.0)
            tl = body.axis(t)
            dl = math.atan2(math.sin(tl - a), math.cos(tl - a))
            a += dl * (0.22 + 0.20 * f) + rng.gauss(0.0, 0.09)
            hump = 0.34 + 0.40 * math.sin(math.pi * min(1.0, f * 1.15))
            ln = scale * d["len"] * hump * rng.uniform(0.60, 1.00)
            # The bow, walked in four steps and raking further back as it
            # leaves the body — the same arc-length integration the barbels
            # needed, for the same reason.
            cur = dl * rng.uniform(0.10, 0.26)
            sx, sy, aa = px, py, a
            sp = [(sx, sy)]
            st = ln / 4.0
            for _ in range(4):
                aa += cur * 0.25
                sx += math.cos(aa) * st
                sy += math.sin(aa) * st
                sp.append((sx, sy))
            _line(cv, sp, max(0.45, L * 0.0058), max(0.26, L * 0.0014),
                  nong, (0.26 + 0.24 * k) * rng.uniform(0.7, 1.0))

    # 斑 — mottling. Kept well inside the silhouette: these are brush dabs on
    # a wet body, and one that crosses the edge turns into a dent in the fish.
    # Each blotch is two or three overlapping squashed dabs at a shared angle
    # rather than one round disc, because 鳜鱼's markings are irregular dark
    # clouds and a scatter of even circles reads as bubbles trapped under the
    # fish rather than as pattern on it.
    if spec["mottle"] and depth < 0.7:
        for _ in range(int(rng.randint(4, 7) * k) + 2):
            u = rng.uniform(0.14, 0.88)
            v = rng.uniform(-0.45, 0.82)
            r = L * rng.uniform(0.028, 0.060) * (1.0 - 0.35 * abs(v))
            for _ in range(rng.randint(2, 3)):
                du = u + rng.gauss(0.0, 0.030)
                dv = v + rng.gauss(0.0, 0.22)
                px, py = body.pos(du, dv)
                disc(cv, px, py, r * rng.uniform(0.6, 1.0),
                     mix(rgb(pal["body"][1]), ink, 0.45),
                     rng.uniform(0.14, 0.28) * k, soft=0.75,
                     squash=rng.uniform(0.45, 0.8))

    # 鳃 — the gill cover: one arc from the back down across the cheek. The
    # mark that separates head from body, and on an ink fish it is often the
    # only line in the whole painting.
    # Thin, and barely bowed. Drawn once at L*0.013 it came out as a thick
    # black V across the fish's shoulder — `brush_seg`'s width is a *radius*,
    # so that was a 13-pixel band, and the bow was set so far back that the
    # two halves met at an angle instead of curving.
    if depth < 0.8:
        g0 = body.pos(spec["eye_u"] + 0.095 + rng.gauss(0, 0.006), 0.74)
        gc = body.pos(spec["eye_u"] + 0.150 + rng.gauss(0, 0.010),
                      rng.gauss(0.0, 0.05))
        g1 = body.pos(spec["eye_u"] + 0.048 + rng.gauss(0, 0.008), -0.58)
        # Sampled as a quadratic through the bow instead of drawn as three
        # points: a 3-point `brush_path` is two straight segments, and two
        # straight segments meeting at the cheek is a chevron, not a gill.
        # The wander goes into the three control points, once, *before* the
        # curve is sampled. Jittering the eleven sampled points instead gave
        # each of them its own independent kick and the gill came out as a
        # zigzag — a hand wobbles over the length of a stroke, not between one
        # pixel and the next, and per-sample noise on a path is the one kind of
        # irregularity that looks less like a brush than a clean curve does.
        pts = []
        ng_ = max(11, int(L * 0.05))
        for i in range(ng_ + 1):
            t = i / ng_
            w0, w1, w2 = (1 - t) ** 2, 2 * t * (1 - t), t * t
            pts.append((g0[0] * w0 + gc[0] * w1 + g1[0] * w2,
                        g0[1] * w0 + gc[1] * w1 + g1[1] * w2))
        # 淡墨, and thin. At 焦墨 and six pixels wide on a big fish the gill
        # arrived as a hard black band across the shoulder — the one line in an
        # ink fish should be the strongest mark on the body and still a *line*.
        # Two pulls, not one line, and neither of them reaches an end.
        #
        # A single path of even width from the dorsal contour to the chin is a
        # vector arc, and on a fish whose body had just stopped looking like
        # one it was the most mechanical mark left on the sheet. A painter
        # starts this stroke in the air, presses through the cheek and lifts
        # before the belly, which puts the weight in the middle and nothing at
        # either tip; and the stroke usually breaks at the cheek, where the
        # hand changes direction. So: two overlapping pulls, each tapered at
        # its free end, the second drier than the first.
        gw = max(0.55, L * 0.0052)
        gh = int(ng_ * 0.62)
        _line(cv, pts[:gh + 1], gw * 0.22, gw,
              mix(nong, I("dan"), 0.18), (0.32 + 0.30 * k))
        _line(cv, pts[int(ng_ * 0.48):], gw * 0.92, gw * 0.18,
              mix(nong, I("dan"), 0.46), (0.24 + 0.26 * k))

    # 侧线 — the lateral line. One long shallow stroke from behind the gill to
    # the wrist, riding a little above the spine and sagging over the belly.
    #
    # This is the cheapest mark in the whole module and close to the most
    # valuable. Without it the flank is a smooth gradient and the fish reads as
    # an airbrushed balloon; with it the same gradient reads as a surface with
    # a seam running along it, and the 分墨 above and below the line becomes
    # back and belly instead of dark and light.
    # It is also drawn in two or three touches with gaps between them, and
    # never from end to end. One continuous hairline the whole length of the
    # fish is the same fault as the old gill: an even line is a drawn line,
    # and this one had the added problem of being the longest mark on the
    # sheet. 意到笔不到 — a couple of dashes in the right place and the eye
    # supplies the rest of the seam.
    if depth < 0.7:
        u0 = 0.22 + rng.uniform(-0.03, 0.05)
        for seg in range(rng.randint(2, 3)):
            ln = rng.uniform(0.14, 0.26)
            lp = []
            dv = rng.gauss(0.0, 0.05)
            for i in range(6):
                t = i / 5.0
                u = u0 + ln * t
                lp.append(body.pos(u, 0.30 - 0.26 * u * u + dv))
            _line(cv, lp, max(0.35, L * 0.0030),
                  max(0.28, L * 0.0014),
                  mix(rgb(pal["body"][1]), rgb(pal["ink"]), 0.25),
                  (0.10 + 0.13 * k) * rng.uniform(0.7, 1.2))
            u0 += ln + rng.uniform(0.05, 0.13)
            if u0 > 0.80:
                break

    # 鳞 — scales, as staggered rows of arcs opening toward the tail.
    # 意到笔不到: only a fraction are drawn. A fish with every scale on it is
    # a diagram, and in 国画 the convention is to suggest a patch of them over
    # the shoulder and let the eye finish the job.
    if spec["scales"] and depth < 0.45:
        rows = int(2 * spec["scales"]) + 2
        # Darker than the wash, not lighter. These were mixed 40% toward the
        # pale stop and painted `over`, which puts opaque near-white crescents
        # on a grey flank: at any size they read as chips of eggshell stuck to
        # the fish, and they were the one mark in the module that was lighter
        # than the paint around it for no reason a brush could produce. A scale
        # is suggested with a touch of the same ink, a shade deeper.
        pale = mix(rgb(pal["body"][1]), rgb(pal["ink"]), 0.30)
        # A *patch*, which means consecutive arcs in each row.
        #
        # Dropping each scale independently at 40 % broke every row it was
        # supposed to make, and what came out was a scatter of little hooks
        # over the whole flank — the back included, where there is nothing for
        # a pale arc to say — reading as specks of debris on the fish rather
        # than as 鳞. Rows of three to five touching scales over the shoulder,
        # and bare body everywhere else, is the convention and also the only
        # version the eye reads as a surface.
        # …and then every arc has to be a different arc. Laid on an exact grid
        # at one radius, one aspect and one span — which is what the first
        # patch was, the irregularity being only in *which* arcs got drawn —
        # the result is a lattice of identical crescents, and a lattice reads as
        # knitting rather than as 鳞 however correct each individual mark is.
        # Position, radius, aspect, angle and span all jitter, and the alpha
        # range is wide enough that some scales are barely there.
        for r_i in range(rows):
            v = -0.18 + 0.68 * r_i / (rows - 1.0) + rng.gauss(0.0, 0.05)
            off = 0.5 * (r_i % 2)
            c0 = rng.randint(0, 3)
            c1 = c0 + rng.randint(2, 4)
            for c_i in range(c0, c1):
                u = 0.26 + (c_i + off) * 0.050 + rng.gauss(0.0, 0.010)
                if u > 0.56:
                    continue
                px, py = body.pos(u, v + rng.gauss(0.0, 0.035))
                rr = L * 0.020 * rng.uniform(0.76, 1.18)
                # Bulging toward the *tail*: the free edge of a scale is its
                # back edge. Centred on heading + π they came out as a row of
                # commas all facing the wrong way, and a fish with reversed
                # scales looks like it has been brushed backwards.
                a0 = body.axis(u) + rng.gauss(0.0, 0.16)
                sp = rng.uniform(0.74, 1.18)
                ellipse_arc(cv, px, py, rr, rr * rng.uniform(0.72, 1.00), a0,
                            -sp, sp * rng.uniform(0.80, 1.00),
                            max(0.40, L * 0.0032),
                            max(0.30, L * 0.0022), pale,
                            rng.uniform(0.10, 0.30) * k)

    # 口 — the mouth. One short stroke hooked back from the snout along the
    # lower lip: it starts at the very tip and runs *backwards* down the
    # underside, which is the whole mouth in one mark. Drawn straight down
    # instead, it reads as a chin.
    mx, my = body.pos(0.004, -0.06)
    mdn = body.out_ang(0.02, -1.0)
    mbk = body.axis(0.0)
    mdl = math.atan2(math.sin(mbk - mdn), math.cos(mbk - mdn))
    ml = L * 0.050
    m1 = (mx + math.cos(mdn + mdl * 0.30) * ml,
          my + math.sin(mdn + mdl * 0.30) * ml)
    m2 = (m1[0] + math.cos(mdn + mdl * 0.80) * ml * 0.8,
          m1[1] + math.sin(mdn + mdl * 0.80) * ml * 0.8)
    # Sampled as a quadratic, for the same reason the gill is: three points
    # handed to `brush_path` are two straight segments, and two straight
    # segments meeting at the corner of the mouth put a hard angle on the
    # snout. Together with the gill's arc the head came out as a drawn
    # polygon — the one part of the fish still reading as an outline after the
    # body and fins had both become strokes.
    mp = []
    nm_ = max(7, int(L * 0.02))
    for i in range(nm_ + 1):
        t = i / nm_
        w0, w1, w2 = (1 - t) ** 2, 2 * t * (1 - t), t * t
        mp.append((mx * w0 + m1[0] * w1 + m2[0] * w2,
                   my * w0 + m1[1] * w1 + m2[1] * w2))
    _line(cv, mp, max(0.50, L * 0.0052), max(0.34, L * 0.0022), nong,
          0.26 + 0.28 * k)

    # 须 — barbels. 鲤鱼 has two, and they are what makes a carp a carp.
    # They hang from the corners of the mouth and *drift* — they start out
    # along the belly normal and curve back toward the tail, so the stroke is
    # one continuous droop. Aimed by a fixed angle off the heading instead,
    # they stuck out forward in two hard kinks and the carp grew a pair of
    # insect legs.
    if spec["barbel"] and depth < 0.6:
        down = body.out_ang(0.0, -1.0)
        back = body.axis(0.0)
        dlt = math.atan2(math.sin(back - down), math.cos(back - down))
        for s_i in (1, -1):
            # Hung off the mouth corner, not the snout tip. At the tip they
            # crossed the mouth stroke and the three marks together read as a
            # scribble under the chin rather than as two whiskers.
            bxx, byy = body.pos(0.050 + 0.028 * (s_i > 0), -0.55)
            ln = L * rng.uniform(0.13, 0.21) * (1.0 if s_i > 0 else 0.74)
            a0 = down + dlt * rng.uniform(0.18, 0.34)
            sweep = dlt * rng.uniform(0.80, 1.25)
            # Walked, not fanned. The old version placed each point at
            # `root + direction(t) × length × t`, which is a radial fan: every
            # sample lies on a straight line out of the root, so however the
            # angle is swept the result is a nearly straight spoke with a kink
            # in it, and a carp came out with two insect legs. Stepping along
            # the curve and turning a little at each step is an arc-length
            # integration and it actually curls — which is the whole character
            # of a barbel, a line with no straight part anywhere in it.
            nb_ = 9
            bxc, byc, aa = bxx, byy, a0
            pts = [(bxc, byc)]
            st = ln / nb_
            for j in range(nb_):
                t = (j + 0.5) / nb_
                aa += sweep * (0.6 + 1.4 * t) / nb_
                bxc += math.cos(aa) * st
                byc += math.sin(aa) * st
                pts.append((bxc, byc))
            _line(cv, pts, max(0.40, L * 0.0034),
                  max(0.22, L * 0.0010),
                  mix(nong, I("dan"), 0.48), (0.18 + 0.20 * k))

    # 点睛 — the eye, last, and the most important mark on the sheet.
    #
    # Three parts: the socket, a pale disc; the pupil, 焦墨, pushed *forward*
    # and *up* inside it; and one speck of 白粉. The offset is the whole
    # trick. A pupil in the middle of the socket is an ornament and the fish
    # is asleep; the same pupil shoved against the front of the socket is
    # looking where it is going, and 八大山人 built a career on pushing it
    # further than anyone thought reasonable.
    # eye_v is about a third of the way down from the back, not half: at 0.46
    # of a head that is already narrow the socket sat on the dorsal contour and
    # half of it hung outside the silhouette, which reads as a fish with its
    # eye stuck on rather than in.
    ex, ey = body.pos(spec["eye_u"], spec["eye_v"])
    er = L * spec["eye"] * 0.5
    ea = body.face()
    eb = body.axis(spec["eye_u"])
    # The socket is the *paper*, barely tinted, and the ring around it is a
    # **broken** arc rather than an outline. Two faults were in here and both
    # of them were the same fault: a disc with an even rim and a bead inside it
    # is a cartoon eye, and so is a bead with a white glint on it. The glint is
    # gone — it was the single most mechanical mark in the module, a specular
    # highlight being a statement about a light source and a glossy surface,
    # neither of which exists in a painting with no water in it — and the ring
    # is now drawn as a stroke that goes round about five sixths of the way and
    # lifts, which is what one turn of a brush leaves.
    # 眼眶 — the socket, and then the ring, and neither of them is a circle.
    #
    # What was here was a pale disc, a full elliptical outline at even weight
    # and a dark bead in the middle of it, and that is a cartoon eye in three
    # independent ways. A disc of near-white inside a coloured body is an
    # eyeball with a sclera, which no ink fish has: the socket is *paper*, so
    # it is three overlapping dabs at paper tone rather than one circle of
    # white, and its outline is the union of them. An outline of constant
    # weight is a drawn circle however broken it is, so the ring is now a
    # sampled path whose radius and width both wander and which lifts a third
    # of the way round — one turn of a brush, pressed hardest where the hand
    # was coming down and gone where it left.
    eb_r = er * spec["bulge"]
    ra = rng.uniform(0, TAU)
    sweep = TAU * rng.uniform(0.62, 0.76)
    rp = []
    nr = max(15, int(eb_r * 0.9))
    for i in range(nr):
        t = i / (nr - 1.0)
        aa = ra + sweep * t
        rk = eb_r * (0.94 + 0.13 * math.sin(ra * 2.3 + 4.1 * t))
        rp.append((ex + math.cos(aa) * rk,
                   ey + math.sin(aa) * rk * rng.uniform(0.94, 1.02)))
    ew = max(0.48, eb_r * 0.13)
    nh = nr // 2
    _line(cv, rp[:nh + 1], ew * 0.40, ew, nong, (0.42 + 0.26 * k))
    _line(cv, rp[nh:], ew, ew * 0.22, mix(nong, I("dan"), 0.30),
          (0.34 + 0.24 * k))
    # 睛 — one pressed 点, off-centre, built from two overlapping dabs at a
    # shared angle so its outline is not a circle. The offset is the whole
    # trick: a pupil in the middle of its socket is an ornament and the fish is
    # asleep, while the same pupil shoved against the front is looking where it
    # is going, and 八大山人 built a career on pushing it further than anyone
    # thought reasonable.
    # And the 点 nearly fills it. A small bead centred in a pale socket is
    # an eyeball with a sclera round it, which is a cartoon; on an ink fish
    # the pupil is a pressed dot almost as wide as the ring it sits in, and
    # the only paper left inside is a crescent at the *back* of the socket,
    # because the dot is shoved forward. That crescent is the whole of the
    # expression, and the amount of it is the difference between a fish that
    # is looking at something and a fish with a toy eye glued on.
    # Sized off the *ring*, not off `er`. 金鱼 and 鳜鱼 carry a `bulge` of a
    # third again, so a pupil scaled from the bare eye radius sat at barely
    # half the ring's width and left a wide bright crescent all the way round
    # it: a black bead in a pale socket, which is the eye a cartoon fish has.
    # The paper that is left should be a sliver at the back of the ring and
    # nothing anywhere else.
    gx = ex + math.cos(ea) * eb_r * 0.22
    gy = ey + math.sin(ea) * eb_r * 0.22
    pa = ea + rng.uniform(1.5, 2.3)
    for dd, rk, ak in ((0.0, 0.76, 1.00), (0.20, 0.62, 0.80)):
        disc(cv, gx + math.cos(pa) * eb_r * dd, gy + math.sin(pa) * eb_r * dd,
             eb_r * rk, I("jiao"), (0.86 + 0.12 * k) * ak,
             squash=rng.uniform(0.74, 0.94), soft=0.24)

    if occ is not None:
        # Three discs down the spine, not one big one. A fish is nothing like
        # a circle, and the tail is the half of the mark that gets into
        # trouble: with a single body-centred disc the caudal fin — which for
        # 金鱼 is two thirds of the body length again — was outside the
        # occupancy map entirely, and tails kept being painted straight
        # through the 题款 column. Whatever is *drawn* has to be declared.
        for u, r in ((0.22, 0.40), (0.62, 0.40)):
            px, py = body.pos(u, 0.0)
            occ.add(px, py, L * r)
        tl = L * spec["tail"]["len"]
        wx, wy = body.pos(1.0, 0.0)
        ta = body.axis(1.0)
        occ.add(wx + math.cos(ta) * tl * 0.55,
                wy + math.sin(ta) * tl * 0.55, tl * 0.66)
    return body


# --------------------------------------------------------------------------
# 水族 — the companions. 虾 and 蝌蚪, the two other creatures a fish painter
# is expected to be able to write without thinking about it.
# --------------------------------------------------------------------------

def paint_shrimp(cv, rng, pal, x, y, s, ang, depth=0.3):
    """一只虾 — 齐白石's shrimp, which is six arched segments, two claws and a
    cloud of whiskers.

    It is almost all 淡墨 and almost all transparent; the only dark marks are
    the eyes and the tips of the claws. The segments are drawn as overlapping
    arcs rather than one body, because that overlap is what gives a shrimp its
    glassy, jointed look — each `mul` pass darkens where two segments meet,
    which is exactly where a real shrimp's shell doubles.
    """
    k = 1.0 - depth
    ca, sa = math.cos(ang), math.sin(ang)

    def pt(f, o=0.0):
        return x + ca * f - sa * o, y + sa * f + ca * o

    lt, dp = pal["fin"]
    dan = mix(I("dan"), rgb(pal["ink"]), 0.25)
    bend = rng.uniform(0.10, 0.22) * rng.choice((-1.0, 1.0))

    # 头胸 — the carapace, one wedge, blunt forward.
    Blade(cx=pt(s * 0.30)[0], cy=pt(s * 0.30)[1], ang=ang + math.pi,
          L=s * 0.34, W=s * 0.115,
          k1=2.2, e1=0.50, k2=2.6, e2=0.44, tip=0.0,
          bend=bend * 0.4, light=rgb(lt), deep=rgb(dp), line=0.0,
          w_base=0.34, p_base=1.6, w_rim=0.30, p_rim=2.4, hl=0.16,
          soft=0.8, alpha=(0.34 + 0.26 * k), blend="mul", wob=0.5).render(cv)

    # 腹 — six segments, each shorter and narrower, curling.
    #
    # They have to overlap by more than half. Advanced by a full segment each
    # time, the six came out as a string of separate pale beads — a caterpillar
    # — because each one closes its own blunt end before the next begins. A
    # shrimp's abdomen is one tapering curved mass with joint lines *in* it, so
    # each stroke is twice as long as its step and the `mul` pools the joints
    # for free.
    px, py, a = pt(s * 0.00)[0], pt(s * 0.00)[1], ang + math.pi
    for i in range(6):
        ln = s * (0.115 - 0.009 * i)
        w = s * (0.105 - 0.010 * i)
        Blade(cx=px, cy=py, ang=a, L=ln * 2.2, W=w,
              k1=2.0, e1=0.52, k2=2.2, e2=0.50, tip=0.0,
              light=rgb(lt), deep=rgb(dp), line=0.0,
              w_base=0.30, p_base=1.6, w_rim=0.34, p_rim=2.2, hl=0.14,
              soft=0.7, alpha=(0.26 + 0.22 * k), blend="mul",
              wob=0.5).render(cv)
        a += bend * 0.9
        px += math.cos(a) * ln * 0.80
        py += math.sin(a) * ln * 0.80

    # 尾 — three flicks.
    for f in (-0.42, 0.0, 0.42):
        aa = a + f
        ln = s * rng.uniform(0.13, 0.19)
        brush_seg(cv, px, py, px + math.cos(aa) * ln, py + math.sin(aa) * ln,
                  max(0.5, s * 0.022), max(0.35, s * 0.006), dan,
                  (0.30 + 0.26 * k))

    # 钳 — the two long claws, reaching forward and down.
    #
    # Thin, close in, and bent *outward*. Drawn at a quarter of the body's
    # width, splayed half a radian each way and folded back across the
    # midline, the two of them plus six whiskers added up to eight straight
    # lines radiating from a small pale lump with a hard dark trapezoid in the
    # middle where the forearms crossed — the whole thing read as a mosquito.
    # A shrimp's mass is its body: every appendage has to be a hair by
    # comparison, and nothing may cross anything else.
    for side in (1, -1):
        a1 = ang + side * rng.uniform(0.18, 0.34)
        bx, by = pt(s * 0.24, side * s * 0.06)
        ln = s * rng.uniform(0.24, 0.34)
        kx, ky = bx + math.cos(a1) * ln, by + math.sin(a1) * ln
        a2 = a1 + side * rng.uniform(0.40, 0.70)
        ex, ey = kx + math.cos(a2) * ln * 0.66, ky + math.sin(a2) * ln * 0.66
        brush_path(cv, [(bx, by), (kx, ky)], max(0.45, s * 0.016),
                   max(0.34, s * 0.010), dan, 0.30 + 0.26 * k)
        brush_path(cv, [(kx, ky), (ex, ey)], max(0.38, s * 0.011),
                   max(0.28, s * 0.006), dan, 0.32 + 0.26 * k)
        for f in (0.20, -0.20):                    # the pincer
            brush_seg(cv, ex, ey,
                      ex + math.cos(a2 + f) * s * 0.06,
                      ey + math.sin(a2 + f) * s * 0.06,
                      max(0.36, s * 0.008), max(0.26, s * 0.004),
                      mix(dan, I("nong"), 0.5), 0.42 + 0.3 * k)

    # 须 — the whiskers. Long, fine, and the reason a painted shrimp looks
    # alive; they must be longer than the shrimp.
    # Two long and two short, all finer than the claws, all curving the
    # *same* way. A radial splay of equal-length hairs is legs however thin
    # they are; what makes 须 read as whiskers is that they are all being
    # dragged by the same water, so the sweep sign is drawn once for the
    # creature and not once per hair.
    sw = rng.choice((-1.0, 1.0))
    for i in range(rng.randint(3, 4)):
        a1 = ang + rng.uniform(-0.40, 0.40)
        bx, by = pt(s * 0.30, rng.uniform(-0.05, 0.05) * s)
        ln = s * (rng.uniform(0.90, 1.25) if i < 2
                  else rng.uniform(0.30, 0.52))
        sweep = sw * rng.uniform(0.55, 1.05)
        pts = [(bx, by)]
        for j in (1, 2, 3, 4):
            t = j / 4.0
            aa = a1 + sweep * t * t
            pts.append((bx + math.cos(aa) * ln * t, by + math.sin(aa) * ln * t))
        brush_path(cv, pts, max(0.4, s * 0.007), max(0.22, s * 0.003),
                   dan, (0.11 + 0.15 * k) * rng.uniform(0.7, 1.0))

    # 眼 — two short dark dashes across the head, not two beads. 齐白石's
    # shrimp eyes are a pair of commas laid at right angles to the body, and
    # they are the only 浓墨 in the creature; as round dots they sat inside
    # the arms and read as a face.
    for side in (1, -1):
        ex, ey = pt(s * 0.30, side * s * 0.050)
        aa = ang + math.pi * 0.5
        brush_seg(cv, ex - math.cos(aa) * s * 0.022,
                  ey - math.sin(aa) * s * 0.022,
                  ex + math.cos(aa) * s * 0.022,
                  ey + math.sin(aa) * s * 0.022,
                  max(0.6, s * 0.022), max(0.4, s * 0.010),
                  I("jiao"), 0.68 + 0.22 * k)


def paint_tadpole(cv, rng, pal, x, y, s, ang):
    """蝌蚪 — a dot and a wriggle. Three brush marks in total, and a sheet of
    fish is immediately a pond rather than an aquarium."""
    ink = mix(I("nong"), rgb(pal["ink"]), 0.3)
    disc(cv, x, y, s * 0.42, ink, rng.uniform(0.55, 0.80), squash=0.78,
         soft=0.3)
    a = ang + math.pi
    pts = [(x, y)]
    sw = rng.uniform(-1.1, 1.1)
    for j in (1, 2, 3):
        t = j / 3.0
        aa = a + sw * t * t
        pts.append((x + math.cos(aa) * s * 1.9 * t,
                    y + math.sin(aa) * s * 1.9 * t))
    brush_path(cv, pts, max(0.45, s * 0.30), max(0.3, s * 0.07), ink,
               rng.uniform(0.40, 0.65))


def paint_bubbles(cv, rng, x, y, s, ang, n):
    """气泡 — a trail of rings off a mouth. Drawn as rings and not as discs:
    a bubble is a film, and a filled dot in the middle of blank paper reads as
    a speck of dirt."""
    a = ang
    for i in range(n):
        d = s * (0.5 + i * rng.uniform(0.7, 1.1))
        a += rng.gauss(0.0, 0.28)
        bx = x + math.cos(a) * d
        by = y + math.sin(a) * d - d * 0.35
        # Small and faint. Rings a fifth of the mouth's width at half opacity
        # read as capital O's typed onto the paper; a bubble is barely there.
        r = s * rng.uniform(0.07, 0.15) * (1.0 + i * 0.16)
        ellipse_arc(cv, bx, by, r, r, 0.0, 0.0, TAU,
                    max(0.40, r * 0.18), max(0.40, r * 0.18),
                    I("qing"), rng.uniform(0.16, 0.30))


# --------------------------------------------------------------------------
# 水草 荷叶 浮萍 — the only furniture the water is allowed
# --------------------------------------------------------------------------

def water_weed(cv, rng, pal, x, yb, h, n, alpha, flow):
    """水草 — a clump of ribbon weed rising from below the lower border.

    Rooted *off* the sheet. Weed that starts inside the picture has a visible
    bottom end, and a visible bottom end is a riverbed — which puts a floor
    under a painting whose whole subject is that there is no floor.
    """
    # Bent hard, and bent *together* — but not by the same amount.
    #
    # Straight blades at 0.3 radians of scatter came out as a stand of grass
    # across the bottom of the sheet, which put a lawn under the fish, and a
    # lawn is a floor. Water weed leans downstream with the same current the
    # fish are swimming in, and it curves along its whole length because
    # nothing is holding it up.
    #
    # Giving every blade the same bend fixed the lawn and bought a second
    # fault: a bundle of identical arcs, evenly spaced, reading as fibre-optic
    # cable. Shared *direction* is the composition; shared *curvature* is the
    # mistake. So the bend spans a factor of four, the lengths a factor of
    # three, the roots are clustered rather than spaced, and one blade in five
    # curls back against the current — which is what makes the clump cross
    # itself, and a clump that never crosses itself is a comb.
    lt, dp = pal["weed"]
    sgn = 1.0 if (flow or 1.0) > 0 else -1.0
    for i in range(n):
        dx = rng.gauss(0.0, h * 0.14) + (i - (n - 1) * 0.5) * h * 0.03
        L = h * rng.uniform(0.30, 1.00)
        a = -math.pi * 0.5 + flow * rng.uniform(0.3, 1.3) + rng.gauss(0, 0.20)
        against = rng.random() < 0.20
        bend = (-sgn * rng.uniform(0.10, 0.30) if against
                else sgn * rng.uniform(0.22, 0.95))
        Blade(cx=x + dx, cy=yb, ang=a, L=L,
              W=L * rng.uniform(0.012, 0.034),
              k1=1.3, e1=0.55, k2=1.9, e2=0.85, tip=0.30,
              bend=bend,
              light=rgb(lt), deep=rgb(dp), line=0.0,
              w_base=0.50, p_base=1.3, w_rim=0.20, p_rim=2.6,
              soft=1.1, alpha=alpha * rng.uniform(0.45, 0.90), blend="mul",
              wob=0.8).render(cv)


def duckweed(cv, rng, pal, W, H, n, band=0.22, occ=None):
    """浮萍 — duckweed, in clumps near the top.

    Floating, so it belongs to the surface, so it goes near the upper border
    and nowhere else. It is also the only thing in the painting that tells you
    which way is up, which is why even 独游 usually gets a few.

    In *clumps*, and with the root thread almost always left off. Drawn as
    single evenly-scattered discs each trailing a two-radius hair, they came
    out as a row of pins stuck in the paper — which is what a drop-shaped mark
    with one straight line hanging off it looks like, however small it is.
    浮萍 mats: three or four leaves touching, and the rest of the sheet bare.
    """
    lt, dp = pal["weed"]
    S = min(W, H)
    for _ in range(n):
        cx = rng.uniform(W * 0.04, W * 0.96)
        cy = rng.uniform(H * 0.01, H * band)
        # Small and faint. At twice this size and half again this opacity the
        # clumps stopped reading as 浮萍 at all and read as splashes of mud on
        # the top of the sheet — the surface of the water is the one place in
        # the painting where a mark has nothing to be measured against, so it
        # has to be the lightest mark on it.
        r0 = S * rng.uniform(0.005, 0.011)
        al = rng.uniform(0.20, 0.42)
        # It claims its ground, so the 题款 column finds somewhere else to be.
        # Drawn before the inscription and invisible to it, a clump landed in
        # the title's first two characters often enough to matter.
        if occ is not None:
            w = r0 * 2.6
            if occ.overlap(cx - w, cy - w, cx + w, cy + w, 0.0) != 0.0:
                continue
            occ.add(cx, cy, w)
        for j in range(rng.randint(2, 4)):
            r = r0 * rng.uniform(0.70, 1.15)
            x = cx + rng.gauss(0.0, r0 * 1.4)
            y = cy + rng.gauss(0.0, r0 * 0.9)
            c = mix(rgb(lt), rgb(dp), rng.uniform(0.25, 0.75))
            disc(cv, x, y, r, c, al * rng.uniform(0.8, 1.0),
                 squash=rng.uniform(0.50, 0.72), soft=0.45)
            if j == 0 and rng.random() < 0.22:        # the root thread
                brush_seg(cv, x, y + r * 0.5, x + rng.uniform(-r, r) * 0.6,
                          y + r * rng.uniform(1.0, 1.8),
                          max(0.4, r * 0.09), max(0.3, r * 0.04),
                          mix(rgb(dp), I("qing"), 0.55),
                          rng.uniform(0.10, 0.20))


def lotus_pad(cv, rng, pal, x, y, r, alpha):
    """荷叶 — one lotus pad seen almost edge-on from below, which is how it
    looks to a fish and how 国画 almost always places it: a dark mass hanging
    into the top of the sheet with the light coming through it.

    Built from a ring of overlapping squashed dabs rather than one disc. A
    single flat ellipse with veins radiating right out to its rim is a
    satellite dish — the giveaway is the perfectly smooth outline, since a
    lotus pad's edge is always cupped and torn. The dabs give it a ragged
    silhouette for free, the `mul` blend pools ink where they overlap, and the
    veins are kept to the inner half so they never reach that edge.
    """
    lt, dp = pal["lotus"]
    rot = rng.uniform(-0.28, 0.28)
    c, s = math.cos(rot), math.sin(rot)
    sq = rng.uniform(0.30, 0.46)
    body = mix(rgb(lt), rgb(dp), 0.52)

    def lp(f, o):
        """a point `f` across the pad and `o` up it, in pad-local units"""
        px, py = f * r, o * r * sq
        return x + px * c - py * s, y + px * s + py * c

    # The ring has to be *uneven* — in radius, in dab size and in squash.
    # Nine dabs of one size on one circle is a smooth ellipse again however
    # many of them there are, and a smooth ellipse with spokes in it is the
    # satellite dish this function exists to avoid. A pad's edge is cupped,
    # torn and darker than its middle, so the dabs carry more ink than the
    # centre and the centre is mostly left as light coming through the leaf.
    n = rng.randint(8, 11)
    a0 = rng.uniform(0, TAU)
    rim = mix(rgb(lt), rgb(dp), 0.64)
    for i in range(n):
        a = a0 + TAU * (i + rng.gauss(0.0, 0.18)) / n
        d = rng.uniform(0.34, 0.72)
        dx, dy = lp(math.cos(a) * d, math.sin(a) * d)
        disc(cv, dx, dy, r * rng.uniform(0.34, 0.60), rim,
             alpha * rng.uniform(0.45, 0.85),
             squash=sq * rng.uniform(0.95, 1.6),
             soft=rng.uniform(0.5, 1.4))
    disc(cv, x, y, r * 0.54, body, alpha * 0.52, squash=sq * 1.3, soft=0.9)

    # 叶脉 — short, uneven, and not from the centre.
    #
    # Eight equal spokes meeting at one point is the other half of the
    # satellite dish, and it survives any amount of work on the outline: the
    # eye reads a perfect hub as machinery. A real pad's veins start at the
    # stalk's attachment, which is a patch and not a point, they are spaced
    # unevenly, and about a third of them are too faint to follow.
    n_v = rng.randint(6, 9)
    for i in range(n_v):
        if rng.random() < 0.22:
            continue
        a = rot + math.pi * (0.08 + 0.84 * i / (n_v - 1.0)) \
            + rng.gauss(0, 0.14)
        r0 = r * rng.uniform(0.06, 0.20)
        ln = r * rng.uniform(0.26, 0.66)
        bx = x + math.cos(a) * r0
        by = y + math.sin(a) * r0 * sq * 1.6
        # Bowed, not ruled. A vein follows the fold of the leaf, and a set of
        # dead straight spokes is the last piece of the machinery — it survives
        # every amount of work on the outline, because what the eye calls a
        # wheel is the straightness and not the hub.
        bow = rng.gauss(0.0, 0.16)
        pv = []
        for j in range(5):
            t = j / 4.0
            aa = a + bow * t * t
            pv.append((bx + math.cos(aa) * ln * t,
                       by + math.sin(aa) * ln * t * sq * 1.6))
        brush_path(cv, pv, max(0.45, r * 0.010), max(0.3, r * 0.004),
                   rgb(dp), alpha * rng.uniform(0.12, 0.28))

    # 梗 — the stalk, going up out of the sheet. It is what says the pad is
    # floating on a surface above the fish rather than lying on a floor.
    if rng.random() < 0.7:
        brush_path(cv, [(x, y), (x + rng.gauss(0, r * 0.10), y - r * 0.5),
                        (x + rng.gauss(0, r * 0.18), y - r * 1.1)],
                   max(0.5, r * 0.022), max(0.4, r * 0.014),
                   rgb(dp), alpha * 0.7)


# 涟漪 — there is deliberately no ripple function.
#
# There was one: two or three long shallow arcs in 清墨 near the top of the
# sheet, rationed hard on the grounds that a breath of wind is not the same as
# painting water. On paper they were wires. A near-flat arc 40 % of the sheet
# wide has no curvature the eye can read as a surface, so it reads as what it
# geometrically is — a long thin line drawn across a finished painting — and
# one of them went straight through a lotus pad.
#
# The fix was not to tune it. 不画水 is the first line of this module's
# docstring, and the function existed only because a blank sheet felt
# under-furnished. It wasn't: that blank sheet is the water.


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
#
# 濠梁观鱼: Zhuangzi and Huizi on the bridge over the Hao, where Zhuangzi says
# the fish are happy and Huizi asks how he could possibly know. Two thousand
# years later it is still the thing you write on a painting of fish, and
# 知鱼 / 濠上 / 鱼乐 are all that argument in two characters.
# --------------------------------------------------------------------------

TITLE_A = ["春水", "游鱼", "清波", "鱼乐", "濠梁", "荷塘", "柳荫", "碧波",
           "秋水", "活水", "寒潭", "桃花", "藻荇", "墨戏", "一池", "知鱼"]
TITLE_B = ["图", "小景", "清趣", "写意", "册页", "小品", "之乐", "图卷"]
ACTION = ["写生", "戏作", "写意", "拟古", "池上写生", "客窗戏墨", "并记"]
SEAL_WORDS = ["知鱼", "濠上", "鱼乐", "清波", "墨缘", "闲心", "写心", "自在",
              "游心", "活水", "一勺"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into fish_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [v["cn"] for v in SCENES.values()]
             + [v["cn"] for v in SPECIES.values()]
             + [v["season"] for v in SPECIES.values()]
             + [v["cn"] for v in PALETTES.values()]
             + [TIANGAN, DIZHI, "日月年写于图春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a blob baked for this vocabulary; fall back to the other three
    painters' (they share most characters), then to no 题款 at all."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("fish_font", "gongbi_font", "bamboo_font", "shanshui_font"):
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


def inscribe(cv, rng, occ, S, title):
    """题款 — measure the block, then let the 留白 decide where it goes."""
    W, H = cv.w, cv.h
    gs = S * rng.uniform(0.042, 0.058)
    ss = gs * rng.uniform(0.50, 0.64)
    lead, lead_s = rng.uniform(1.12, 1.26), rng.uniform(1.18, 1.34)
    season = "春夏秋冬"[rng.randrange(4)]
    sign = (TIANGAN[rng.randrange(10)] + DIZHI[rng.randrange(12)]
            + season + "日" + ACTION[rng.randrange(len(ACTION))])
    gap = gs * 0.45
    sign_dy = gs * rng.uniform(1.0, 2.0)
    seal_s = gs * rng.uniform(0.80, 1.0)
    bw = gs + gap + ss
    bh = max(len(title) * gs * lead,
             sign_dy + len(sign) * ss * lead_s + gs * 0.25 + seal_s)
    margin = S * 0.040
    if FONT and bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.5)
        paint_column(cv, rng, FONT, title, bx + gap + ss, by, gs, lead=lead)
        end = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, ss,
                           lead=lead_s, alpha=(0.60, 0.84))
        paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                   bx + (ss - seal_s) * 0.5, end + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
    if FONT and rng.random() < 0.5:                # 压角章
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
    """Squeeze a value toward the middle of [lo, hi] — the soft clamp from
    bamboo.py. Used on headings, where a hard clamp piles outliers up exactly
    on the limit and makes the thing it was meant to prevent obvious."""
    m, r = (lo + hi) * 0.5, (hi - lo) * 0.5
    if r <= 0.0:
        return m
    t = (p - m) / r
    return m + r * math.tanh(t * k / 2.0) if abs(t) > 1.0 else p


def _collinear(x, y, sites, base, tol=0.20):
    """Is (x, y) strung on a line through two already-placed fish?

    One Gaussian cloud of five or six points lands on a near-perfect diagonal
    far more often than the word "random" suggests, and a diagonal of fish at
    even gaps is exactly the printed-textile failure the scatter replaced the
    blue noise to avoid — worse, actually, because a line also reads as a
    *direction*, and it is never the direction the fish are swimming.

    Only near pairs are tested. Three fish spread across a whole sheet do not
    read as a line; three in one knot do.
    """
    span = base * 2.6
    m = len(sites)
    for i in range(m):
        ax, ay = sites[i][0], sites[i][1]
        for j in range(i + 1, m):
            bx, by = sites[j][0], sites[j][1]
            ux, uy = bx - ax, by - ay
            d = math.hypot(ux, uy)
            if d < 1e-6 or d > span * 2.0:
                continue
            if (math.hypot(x - ax, y - ay) > span
                    and math.hypot(x - bx, y - by) > span):
                continue
            if abs((x - ax) * uy - (y - ay) * ux) < tol * base * d:
                return True
    return False


def fish_sites(rng, W, H, S, n, flow, spread, back, lo, hi, base, tl):
    """Where the fish go, how big each is, and which way it swims.

    Three rules, all of them 忌 — things a fish painting may not do:

    * **忌平行.** Fish all on one heading is a printed textile. So each gets
      `flow + gauss(0, spread)`, soft-clamped to ±0.8 rad, which is variety
      inside a shared current. 势 before individuality: the bamboo painter
      learned this the hard way, when one culm at a wild angle wrecked every
      sheet it appeared on.
    * **忌对头.** Two fish nose to nose is a fight, not a school. This one
      comes out *free* from the rule above: if every heading is within 0.8 rad
      of the current, no two can be more than 1.6 rad apart, and nose-to-nose
      needs nearly π. The exception is the 回首 fish below, which is the one
      case that has to be checked.
    * **忌等距同大.** Equal spacing and equal size.

      The first version farthest-point sampled the positions, reasoning that
      blue noise is not a grid. That is true and it is still 等距: blue noise
      is *the* maximally even scatter, so six fish came out spread across the
      sheet at near-identical gaps, all the same size, all on the same
      heading. It read as printed textile.

      So the positions are drawn from a Gaussian around a single 鱼群 centre
      instead, with only a hard minimum separation enforced. That gives
      genuinely uneven gaps — a tight knot of two or three and a straggler —
      which is 疏密. And the sizes ramp over a wide range rather than a narrow
      one, so there is one unmistakable 主鱼 and the rest are subordinate.
      大小相间 is about *ratio*, and 0.70–1.00 is not a ratio anyone can see.
    * **忌出界.** A fish may not be cut by the edge of the paper.

      The margin cannot be a constant, because what has to stay on the sheet
      is not the site: it is the whole mark, from the snout forward of the
      site to the tip of a caudal fin that for a 金鱼 trails three quarters of
      a body length behind it. With a flat margin the biggest fish in the
      school kept losing its tail off the left border, and a fish with its
      tail cropped is not a composition with a fish in it, it is a mistake.
      So the heading is chosen *before* the position, and both ends of the
      swept extent are tested.

    `base` is the body length of a size-1.0 fish and `tl` the tail length as a
    fraction of it, which is all this needs to know about the species.
    """
    pad = S * 0.035
    cx0 = rng.uniform(W * 0.32, W * 0.68)
    cy0 = rng.uniform(H * 0.34, H * 0.60)
    # The cluster's spread. At 0.34 S the "school" covered most of the sheet
    # and came out as a diagram of fish positions — the gaps between the fish
    # were bigger than the fish. 鱼群 means a knot of two or three close
    # enough to overlap tails, with the rest hung off it; that needs a sigma
    # of about one body length, not three.
    sig = S * (0.22 if n > 3 else 0.17)
    sites = []
    # 回首 — one fish turning back against the current. A classic device, and
    # the only licensed exception to the shared 势; it goes in first so that
    # it gets the room it needs, and so the others are placed clear of it.
    rev = back > 0.0 and rng.random() < back
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1.0)
        size = hi + (lo - hi) * t ** 0.8
        L = base * size
        # the heading first, because the room a fish needs depends on it
        if rev and i == n - 1:
            a = flow + math.pi + rng.gauss(0.0, 0.22)
        else:
            a = _hold(flow + rng.gauss(0.0, spread), flow - 0.8, flow + 0.8)
        ca, sa = math.cos(a), math.sin(a)
        # the snout is half a body ahead of the site, the tail tip is half a
        # body plus the fin behind it, and both have to clear the border
        fwd, bwd = L * 0.58, L * (0.50 + tl)
        best = None
        for try_i in range(60):
            if i == 0 and try_i == 0:
                # the 主鱼 gets the centre if the centre fits, and a nudge
                # off it if it does not
                x, y = cx0, cy0
            else:
                x = cx0 + rng.gauss(0.0, sig)
                y = cy0 + rng.gauss(0.0, sig * 0.86)
            ok = True
            for d in (fwd, -bwd):
                px, py = x + ca * d, y + sa * d
                if not (pad < px < W - pad and pad < py < H - pad):
                    ok = False
                    break
            if not ok:
                continue
            # Clear of its neighbours by the mean of the two body lengths.
            # A flat separation in sheet units let the two biggest fish of a
            # school overlap at the belly while keeping the small ones far
            # apart — the gap has to be measured in fish, not in paper.
            if not all(math.hypot(x - sx, y - sy)
                       > 0.52 * base * (size + ss)
                       for sx, sy, ss, _sa in sites):
                continue
            # 忌三点一线. Given up on after two thirds of the tries, because a
            # fish placed somewhere is worth more than a rule obeyed.
            if try_i < 40 and _collinear(x, y, sites, base):
                continue
            best = (x, y)
            break
        if best is None:
            continue
        sites.append((best[0], best[1], size, a))

    # the 回首 fish is the one case where 忌对头 can actually happen, so it is
    # the one case that is checked: if it is pointing at a neighbour that is
    # pointing back, swing it aside rather than reversing it again.
    if rev and len(sites) > 1:
        hx, hy, hs, ha = sites[-1]
        for ox, oy, _os, oa in sites[:-1]:
            to = math.atan2(oy - hy, ox - hx)
            if (abs(math.atan2(math.sin(to - ha), math.cos(to - ha))) < 0.55
                    and math.hypot(ox - hx, oy - hy) < S * 0.42):
                ha += rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.1)
                sites[-1] = (hx, hy, hs, ha)
                break
    return sites


def compose(cv, rng, scene_key, species_key, pal_key):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc = SCENES[scene_key]
    spec = SPECIES[species_key]
    pal = PALETTES[pal_key]

    ground = gb.paint_ground(cv, rng)
    gb.set_light(rng)
    occ = Occupancy(W, H)

    # 水势 — the current. One direction for the whole sheet, roughly
    # horizontal: fish swim across a painting, not up and down it.
    flow = rng.choice((0.0, math.pi)) + rng.gauss(0.0, 0.30)

    # --- 水草 first: it is behind everything ---------------------------
    if rng.random() < sc["weed"]:
        for _ in range(rng.randint(1, 2 + int(sc["weed"] * 2))):
            water_weed(cv, rng, pal, rng.uniform(W * 0.05, W * 0.95),
                       H + S * 0.02, S * rng.uniform(0.28, 0.62),
                       rng.randint(3, 7), rng.uniform(0.30, 0.55),
                       flow * 0.10 + rng.gauss(0, 0.12))

    # --- 荷叶 hanging into the top ------------------------------------
    if rng.random() < sc["lotus"]:
        for _ in range(rng.randint(1, 2)):
            lx = rng.uniform(W * 0.12, W * 0.88)
            ly = rng.uniform(-S * 0.03, S * 0.10)
            lr = S * rng.uniform(0.16, 0.28)
            lotus_pad(cv, rng, pal, lx, ly, lr, rng.uniform(0.40, 0.62))
            occ.add(lx, ly, lr * 0.8)

    # --- 鱼 — far to near, so the near ones overlap the far ones -------
    n = rng.randint(*sc["n"])
    base = S * rng.uniform(0.23, 0.29) * (1.30 if n <= 2 else 1.0)
    sites = fish_sites(rng, W, H, S, n, flow, sc["spread"], sc["back"],
                       sc["size"][0], sc["size"][1], base,
                       spec["tail"]["len"])
    # 分墨 — depth is assigned by size, not at random: a fish painted small
    # and dark reads as a small fish, while small *and* pale reads as a fish
    # further away, which is the one that makes the sheet deep.
    sizes = sorted((s[2] for s in sites), reverse=True)
    rank = {v: i for i, v in enumerate(sizes)}
    bodies = []
    for sx, sy, ss, sa in sorted(sites, key=lambda s: -s[2]):
        depth = 0.0 if n == 1 else rank[ss] / (n - 1.0) * rng.uniform(0.7, 1.0)
        bodies.append((depth, sx, sy, ss, sa))
    for depth, sx, sy, ss, sa in sorted(bodies, key=lambda b: -b[0]):
        L = base * ss
        # the anchor is the snout and `sa` is the facing, so step *forward*
        # half a body to keep the fish centred on the site it was given
        hx = sx + math.cos(sa) * L * 0.42
        hy = sy + math.sin(sa) * L * 0.42
        body = paint_fish(cv, rng, spec, pal, hx, hy, L, sa, depth, occ)
        if depth < 0.4 and rng.random() < 0.30:
            mx, my = body.pos(0.0, -0.1)
            paint_bubbles(cv, rng, mx, my, L * 0.22, body.face(),
                          rng.randint(2, 4))

    # --- 水族 — one kind of companion, never both ----------------------
    comp = rng.random()
    if comp < 0.22:
        ss = S * rng.uniform(0.085, 0.130)
        # anchored low: 虾 and 蝌蚪 are bottom-dwellers, and one floating in
        # the top third of the sheet reads as a shrimp in the sky — it is the
        # 浮萍 that tells you which way is up, so everything else has to obey
        # it
        spot = occ.clear_spot(rng, [(W * 0.5, H * 0.72, S * 0.22)],
                              ss * 1.3, ss * 0.7, tries=50)
        if spot:
            paint_shrimp(cv, rng, pal, spot[0], spot[1], ss,
                         flow + rng.gauss(0.0, 0.9),
                         rng.uniform(0.1, 0.4))
            occ.add(spot[0], spot[1], ss * 0.8)
    elif comp < 0.46:
        # 蝌蚪成群 — tadpoles come in little crowds, all drifting one way
        a0 = flow + rng.gauss(0.0, 0.7)
        cx = rng.uniform(W * 0.15, W * 0.85)
        cy = rng.uniform(H * 0.45, H * 0.88)
        for _ in range(rng.randint(3, 7)):
            # Bigger and closer together than the first version. At 0.01 S,
            # scattered over 0.07 S, four of them in open paper were specks
            # of dirt rather than a crowd of tadpoles — too small to show a
            # tail, too far apart to be a group.
            ts = S * rng.uniform(0.013, 0.022)
            tx = cx + rng.gauss(0.0, S * 0.048)
            ty = cy + rng.gauss(0.0, S * 0.040)
            if occ.overlap(tx - ts * 2, ty - ts * 2, tx + ts * 2,
                           ty + ts * 2, ts) == 0.0:
                paint_tadpole(cv, rng, pal, tx, ty, ts,
                              a0 + rng.gauss(0.0, 0.4))

    # --- 浮萍 on the surface, in front of everything -------------------
    if rng.random() < sc["duck"]:
        duckweed(cv, rng, pal, W, H, rng.randint(4, 12), occ=occ)

    # --- 题款 ----------------------------------------------------------
    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))])
    inscribe(cv, rng, occ, S, title)
    return title, ground


def age(cv, rng):
    """A last breath of ink dust, as all three other painters do."""
    W, H = cv.w, cv.h
    for _ in range(int(W * H * 0.00008)):
        disc(cv, rng.uniform(0, W), rng.uniform(0, H),
             rng.uniform(0.6, 1.8), I("dan"), rng.uniform(0.05, 0.22))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _key(tbl, name, what):
    """Look a table key up by key *or* by 中文 name.

    `--list` prints 金鱼 next to `goldfish`, so `--species 金鱼` has to work;
    being shown a name and then told it is unknown is the kind of small
    rudeness that makes a tool feel unfinished."""
    if name in tbl:
        return name
    for k, v in tbl.items():
        if v.get("cn") == name or v.get("en") == name:
            return k
    raise SystemExit("unknown %s %r; try --list" % (what, name))


def paint(path, size=(900, 1200), seed=None, scene=None, species=None,
          palette=None, quiet=False):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    pk = species or sorted(SPECIES)[rng.randrange(len(SPECIES))]
    lk = palette or sorted(PALETTES)[rng.randrange(len(PALETTES))]
    sk = _key(SCENES, sk, "scene")
    pk = _key(SPECIES, pk, "species")
    lk = _key(PALETTES, lk, "palette")

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, pk, lk)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-24s %s  %s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk, SPECIES[pk]["cn"],
                 PALETTES[lk]["cn"], ground, seed, size[0], size[1],
                 time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese fish paintings (游鱼).")
    ap.add_argument("-o", "--out", default="fish.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="群游|双鱼|独游|荷塘|藻荇")
    ap.add_argument("--species",
                    help="goldfish|carp|mandarin|crucian (or 金鱼|鲤鱼|…)")
    ap.add_argument("--palette", help="水墨|朱砂|赭金|墨彩")
    ap.add_argument("--size", default="900x1200", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("鱼 (species):")
        for k, v in SPECIES.items():
            print("  %-10s %-4s (%s)  %s" % (k, v["cn"], v["season"], v["en"]))
        print("章法 (composition):")
        for k, v in SCENES.items():
            print("  %-6s %-2d-%-2d 尾  %s" % (k, v["n"][0], v["n"][1],
                                               v["en"]))
        print("设色 (colour):")
        for k, v in PALETTES.items():
            print("  %-6s %s" % (k, v["en"]))
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1200")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("游鱼 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.species, a.palette)
    return 0


if __name__ == "__main__":
    sys.exit(main())
