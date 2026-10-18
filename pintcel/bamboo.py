#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bamboo.py — 随机生成中国墨竹图 / Random Chinese bamboo paintings (墨竹).

墨竹 is the oldest and strictest of the 写意 subjects: no colour is required,
no outline is drawn, and the whole picture is made of exactly three marks —
竿, 枝, 叶. Because there is nothing to hide behind, the tradition is almost
entirely a set of *rules about where strokes may go*, handed down as rhymed
formulas (叶诀) and prohibitions (忌). This generator is built as those rules:

    纸本   paper ground, and a flat wash for the weather
    立竿   the culms — a stack of straight 节 hinged at the nodes, never a pipe
    点节   the node rings, the darkest ink in the painting
    生枝   branches off the nodes, alternating sides, 鹿角 / 鱼骨
    布叶   leaf clusters written as 个字 / 介字 / 飞燕 / 落雁 / 惊鸦
    分墨   two or three ink planes: 近浓远淡, never two culms at one tone
    坡石   a 折带 rock at the foot, 苔点 moss on its crest
    天候   晴 / 风 / 雨 / 雪 / 月 — one weather, obeyed by every leaf
    题款   inscription column and seals in the 留白

The idea carried over from gongbi.py is that **a shape is a field, not a
polygon**; from shanshui.py, that **the primitive should be the thing the
painter actually practises**. Here that thing is the single pressed-and-lifted
brush stroke: 起笔 lands, 行笔 carries, 收笔 lifts to a point. One 竹叶 is one
such stroke and nothing else, so it is drawn with gongbi's `Blade` field — the
same renderer that draws petals — reparametrised as a lancet with no contour
line, since 写意 has no 勾线.

Infrastructure (PNG writer, value noise, paper ground, the signed-field blade,
brush strokes, baked calligraphy outlines, seals, the occupancy map) is shared
with gongbi.py rather than copied — this module imports it.

Usage
-----
    python3 bamboo.py                             # one random painting
    python3 bamboo.py -n 6                        # six of them
    python3 bamboo.py --scene 竹石 --weather 风
    python3 bamboo.py --style 翠竹 --seed 7
    python3 bamboo.py --size 900x1300 -o scroll.png
    python3 bamboo.py --list                      # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Blade, Canvas, Noise, Occupancy, BrushFont, P, bell,
                    bezier, brush_path, brush_seg, clamp01, disc, jitter,
                    mix, rgb, saturate, tangent, paint_column, paint_seal,
                    TIANGAN, DIZHI)

# --------------------------------------------------------------------------
# 墨色 — ink tones, and the three colour schools a bamboo painter may use
# --------------------------------------------------------------------------

INK = {
    "jiao":  "#1C1917",   # 焦墨  burnt — node rings and the darkest accents
    "nong":  "#2C2825",   # 浓墨  thick
    "zhong": "#4A4337",   # 重墨  heavy
    "dan":   "#7E7668",   # 淡墨  diluted
    "qing":  "#ABA597",   # 清墨  clear, a breath of grey
}


def I(name):
    return rgb(INK[name])


# Every style gives the two stops of a stroke's ramp — the pale flank and the
# loaded core — for leaf and culm separately, plus the ink of the node rings.
# 墨竹 is the orthodox form; 朱竹 is Su Shi's joke that became a genre (asked
# why he painted bamboo in vermilion, he asked back what colour ink bamboo
# is); 翠竹 is 石绿 over the same ink drawing.
STYLES = {
    "墨竹": dict(cn="墨竹", en="ink bamboo",
                 leaf=("#7D776A", "#2B2724"), culm=("#9E9787", "#4A4337"),
                 node="#1F1C19", rock=("#B6B0A0", "#3A352E"), bow=0.03),
    "朱竹": dict(cn="朱竹", en="vermilion bamboo",
                 leaf=("#D08264", "#982518"), culm=("#D9A184", "#A04E31"),
                 node="#5E1D13", rock=("#BCADA0", "#4A332A"), bow=0.16),
    "翠竹": dict(cn="翠竹", en="malachite bamboo",
                 leaf=("#A4B682", "#32543A"), culm=("#BDC49C", "#5C6A3D"),
                 node="#2A3322", rock=("#B2B6A4", "#3C443A"), bow=0.13),
}

# 章法 — the four standard framings. `culms` lists (ink strength, how many)
# from back to front: the painting is built in planes and no two planes share
# a tone. `lead` is the strength of the 主竿, the one culm that carries the
# picture — 丛竹 without a leader is a thicket, not a composition.
SCENES = {
    "丛竹": dict(cn="丛竹", en="a grove",
                 culms=((0.26, (2, 4)), (0.58, (1, 2)), (1.0, (2, 3))),
                 rock=0.22, lead=1.0, twigs=(2, 4), leaf=1.0, grass=0.5),
    # 折枝 carries the whole sheet on one branch, so that branch has to be
    # fully leaved — five or six twigs, not three. With three the diagonal
    # is elegant and two thirds of the paper is simply unused, which is
    # 空 rather than 留白: emptiness has to be *left*, and nothing is left
    # over if there was never anything to leave.
    "折枝": dict(cn="折枝", en="one broken branch",
                 culms=((0.30, (1, 2)), (1.0, (1, 1)),),
                 rock=0.0, lead=1.0, twigs=(6, 8), leaf=1.15, grass=0.15,
                 broken=True, dense=1.5),
    "竹石": dict(cn="竹石", en="bamboo and rock",
                 culms=((0.24, (1, 3)), (0.60, (1, 2)), (1.0, (2, 3))),
                 rock=1.0, lead=1.05, twigs=(2, 4), leaf=0.95, grass=0.8),
    # 新篁 is young bamboo, not a single cane. Allowed to roll one near culm
    # and no companion — which is what (0, 1) and (1, 2) permitted — the
    # whole sheet came down to one thin stem against the edge of the band
    # carrying three small groups, and nine tenths of the paper was blank.
    # That is not 留白: 留白 is what a composition leaves, and a composition
    # of one stem has nothing to leave. Two or three young culms with a pale
    # companion behind them is the subject the scene is named for.
    "新篁": dict(cn="新篁", en="young bamboo",
                 culms=((0.22, (1, 2)), (1.0, (2, 3))),
                 rock=0.3, lead=0.92, twigs=(3, 5), leaf=0.80, grass=0.45),
}

# 天候 — one weather for the whole sheet. `sweep` is how far the leaves are
# turned from their natural lie, `spread` how much the fan of a cluster is
# compressed (a gale makes every leaf parallel), `droop` the pull on the
# branches, `bow` the lean of the culms themselves.
WEATHER = {
    "晴": dict(cn="晴竹", en="fair weather",
               sweep=0.0, spread=1.0, droop=0.0, bow=0.6, lift=0.22,
               veil=None, k=1.0),
    "风": dict(cn="风竹", en="wind",
               sweep=0.95, spread=0.74, droop=0.5, bow=1.9, lift=0.0,
               veil=None, k=1.0, fly=(2, 4)),
    "雨": dict(cn="雨竹", en="rain",
               sweep=0.0, spread=0.78, droop=1.0, bow=1.1, lift=-0.30,
               veil=("#8C9A9E", 0.13), k=0.86, rain=1.0),
    "雪": dict(cn="雪竹", en="snow",
               sweep=0.0, spread=0.84, droop=0.72, bow=0.9, lift=-0.14,
               veil=("#F4F2EA", 0.20), k=1.06, snow=1.0),
    "月": dict(cn="月竹", en="moonlight",
               sweep=0.2, spread=0.94, droop=0.18, bow=0.7, lift=0.14,
               veil=("#6E7A86", 0.10), k=1.02, moon=1.0),
}

# 叶诀 — the leaf formulas, learned as characters because that is what the
# groups look like. Each entry is a list of (angle off the twig, length
# factor, bend) — the *skeleton* of a group, before any jitter. The classic
# prohibitions are built into these numbers: no two leaves at one angle
# (忌并指, fingers of a hand), no symmetric pair of equal length (忌蜻蜓, a
# dragonfly), no crossing near a right angle (忌井字).
CLUSTERS = {
    "一笔": [(0.00, 1.00, 0.10)],
    "飞燕": [(-0.58, 0.96, -0.13), (0.52, 0.84, 0.15)],
    "个字": [(-0.62, 0.84, -0.11), (-0.04, 1.00, 0.04), (0.50, 0.78, 0.14)],
    "介字": [(-0.74, 0.76, -0.14), (-0.26, 0.97, -0.05),
             (0.20, 1.00, 0.07), (0.66, 0.72, 0.15)],
    "落雁": [(-0.40, 0.70, -0.08), (0.04, 0.90, 0.05),
             (0.44, 1.00, 0.13), (0.86, 0.66, 0.21)],
    "惊鸦": [(-0.86, 0.66, -0.17), (-0.44, 0.88, -0.07), (-0.02, 1.00, 0.03),
             (0.40, 0.92, 0.11), (0.82, 0.68, 0.19)],
    "双叠": [(-0.70, 0.90, -0.12), (-0.18, 0.74, -0.02),
             (0.26, 0.98, 0.08), (0.58, 0.62, 0.16), (0.98, 0.78, 0.22)],
}

# Which formulas suit which plane. A 远竹 group is read as one patch, so the
# big open fans are wasted on it and the small ones are illegible; the near
# plane is where a painter shows off a 惊鸦.
BY_PLANE = {
    "near": ("个字", "介字", "落雁", "惊鸦", "双叠", "飞燕"),
    "mid":  ("个字", "介字", "落雁", "飞燕", "惊鸦"),
    "far":  ("个字", "介字", "飞燕", "落雁"),
}


def dilute(c, k):
    """兑水 — ink thinned with water.

    Every stroke in this painting multiplies into the paper instead of
    covering it, exactly as a wash does, and under a multiply white is the
    identity. So diluting ink *is* mixing it toward white: 淡墨 is not grey
    paint, it is less pigment in the same water.
    """
    return mix(c, (255.0, 255.0, 255.0), 1.0 - clamp01(k))


def plane_tone(st, part, k):
    """The two ramp stops for one ink plane. `k` is 墨色: 1.0 is 浓墨 straight
    from the stone, 0.2 is the far wash."""
    lo, hi = st[part]
    return dilute(rgb(lo), 0.22 + 0.78 * k), dilute(rgb(hi), 0.14 + 0.86 * k)


def veil(cv, color, alpha):
    """A flat wash over the whole sheet — grey for 雨, white for 雪, dusk-blue
    for 月. One byte table per channel, applied by `bytes.translate`, so a
    full-sheet wash costs about as much as a single leaf."""
    c = color if not isinstance(color, str) else rgb(color)
    for ch in range(3):
        t = bytes(min(255, int(v + (c[ch] - v) * alpha)) for v in range(256))
        cv.buf[ch::3] = cv.buf[ch::3].translate(t)


# --------------------------------------------------------------------------
# 竿 — the culm. 竹有节: a stack of straight internodes, hinged at the nodes.
#
# The single most common failure in generated bamboo is a smooth curve: draw
# a culm as one bezier and it reads as a garden hose, because a real culm is
# *rigid between nodes* and only changes direction where it is jointed. So the
# spine is built as a polyline whose vertices are the nodes, and every piece
# between two nodes is rendered as its own straight tapered field.
# --------------------------------------------------------------------------

def culm_spine(rng, x, y, h, ang, w0, w1, bow, nodes, kink=0.0):
    """立竿 — node points of one culm, foot first.

    Returns [(x, y, w, ang_of_segment_above)], length `nodes` + 1.

    节 spacing is the giveaway of a painter who has looked at bamboo: short
    at the foot, longest around the lower middle, shortening again toward the
    tip. A uniform ladder looks like scaffolding.
    """
    segs = []
    for i in range(nodes):
        t = (i + 0.5) / nodes
        # one hump, peaking at a third of the way up
        f = 0.42 + 0.95 * math.sin(math.pi * min(1.0, t * 0.78 + 0.16)) ** 1.1
        segs.append(f * rng.uniform(0.88, 1.14))
    k = h / sum(segs)
    segs = [s * k for s in segs]

    # 一节之折 — a culm is not a bent rod but a jointed one, and somewhere
    # along it there is usually a single joint that gives much more than its
    # neighbours. Spread the same total bend evenly over every node and the
    # culm comes out as a drawing-board French curve; put a good part of it
    # into one hinge and it comes out as bamboo that has stood in weather.
    # How much it gives goes with `kink` as well, not just how often. A cut
    # 折枝 branch wants a real elbow; a culm standing in a grove does not,
    # because a hinge of a tenth of a radian is six degrees and six degrees
    # over the height of a sheet is a hundred pixels away from whatever its
    # neighbours are doing — one hinged culm in a row of straight ones is
    # read as a mistake rather than as character.
    #
    # *Where* it gives decides whether it reads as character or as a fault,
    # and this had it wrong. A hinge is a step in the angle, and every node
    # above it inherits the step; put it a third of the way up a culm and
    # two thirds of the culm is leaning somewhere its neighbours are not, so
    # the base-to-tip line comes out ten degrees off theirs and that one
    # culm is the one the eye goes to. Put the same hinge in the upper third
    # and only the part that is *supposed* to give has moved. So a grove
    # culm is hinged near its tip and lightly; a cut 折枝 branch, which has
    # no neighbours to disagree with, may elbow anywhere along itself and
    # hard — which is what `kink` now selects between.
    hinge = -1
    if kink > 0.0 and nodes >= 4 and rng.random() < kink:
        held = 1.0 - 0.72 * min(1.0, kink * 1.3)      # 0 = free, 1 = tip only
        lo_h = 1 + int((nodes - 2) * held)
        hinge = rng.randrange(min(lo_h, nodes - 2), nodes - 1)
        # usually with the bow, now and then against it
        hsgn = 1.0 if rng.random() < 0.72 else -1.0
        hamt = hsgn * rng.uniform(0.02, 0.10) * (0.42 + 0.68 * kink)

    out = []
    cx, cy, a = x, y, ang
    for i, s in enumerate(segs):
        t = i / max(1, nodes - 1)
        out.append((cx, cy, w0 + (w1 - w0) * t ** 1.5, a))
        cx += math.cos(a) * s
        cy += math.sin(a) * s
        if i == hinge:
            a += hamt
        # The culm bends more the higher it goes — the tip of a bamboo is
        # always the part that gives. Each node adds a little of the bow plus
        # its own small error, and the error never unwinds, which is why a
        # real culm wanders instead of arcing.
        #
        # Divided by `nodes`, so `bow` is the *total* turn over the culm and
        # not the turn per joint. A fixed amount per joint made the bend a
        # function of how many joints a culm happened to get, and that count
        # comes from 节 spacing — seven to eleven diameters — so a thin
        # back-plane culm is given half again as many nodes as a thick front
        # one, and a short 出梢 culm half as many as a tall 出纸 one. Three
        # culms standing in one grove under one wind would then bend 23°, 15°
        # and 11°: the pale one at the back came out as a drawn bow beside
        # neighbours that were nearly straight, which is exactly the member
        # of the grove that catches the eye as wrong.
        a += bow * 0.17 / nodes * (0.25 + t * 1.6) \
            + rng.gauss(0.0, 0.010 * math.sqrt(10.0 / nodes))
    out.append((cx, cy, w1, a))
    return out


def node_mark(cv, rng, x, y, ang, w, ink, alpha, crisp=True):
    """点节 — the ring at a joint.

    Written as one 乙-shaped stroke: the brush lands on the far side, drags
    across the culm bowing slightly toward the foot, and flicks up at the
    near side. It is the darkest ink on the sheet, and it is what makes the
    viewer read a cylinder rather than a stripe.
    """
    if w < 1.0:
        return
    nx, ny = -math.sin(ang), math.cos(ang)      # across the culm
    ux, uy = math.cos(ang), math.sin(ang)       # up it
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    ex = 1.04 + 0.12 * rng.random()
    pts = []
    for i in range(7):
        t = i / 6.0
        v = (t * 2.0 - 1.0) * ex * w * sgn
        # the bow: the middle of the ring sags toward the foot
        o = -(1.0 - (t * 2.0 - 1.0) ** 2) * w * (0.30 if crisp else 0.18)
        pts.append((x + nx * v + ux * o, y + ny * v + uy * o))
    # A 节 is a *written* mark, not a hoop of pipe lagging: one thin stroke
    # that happens to be the darkest ink on the sheet. Drawn heavy it turns
    # the culm into a length of bamboo scaffolding, which is the single
    # quickest way to lose the painting.
    lw = max(0.5, w * (0.15 if crisp else 0.11))
    brush_path(cv, pts, lw * 0.75, lw * 1.2, ink, alpha)
    if crisp and w > 2.5:                       # 收笔 — the flick at the end
        hx, hy = pts[-1]
        brush_seg(cv, hx, hy, hx + ux * w * 0.75 + nx * sgn * w * 0.3,
                  hy + uy * w * 0.75 + ny * sgn * w * 0.3,
                  lw * 0.9, lw * 0.2, ink, alpha * 0.85)


def draw_culm(cv, rng, spine, lt, dp, node_ink, alpha, soft=0.0, k=1.0,
              dry=0.0):
    """竿 — render the internodes, then ring the joints.

    Each internode is a `Blade` in stem mode: a straight tapered field with
    the light raked across it, which is exactly a cylinder. `tuck=0` keeps
    consecutive pieces at a shared width where they meet, so the silhouette
    runs on through the joint instead of pinching.
    """
    n = len(spine) - 1
    for i in range(n):
        x, y, w, a = spine[i]
        w2 = spine[i + 1][2]
        s = math.hypot(spine[i + 1][0] - x, spine[i + 1][1] - y)
        if s < 1.5 or w < 0.45:
            continue
        Blade(cx=x, cy=y, ang=a, L=s * 1.04, W=w,
              mode="stem", stem_taper=max(0.0, min(0.9, 1.0 - w2 / w)),
              tuck=0.0, caps=True,
              light=lt, deep=dp, ink=node_ink,
              # 竹竿 is drawn 中锋 in one pass, so the ink is heaviest where
              # the brush started — at the node below — and lightens upward.
              w_base=0.40 * k, p_base=1.5, w_tip=0.0,
              # The edge of a culm is a *drawn* edge: the brush's own side,
              # darker than the middle of the stroke. A smooth cylinder
              # gradient instead — which is what a high `lit` gives — is an
              # airbrush, and reads as a photograph of a bamboo pole.
              w_rim=0.44, p_rim=3.0, w_mid=0.0,
              lit=0.40, hl=0.0, line=0.0, soft=soft,
              alpha=alpha, blend="mul", bow=0.05).render(cv)
        if dry and w > 2.0 and rng.random() < dry:
            # 飞白 — the dry side of the brush leaves the paper showing in a
            # thin streak. Painted light, along the culm, never across it.
            o = w * rng.uniform(-0.45, 0.45)
            nx, ny = -math.sin(a), math.cos(a)
            brush_path(cv, [(x + nx * o, y + ny * o),
                            (x + math.cos(a) * s + nx * o,
                             y + math.sin(a) * s + ny * o)],
                       w * 0.11, w * 0.16, (255, 253, 246),
                       rng.uniform(0.10, 0.22))
    # 点节 last, over the joints. A 远竹 gets none: at that distance the ring
    # is below the threshold of what the eye resolves, and painting it anyway
    # is what makes a pale background culm read as a near one that has merely
    # been under-inked.
    if k < 0.34:
        return
    for i in range(1, n):
        x, y, w, a = spine[i]
        node_mark(cv, rng, x, y, a, w, node_ink, alpha * (0.95 if k > 0.6
                                                          else 0.75),
                  crisp=k > 0.55)


# --------------------------------------------------------------------------
# 枝 — branches. 鹿角 (deer-horn, lifting) in fair weather, 鱼骨 (fishbone,
# hanging) in rain; both are the same recursion with a different sign.
# --------------------------------------------------------------------------

def _hold(p, lo, hi, k=4.0):
    """Fold an angle back into the band [lo, hi] without flattening onto it.

    A hard clamp would be wrong here even though the band is: in 晴 every
    group is lifted past the ceiling, so `min`/`max` would set them all to
    exactly the ceiling and a dozen sprays would lie at one identical angle.
    What is wanted is a rail that gives a little — the overshoot is kept but
    compressed, never by more than `1/k`, so the order and the spacing of
    the angles survive while the extremes stop being extreme.
    """
    if p < lo:
        d = lo - p
        return lo - d / (1.0 + k * d)
    if p > hi:
        d = p - hi
        return hi + d / (1.0 + k * d)
    return p


def _pitch(ang):
    """The angle's rise or fall, with its left/right sense thrown away.
    Positive is downward, as y is."""
    return math.atan2(math.sin(ang), abs(math.cos(ang)))


def _aim(pitch, out):
    """...and back again, onto the given side."""
    return math.atan2(math.sin(pitch), out * math.cos(pitch))


def draw_twig(cv, rng, x, y, ang, L, w, lt, dp, alpha, tips, depth=0,
              droop=0.0, soft=0.0, sweep=0.0, out=None):
    """One branch, 节 by 节, recording its free ends for the leaves.

    A 竹枝 is jointed like the culm, only faster: three or four short pieces,
    each kicking a few degrees, and a side twig leaving at a joint. Branches
    never leave a culm in pairs on the same side — the nodes alternate, and
    so does everything that grows from them.

    枝不乱 — and a branch that leaves a node going outward keeps going
    outward. The angle used to be accumulated by free addition, so four
    joints of jitter, a droop term and a sub-twig's kick could between them
    turn a twig back across its own culm or stand it on end; the leaves at
    its end then pointed any way at all, which is the one thing a reader
    notices immediately. So the branch is held in a sector instead — a fixed
    side in x, and a pitch free to rise somewhat above the horizontal or
    fall well below it, which is the whole range a leafed branch occupies.
    """
    if out is None:
        out = 1.0 if math.cos(ang) >= 0.0 else -1.0
    plo, phi = -0.95, 0.80 + 0.45 * droop
    sw = sweep * out                  # positive when this branch lies downwind
    p = _hold(_pitch(ang) + 0.18 * droop - 0.10 * sw, plo, phi)
    n = rng.randint(2, 4)
    cx, cy, a = x, y, _aim(p, out)
    for i in range(n):
        s = L / n * rng.uniform(0.82, 1.18)
        w0 = w * (1.0 - 0.55 * i / n)
        w1 = w * (1.0 - 0.55 * (i + 1) / n)
        nx = cx + math.cos(a) * s
        ny = cy + math.sin(a) * s
        if w0 > 0.85:
            Blade(cx=cx, cy=cy, ang=a, L=s * 1.05, W=w0,
                  mode="stem", stem_taper=max(0.0, min(0.9, 1.0 - w1 / w0)),
                  tuck=0.0, light=lt, deep=dp, ink=dp,
                  w_base=0.55, p_base=1.6, w_rim=0.22, p_rim=2.2,
                  lit=0.55, line=0.0, soft=soft, alpha=alpha,
                  blend="mul").render(cv)
        else:
            brush_path(cv, [(cx, cy), (nx, ny)], max(0.45, w0),
                       max(0.4, w1), dp, alpha * 0.85)
        cx, cy = nx, ny
        # the kick at each joint, plus gravity and the wind — all of it in
        # pitch, so none of it can turn the branch round
        p = _hold(p + rng.gauss(0.0, 0.09) + 0.10 * droop - 0.05 * sw,
                  plo, phi)
        a = _aim(p, out)
        # Leaves grow from the twig's own nodes, not only from its end — so
        # an interior joint is a leaf site too. Without this a culm carries
        # one group per branch and the sheet stays bare.
        if i < n - 1 and rng.random() < 0.45:
            tips.append((cx, cy, _aim(_hold(p + rng.gauss(0.0, 0.25),
                                            plo, phi), out), w * 0.4, out))
        if depth < 2 and i < n - 1 and rng.random() < 0.5:
            p2 = _hold(p + rng.choice((1.0, -1.0)) * rng.uniform(0.35, 0.8),
                       plo, phi)
            draw_twig(cv, rng, cx, cy, _aim(p2, out),
                      L * rng.uniform(0.38, 0.6), w * 0.56,
                      lt, dp, alpha, tips, depth + 1, droop, soft, sweep,
                      out=out)
    tips.append((cx, cy, a, w * 0.45, out))


# --------------------------------------------------------------------------
# 叶 — the leaf. One stroke: 起笔 lands, 行笔 carries, 收笔 lifts to a point.
# --------------------------------------------------------------------------

def leaf(cv, rng, x, y, ang, L, lt, dp, alpha, w=None, bend=0.08, soft=0.0,
         k=1.0, back=False):
    """One 竹叶.

    The silhouette is the lancet a flat brush leaves when it is pressed and
    then lifted: it opens fast from the landing point, carries its width for
    half its length and runs out in a long acuminate tip (`tip`). The ink is
    heaviest at the base and in the core of the stroke — `w_base` along the
    leaf, `w_mid` across it — because that is where the brush held its load.
    There is no contour line: 写意 draws no 勾线, and a leaf with one is a
    cut-out of a leaf.
    """
    # 竹叶 is a long stroke: eleven or twelve times its own width. Fatter
    # than that and the formulas stop reading as bamboo — a 个字 of broad
    # blades is a maple spray, which is exactly what the first draft drew.
    W = w if w is not None else L * rng.uniform(0.072, 0.098)
    if L < 4.0 or W < 0.8:
        return
    if back:                      # 反叶 — the pale underside, turned over
        lt = dilute(lt, 0.45)
        dp = dilute(dp, 0.52)
    Blade(cx=x, cy=y, ang=ang, L=L, W=W,
          k1=1.8, e1=0.44, k2=4.0, e2=0.72, tip=0.40,
          bend=bend, tuck=1.0,
          light=lt, deep=dp, ink=dp, line=0.0,
          w_base=0.92 * k, p_base=1.15,
          w_mid=0.34, p_mid=1.5, w_rim=0.0,
          lit=0.30, soft=soft, alpha=alpha,
          # A bamboo leaf is one fast stroke and its edge is clean: the
          # hand-wobble gongbi puts on a petal contour has a wavelength of
          # about nine pixels, which on a blade this narrow comes out as
          # serration — barbs, and the group reads as a feather.
          wob=0.0, bow=0.06, blend="mul").render(cv)


def leaf_cluster(cv, rng, x, y, a0, L, lt, dp, alpha, we, plane="near",
                 scale=1.0, soft=0.0, k=1.0, form=None, snow=0.0, out=None):
    """布叶 — one group of leaves written as a character.

    Three things make a group read as bamboo rather than as a fern:

    *一笔一叶* — every leaf is a separate stroke with its own load of ink, so
    the tones inside a group differ. A group drawn at one alpha is a leaf
    *shape* repeated, which is what printed wallpaper does.

    *不同根* — the bases are offset along the twig by a few pixels instead of
    radiating from one point. Real leaves grow from successive nodes of the
    twig; a perfect fan is the mark of a stencil.

    *一势* — the group has a single direction of travel, given by the weather,
    and the formula only decides how the leaves deviate from it.
    """
    form = form or BY_PLANE[plane][rng.randrange(len(BY_PLANE[plane]))]
    rec = CLUSTERS[form]
    # How far the group opens. There are two failure modes on either side of
    # this number and they look nothing alike: opened to the formula's full
    # angle the leaves radiate from one point like fingers (忌并指), and
    # closed up they lie parallel and the group reads as one feather. The
    # opening has to stay wide enough that paper shows between the leaves.
    spread = 0.92 * we["spread"] * rng.uniform(0.88, 1.12)
    # 势 — where this group is going, in the world's coordinates and not the
    # twig's. The first version added the weather onto the twig's own angle,
    # which is only right for a group that happens to point to the right:
    # added to one pointing left, a positive `droop` rotates it *upward*. So
    # in 雨 half the groups on a sheet hung and the other half flew straight
    # up (a0 = -2.42 came out at lead = -1.63, which is vertical), and in 晴
    # the same thing happened the other way about. Gravity pulls down and
    # the wind blows one way whatever the twig is doing, so the group is
    # worked as a side plus a pitch, and only the pitch takes the weather.
    out = out if out is not None else (1.0 if math.cos(a0) >= 0.0 else -1.0)
    pitch = (_pitch(a0)
             - we["lift"] * rng.uniform(0.5, 1.4)
             + we["droop"] * rng.uniform(0.25, 0.75)
             - we["sweep"] * we.get("wind", 1.0) * out * rng.uniform(0.5, 1.1))
    # 叶不朝天 — and the group is held in the sector a spray of bamboo can
    # actually occupy. The one direction that never occurs in the genre is a
    # fan standing straight up off a twig: leaves hang from a joint, and
    # even the lifted leaves of 晴 only reach for the diagonal.
    lead = _aim(_hold(pitch, -0.62, 1.10, k=3.0), out)
    ux, uy = math.cos(a0), math.sin(a0)
    order = sorted(range(len(rec)), key=lambda i: -abs(rec[i][0]))
    for idx, i in enumerate(order):
        da, fl, bd = rec[i]
        a = lead + da * spread + rng.gauss(0.0, 0.07)
        ll = L * scale * fl * rng.uniform(0.84, 1.14)
        # 不同根 — and the outermost leaves of the group sit furthest back
        # along the twig, which is what turns a fan into a spray. Only a
        # little, though: stack the bases too far apart and the leaves become
        # the barbs of a quill.
        o = -L * scale * (0.03 + 0.16 * abs(da) + 0.07 * rng.random())
        # 浓淡 — the outermost leaves of a group were drawn first, with the
        # brush at its fullest; by the last ones it is running dry. Plus the
        # ones underneath are simply further away.
        kk = k * (0.80 + 0.26 * idx / max(1, len(rec) - 1))
        aa = alpha * rng.uniform(0.80, 1.0) * (1.0 - 0.10 * (len(rec) - idx))
        leaf(cv, rng, x + ux * o, y + uy * o, a, ll, lt, dp,
             clamp01(aa), bend=bd * rng.uniform(0.7, 1.5) + rng.gauss(0, 0.02),
             soft=soft, k=kk,
             back=(plane == "near" and rng.random() < 0.13))
        if snow and rng.random() < 0.55:
            _snow_on_leaf(cv, rng, x + ux * o, y + uy * o, a, ll)
    return form


def _snow_on_leaf(cv, rng, x, y, ang, L):
    """雪 — snow does not coat a leaf, it sits in the trough of it. A short
    pale stroke along the upper half, narrower than the leaf, so the leaf's
    own edge still reads underneath."""
    Blade(cx=x, cy=y - L * 0.035, ang=ang, L=L * rng.uniform(0.5, 0.8),
          W=L * rng.uniform(0.030, 0.055),
          k1=1.8, e1=0.44, k2=3.4, e2=0.8, tip=0.45,
          light=(255, 255, 252), deep=(250, 249, 243), ink=(250, 249, 243),
          line=0.0, w_base=0.2, w_mid=0.0, w_rim=0.0, lit=0.0,
          soft=0.9, alpha=rng.uniform(0.40, 0.72), blend="over").render(cv)


def cluster_sites(rng, tips, n, R, sep=0.95, sheet=None, edge=0.4, taken=()):
    """聚散 — which twig ends actually get leaves, and how big a group.

    The defect this fixes is the one that makes generated bamboo unreadable:
    hang a formula off every twig end and the groups merge into a single
    black cloud, which is precisely what the old masters' 忌 are about
    (忌重叠如蕉 — piled up like a banana leaf). So the ends are *chosen*,
    farthest-point: offer the whole list, score each by how far its group
    would sit from every group already settled, take the roomiest, stop when
    the best remaining one would collide. A twig left bare is not a mistake —
    空枝 is in every scroll, and the paper between groups is the picture.

    `taken` carries the groups already settled elsewhere on the sheet, and
    it matters more than anything else in here. Called once per culm with an
    empty slate — which is how this started — the spacing only keeps a culm
    away from *itself*, and groups belonging to different culms are free to
    land on each other: in a 丛竹 of nine culms the sheet took fifty groups
    and the upper half went solid. 疏密 is a property of the sheet.

    Returns (sites, marks): each site is its tip tuple with (scale, crowd)
    appended — the group size already discounted where it is tight, and a
    crowding value the caller can use to pull a buried group back in tone —
    and `marks` is the settled (x, y, radius) list, to pass on as `taken`.
    """
    marks = list(taken)
    if not tips:
        return [], marks
    out = []

    def room(x, y, r):
        s = min((math.hypot(x - mx, y - my) - sep * (r + mr)
                 for mx, my, mr in marks), default=1e9)
        if sheet is not None:
            s = min(s, min(x, y, sheet[0] - x, sheet[1] - y) - edge * r)
        return s

    pool = list(tips)
    for _ in range(n):
        best, best_s = None, None
        for j, t in enumerate(pool):
            s = room(t[0], t[1], R)
            if best_s is None or s > best_s:
                best, best_s = j, s
        if best is None or best_s < -0.62 * R:
            break
        t = pool.pop(best)
        sc = 0.72 + 0.34 * clamp01((best_s + 0.5 * R) / R)
        marks.append((t[0], t[1], sc * R))
        out.append(t + (sc, clamp01(-best_s / R)))
    return out, marks


# --------------------------------------------------------------------------
# 石 — the rock at the foot. 竹石 is a genre because the two subjects need
# each other: bamboo is all vertical line, a rock is all mass and horizontal
# shelf, and together they make a composition out of what was a pattern.
# --------------------------------------------------------------------------

def rock(cv, rng, x0, x1, yb, h, lt, dp, ink, alpha=0.9, moss=1.0):
    w = int(x1 - x0)
    if w < 10 or h < 6:
        return None
    # 勾 — the crest first, as a painter does.
    #
    # 石分三面, and a rock in a painting is almost never one rock: it is a
    # big mass with a smaller one shouldering out of its foot. That is also
    # the construction that produces the 折带 silhouette for free — flat
    # shelves meeting steep faces at hard corners — which is what two drafts
    # of random knot heights could not do. One draft came out as a loaf of
    # bread, the next as a shark's fin, the next as a chimney, and the reason
    # each time was that a single profile over the whole width has no way to
    # say *this block ends here and another begins*.
    #
    # So: two or three angular blocks, each with a stepped level top and
    # steep sides, and the rock is their union — per column, whichever block
    # reaches highest.
    nb = rng.randint(2, 3)
    blocks = []
    for b in range(nb):
        # the first block is the mass; the others are its company, lower and
        # narrower, pushed out to one side so they break its outline
        if b == 0:
            c0, c1 = rng.uniform(0.0, 0.22), rng.uniform(0.62, 0.92)
            bh = 1.0
        else:
            # kept wholly inside the box: a block that runs off the end has
            # its flank cut away and comes out as a wedge with one vertical
            # side, which looks like a shard of slate rather than a stone
            cw_ = rng.uniform(0.26, 0.50)
            c0 = rng.uniform(0.0, 1.0 - cw_)
            c1 = c0 + cw_
            bh = rng.uniform(0.28, 0.62)
        # The stepped top of this block: 2–4 level shelves, the tallest of
        # them in the middle and the rest stepping down to either side.
        #
        # Which shelf is the tall one is not a free choice. Put the tall one
        # at an end and the flank taper runs straight into it, the level run
        # is lost, and the block comes out as a spike — a sail, which is what
        # the last draft had standing behind the rock. The tall shelf has to
        # be the one that owns the middle of the block.
        ns = rng.randint(3, 5)
        cuts = [0.0]
        for s in range(1, ns):          # one shelf per band, so each has a run
            cuts.append((s + rng.uniform(-0.30, 0.30)) / ns)
        cuts.append(1.0)
        vals, sv = [], rng.uniform(0.92, 1.0)
        for _ in range(ns):
            vals.append(sv)
            sv *= rng.uniform(0.82, 0.95)
        pk = 0
        while pk < ns - 1 and cuts[pk + 1] < 0.5:
            pk += 1
        shelf = [0.0] * ns
        lo_, hi_ = pk, pk
        for v in vals:                  # outward from the peak, both ways
            if shelf[pk] == 0.0:
                shelf[pk] = v
            elif lo_ > 0 and (hi_ >= ns - 1 or rng.random() < 0.5):
                lo_ -= 1
                shelf[lo_] = v
            elif hi_ < ns - 1:
                hi_ += 1
                shelf[hi_] = v
        # 侧 — how the two flanks come down to the ground. These have to be
        # proper diagonals, a third to a half of the block's own width: a
        # rock widens as it descends. Run them in over a tenth of the width
        # instead and the flank is a vertical wall, the shading reads as an
        # extruded box, and the thing in the painting is a chimney.
        # ...and not more than a third each, so there is always a level run
        # between them. Let the two flanks meet in the middle and the block
        # is a triangle, which is the other failure of this form: a pyramid,
        # i.e. a mountain that has wandered in from the wrong painting.
        # 不可对称 — and one flank steep, the other long. Two matched
        # diagonals under a level top is a plinth, and a plinth under
        # bamboo is furniture.
        f1, f2 = rng.uniform(0.07, 0.15), rng.uniform(0.24, 0.46)
        if rng.random() < 0.5:
            f1, f2 = f2, f1
        blocks.append((c0, c1, bh, cuts, shelf, f1, f2,
                       rng.uniform(-0.26, 0.26)))

    wob = Noise(rng, 7, 3)
    base = Noise(rng, 5, 2)
    hv = [0.0] * w
    for i in range(w):
        t = i / max(1, w - 1.0)
        best = 0.0
        for c0, c1, bh, cuts, shelf, fl, fr, tb in blocks:
            if not (c0 <= t <= c1):
                continue
            u = (t - c0) / max(1e-6, c1 - c0)
            j = 0
            while j < len(cuts) - 2 and u > cuts[j + 1]:
                j += 1
            v = shelf[j] * bh * (1.0 - tb * (u - 0.5))
            # the flanks, and a small wobble so no shelf is dead level
            # linear, so the flank is a straight diagonal face. Any exponent
            # below one rounds the shoulder over and the block comes out as
            # a dome — a boulder, which is the wrong rock for bamboo.
            v *= min(1.0, u / fl) * min(1.0, (1.0 - u) / fr)
            v += (wob.at(t, 0.5) - 0.5) * 0.07 * bh
            if v > best:
                best = v
        hv[i] = max(0.0, best)
    # Normalised against the profile that actually came out, so the caller's
    # `h` *is* the crest height. Trusting the nominal shelf heights instead
    # leaves the flank tapers and the wobble to quietly spend a third of it,
    # and the rock disappears below the bottom border — twice now.
    m = max(hv) or 1.0
    # 石宜横 again, and this is where it is actually won. The block profile
    # runs from nothing at the edges to the plateau in the middle, and with
    # the foot sitting at the bottom border that means only the plateau
    # clears it: a rock drawn 480 pixels wide shows up 130 wide and twice as
    # tall as that. So the profile is lifted onto a floor — every column at
    # least `fl0` of the crest height — and the silhouette lives in what is
    # left above it. The two ends are then run down to nothing over a short
    # steep taper of their own, which is a 折带 face and not a cut edge.
    fl0 = rng.uniform(0.42, 0.58)
    ri = rng.uniform(0.03, 0.08)
    ro = rng.uniform(0.03, 0.08)
    top = []
    for i in range(w):
        t = i / max(1, w - 1.0)
        v = (fl0 + (1.0 - fl0) * hv[i] / m) \
            * min(1.0, t / ri) * min(1.0, (1.0 - t) / ro)
        top.append(yb - v * h)
    # 地 — the foot is a line too, and it is not the bottom of a box
    bot = [yb + (base.at(i / max(1, w - 1.0), 0.2) - 0.5) * h * 0.20
           for i in range(w)]

    # 石无定形 — and no ruled edges either. The profile above is built out of
    # straight shelves and straight flanks, which is the 折带 silhouette and
    # correct, but laid down as a wash boundary it is a *cut* edge: the rock
    # comes out as a piece of grey cardboard with corners. A fine wobble
    # along the crest costs nothing and the edge becomes something a brush
    # could have left.
    rag = Noise(rng, 9, 2)
    rag2 = Noise(rng, 27, 2)
    top = [ty + (rag.at(i / max(1, w - 1.0), 0.3) - 0.5) * h * 0.075
           + (rag2.at(i / max(1, w - 1.0), 0.7) - 0.5) * h * 0.030
           for i, ty in enumerate(top)]
    # 石骨 — the grain. A smooth vertical ramp with one slow wobble in it is
    # an airbrushed solid; stone is mottled at the scale of the brush's own
    # hairs, and without that scale in the wash no amount of 皴 on top of it
    # will read as stone.
    grn = Noise(rng, 19, 2)
    lx = gb.LIGHT[0]
    buf = cv.buf
    cw = cv.w
    for i in range(w):
        px = int(x0) + i
        if px < 0 or px >= cw:
            continue
        ty = top[i]
        by = bot[i]
        d = by - ty
        if d < 1.0:
            continue
        # which way this bit of the crest faces, for the 染
        sl = (top[min(w - 1, i + 2)] - top[max(0, i - 2)]) / 4.0
        face = clamp01(0.5 - 0.5 * max(-1.0, min(1.0, sl)) * lx * 1.4)
        # Only the last few *pixels* of each flank are faded — a fraction of
        # the rock's own height here instead, which is the obvious thing to
        # write, silently erases every column shorter than that fraction.
        # Since those columns are the whole of both flanks and all the low
        # shelves, what survives is the tall middle: a stele.
        thin = clamp01(d / max(3.0, h * 0.035))
        y_lo = max(0, int(ty))
        y_hi = min(cv.h, int(by) + 1)
        for py in range(y_lo, y_hi):
            v = (py + 0.5 - ty) / d
            if v < 0.0:
                continue
            if v > 1.0:
                v = 1.0
            # 染 — dark under the crest, fading to the foot where the rock
            # meets bare paper; plus the shaded flank.
            # A rock beside bamboo is a *foil*: it has to read as mass
            # without taking the eye off the leaves, so its darkest ink
            # stays well short of theirs. The first draft reached 墨 at the
            # crest and the rock became the subject of the painting.
            # The 面 term stays small on purpose. Shading a rock by the way
            # each bit of its crest faces the light is how you render a
            # solid, and a rendered solid in a 写意 painting is a photograph
            # that has wandered into it: one flank comes up bright, the
            # other dark, and the mass reads as an extruded block.
            sh = 0.46 * (1.0 - v) ** 1.5 + 0.22 * math.exp(-(v / 0.09) ** 2) \
                + 0.11 * face + 0.06 * v \
                + (wob.at(i / max(1.0, w - 1.0), clamp01(v)) - 0.5) * 0.34 \
                + (grn.at(i / max(1.0, w - 1.0), clamp01(v)) - 0.5) * 0.22
            c = mix(lt, dp, clamp01(sh))
            # 下虚 — and the foot of a rock is left unfinished: the wash
            # thins away to nothing so the mass sits *in* the paper rather
            # than on top of it. A flat-bottomed wash has a cut edge, and a
            # cut edge along the ground is the one thing that cannot be
            # explained as ink.
            a = alpha * thin * (0.92 - 0.88 * v ** 1.5)
            e = min(py + 0.5 - ty, by - py - 0.5)
            if e < 1.0:
                a *= max(0.0, e)
            if a <= 0.01:
                continue
            idx = (py * cw + px) * 3
            o = buf[idx]
            buf[idx] = int(o + (o * c[0] * 0.00392 - o) * a)
            o = buf[idx + 1]
            buf[idx + 1] = int(o + (o * c[1] * 0.00392 - o) * a)
            o = buf[idx + 2]
            buf[idx + 2] = int(o + (o * c[2] * 0.00392 - o) * a)

    # 勾 — the crest line, drawn along the profile, never as a chord across
    # it, and run in two or three separate passes so it breaks: 石用笔要断,
    # an unbroken outline all the way round turns the rock into a cut-out.
    # Sampled at the profile's *corners*, not at a fixed pixel step. The
    # round cap of each `brush_seg` overlaps its neighbour's, so every joint
    # in a `brush_path` is a double hit of ink; stepping along the crest
    # every ten pixels puts a bead every ten pixels and the rock gets a rim
    # of beads. A 折带 profile has only a handful of corners anyway, and
    # drawing it in a handful of long strokes is also what a painter does.
    live = [i for i in range(w) if yb - top[i] > h * 0.10]
    if live:
        i0, i1 = live[0], live[-1]
        tol = max(1.0, h * 0.03)
        corners, a0 = [i0], i0
        for i in range(i0 + 1, i1):
            f = (top[i] - top[a0]) / max(1, i - a0)
            if any(abs(top[j] - (top[a0] + f * (j - a0))) > tol
                   for j in range(a0 + 1, i, max(1, (i - a0) // 6))):
                corners.append(i - 1)
                a0 = i - 1
        corners.append(i1)
        off = Noise(rng, 6, 2)
        i = 0
        while i < len(corners) - 1:
            n = rng.randint(1, 3)                           # 一笔 per stroke
            seg = [(x0 + j,
                    top[j] + (off.at(j / max(1, w - 1.0), 0.5) - 0.5)
                    * h * 0.055)
                   for j in corners[i:i + n + 1]]
            if len(seg) > 1:
                brush_path(cv, seg, max(0.9, h * 0.017), max(0.6, h * 0.009),
                           mix(dp, ink, 0.55), alpha * rng.uniform(0.40, 0.66))
            i += n + (1 if rng.random() < 0.3 else 0)       # 断 — the gap
    # 折带皴 — the name means *folded ribbon*, and it is literal: a flattish
    # shelf run with the side of the brush, then a short drop off its end,
    # then the next shelf. Radiating them out of the crest instead — which
    # is what the second draft did, since each stroke started at the profile
    # and ran downhill — gives the veins of a leaf.
    # 皴不可满 — a count, not a density. Scaled by pixel area it was a
    # hundred strokes on a big sheet and the rock came out shaded like a
    # woodcut; the number of marks a painter makes on one rock does not
    # depend on how large the paper is.
    for _ in range(rng.randint(13, 22)):
        i = rng.randrange(w)
        ty = top[i]
        if yb - ty < h * 0.20:
            continue
        yy = ty + (yb - ty) * rng.uniform(0.08, 0.46)
        ln = rng.uniform(0.07, 0.20) * w
        sg = rng.choice((1.0, -1.0))
        tilt = rng.uniform(-0.10, 0.10)
        # One segment, no joints at all. `brush_seg` is a round-capped
        # capsule and `brush_path` composites each of its segments on its
        # own, so the caps overlap at every joint and lay ink down twice
        # there: a finely sampled stroke grows a bead at every sample. The
        # earlier nine-point version was a visible string of them. A 折带
        # shelf wants to be ruled anyway — it is the side of the brush
        # dragged once — so the arc is dropped and the whole mark is
        # a single straight pull, with the 带 below it supplying the fold.
        ex = x0 + i + sg * ln
        ey = yy + sg * ln * tilt
        jj = int(max(0, min(w - 1, ex - x0)))
        if (ey < top[jj] + h * 0.04 or ey > bot[jj] - 1.0
                or yy < top[i] + h * 0.04 or yy > bot[i] - 1.0):
            continue
        # Widths in *pixels* here would be hairlines on a big sheet and
        # cudgels on a small one; a brush is a fraction of the form it is
        # drawing, so they go with the rock's own height.
        lw = max(0.6, h * 0.011)
        a = rng.uniform(0.13, 0.26)
        brush_seg(cv, x0 + i, yy, ex, ey,
                  lw * rng.uniform(0.7, 1.3), lw * rng.uniform(0.4, 0.8),
                  mix(dp, ink, 0.5), a)
        if rng.random() < 0.55:                     # 带 — the drop off the end
            dy = h * rng.uniform(0.05, 0.14)
            jj = int(max(0, min(w - 1, ex - x0)))
            if ey + dy < bot[jj] - 1.0:
                brush_seg(cv, ex, ey, ex + sg * ln * rng.uniform(0.04, 0.14),
                          ey + dy, lw * rng.uniform(0.6, 1.0), lw * 0.3,
                          mix(dp, ink, 0.5), a * 0.85)
    # 苔点 — moss on the crest, the last thing on any rock. 点苔宜聚: in two
    # or three tight clusters, not sprinkled evenly along the top, where
    # they come out as a row of beads stuck to the edge.
    for _ in range(int(moss * 2) + 1):
        i = rng.randrange(w)
        if yb - top[i] < h * 0.22:
            continue
        for _ in range(rng.randint(2, 5)):
            j = max(0, min(w - 1, i + int(rng.gauss(0.0, w * 0.028))))
            disc(cv, x0 + j + rng.uniform(-1, 1),
                 top[j] + rng.uniform(h * 0.03, h * 0.16),
                 max(0.7, h * 0.011) * rng.uniform(0.7, 1.4),
                 mix(dp, ink, 0.85), rng.uniform(0.30, 0.65), squash=0.7)
    return top


def grass(cv, rng, x, y, n, lt, dp, alpha, S, we):
    """兰草 — a few blades of grass where the culms enter the ground. They are
    not decoration: they say where the earth is, which a culm cut off by the
    paper's edge cannot."""
    for _ in range(n):
        a = -math.pi / 2 + rng.uniform(-0.8, 0.8) + we["sweep"] * 0.5 * \
            we.get("wind", 1.0)
        L = S * rng.uniform(0.045, 0.11)
        leaf(cv, rng, x + rng.uniform(-S * 0.05, S * 0.05), y, a, L, lt, dp,
             alpha * rng.uniform(0.5, 0.9),
             w=L * rng.uniform(0.030, 0.052),
             bend=rng.uniform(0.10, 0.30) * rng.choice((1.0, -1.0)),
             soft=0.3, k=rng.uniform(0.5, 0.9))


# --------------------------------------------------------------------------
# 天候 — the weather passes that are not just a modified angle
# --------------------------------------------------------------------------

def moon(cv, rng, x, y, r, ink):
    """月 — the moon is 留白. The paper is already the right colour; what a
    painter actually does is wash 淡墨 *around* it, and the disc appears.
    Painting a pale disc on paper gives you a hole in the sheet."""
    x0, y0, x1, y1 = cv.clip(x - r * 2.6, y - r * 2.6, x + r * 2.6, y + r * 2.6)
    for py in range(y0, y1):
        dy = py + 0.5 - y
        for px in range(x0, x1):
            d = math.hypot(px + 0.5 - x, dy) / r
            if d <= 1.0:
                continue
            a = 0.26 * math.exp(-(d - 1.0) * 1.9)
            if d < 1.08:                      # a crisp 水口 at the rim
                a *= 1.0 + (1.08 - d) * 6.0
            if a > 0.004:
                cv.blend(px, py, ink, a)


def rain(cv, rng, S, ink, n=60):
    """雨 — not drawn as drops. A sheet of rain is read from a few long
    slanted strokes of 淡墨 in the empty parts of the paper."""
    W, H = cv.w, cv.h
    a = math.pi * 0.5 + rng.uniform(0.12, 0.34) * rng.choice((1.0, -1.0))
    for _ in range(n):
        x = rng.uniform(-W * 0.1, W * 1.1)
        y = rng.uniform(-H * 0.05, H)
        L = S * rng.uniform(0.10, 0.34)
        aa = a + rng.gauss(0.0, 0.03)
        brush_path(cv, [(x, y), (x + math.cos(aa) * L, y + math.sin(aa) * L)],
                   rng.uniform(0.35, 0.8), rng.uniform(0.3, 0.6),
                   ink, rng.uniform(0.04, 0.11))


def snowfall(cv, rng, S, n):
    for _ in range(n):
        disc(cv, rng.uniform(0, cv.w), rng.uniform(0, cv.h),
             rng.uniform(0.8, 2.4), (255, 255, 253), rng.uniform(0.30, 0.75),
             soft=0.5)


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

TITLE_A = ["清风", "疏影", "月下", "雨后", "雪后", "幽篁", "新篁", "竹石",
           "晓烟", "石畔", "东风", "秋声", "晴窗", "碧玉", "绿云", "虚心"]
TITLE_B = ["图", "小景", "清趣", "写意", "图卷", "册页", "小品"]
ACTION = ["写生", "戏作", "写意", "拟古", "客窗戏墨", "南窗写竹", "并记"]
SEAL_WORDS = ["清风", "墨缘", "虚心", "有节", "君子", "竹友", "闲心", "写心",
              "石友", "听雨", "一枝", "安之"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into bamboo_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [v["cn"] for v in SCENES.values()]
             + [v["cn"] for v in WEATHER.values()]
             + [v["cn"] for v in STYLES.values()]
             + list(CLUSTERS)
             + [TIANGAN, DIZHI, "日月年写于图竹春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a blob baked for this vocabulary; fall back to the other two
    painters' (the three share most characters), then to no 题款 at all."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("bamboo_font", "gongbi_font", "shanshui_font"):
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
    """题款 — measure the block, then let the 留白 decide where it goes.

    In bamboo this matters more than in any other subject: the picture is
    half empty by design, and the writing is the counterweight that stops the
    emptiness from being merely unused. It still may not stand in the middle.
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
    r0 = min(W, H) * 0.18
    for fy in (0.30, 0.48, 0.66):
        occ.add(W * 0.5, H * fy, r0, ink=False)
    block = None
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.6)
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

def culm_xs(rng, n, lo, hi, gap):
    """Where the culms stand.

    One prohibition decides this and nothing else does: 忌等距 — evenly
    spaced culms are a fence, not a grove. Note that *maximising* the
    spacing, which is the right rule for leaf groups, is exactly the wrong
    rule here: the roomiest arrangement of five culms on a strip of paper is
    the fence. A grove is read as groups of two or three standing close with
    real paper between the groups, so the gaps are drawn from two populations
    — 聚, a companion about a culm away, and 散, the break between clumps.
    """
    xs = [0.0]
    for _ in range(n - 1):
        xs.append(xs[-1] + gap * (rng.uniform(0.9, 1.5) if rng.random() < 0.5
                                  else rng.uniform(2.6, 4.6)))
    span = xs[-1]
    room = (hi - lo) - span
    if room < 0.0:                       # too many for the band: squeeze
        k = (hi - lo) / max(1e-6, span)
        xs = [v * k for v in xs]
        room = 0.0
    # ...and not jammed against either end of the band. When there are
    # several culms the span fills it and this does nothing, but a scene
    # with one or two has room to slide and used to use all of it: the
    # subject ended up hard against the border with the whole sheet open
    # behind it, which reads as a crop of a larger painting rather than as
    # a painting. The 留白 wants to be on both sides of the subject, however
    # unequally.
    off = lo + room * rng.uniform(0.14, 0.86)
    return [off + v for v in xs]


def compose(cv, rng, scene_key, weather_key, style_key):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc = SCENES[scene_key]
    we = dict(WEATHER[weather_key])
    st = STYLES[style_key]
    gb.set_light(rng)
    ground = gb.paint_ground(cv, rng)
    occ = Occupancy(W, H)
    scale = S / 1000.0
    # 虚实 — one half of the sheet carries the bamboo, the other carries the
    # emptiness and the writing. Choosing the side *first* is what keeps the
    # composition from drifting into the middle; and the weather has to be
    # told about it, because wind is the one thing that can undo it. Blowing
    # outward it lays a grove that was already against its own border over
    # the border, and every branch on the outward side of every culm leaves
    # the paper with its leaves — so the sheet loses half its foliage and
    # keeps all of its emptiness. Blowing inward, the same grove leans out
    # across the open half and the leaves fly over it, which is what the
    # emptiness was left for.
    side = rng.choice((0, 1))
    we["wind"] = 1.0 if side else -1.0            # which way the weather runs
    node_ink = jitter(rgb(st["node"]), rng, 6)

    if we.get("veil"):
        veil(cv, we["veil"][0], we["veil"][1])
    if we.get("moon"):
        mr = S * rng.uniform(0.11, 0.16)
        mx = W * (rng.uniform(0.18, 0.34) if rng.random() < 0.5
                  else rng.uniform(0.66, 0.82))
        my = H * rng.uniform(0.13, 0.30)
        moon(cv, rng, mx, my, mr, dilute(I("dan"), 0.75))
        occ.add(mx, my, mr * 1.1)

    # The band stops short of the border it leans against. A culm standing
    # at 0.94 of the width has nowhere to put the branches that leave it on
    # that side: they run off the paper, their leaf groups are refused by
    # the margin rule in `cluster_sites`, and the culm keeps only the half
    # of its foliage that happens to face inward. Three culms packed into
    # the last tenth of the sheet lose most of their leaves that way, which
    # is how a scene with the right number of culms still came out bare.
    lo = (0.08 if side else 0.32) * W
    hi = (0.66 if side else 0.88) * W
    ybase = H * rng.uniform(0.92, 1.06)

    rock_top = None
    if sc["rock"] and rng.random() < sc["rock"]:
        # 石出纸外 — the rock's foot is below the paper's edge. Drawn whole,
        # with its own ground line inside the sheet, a rock this wide comes
        # out as a pancake lying on the floor; what a painter shows is the
        # upper mass rising out of the bottom border, and the eye supplies
        # the rest. It also means the rock never competes with the culm feet
        # for the same bit of paper.
        rw = W * rng.uniform(0.52, 0.82)
        rx = (lo + (hi - lo - rw) * rng.uniform(0.0, 1.0)
              if hi - lo > rw else lo - (rw - (hi - lo)) * rng.random())
        # Measured from the crest down, not from the foot up: the foot is
        # off the paper and what has to be right is where the top of the
        # mass sits — a fifth or a quarter of the sheet, no more, or the
        # rock stops being the bamboo's foil and becomes its subject.
        # 石宜横, 竹宜纵 — the rock has to lie *across* the picture. That is
        # the whole reason it is in it: everything else on the sheet is a
        # vertical, and a rock as tall as it is wide is one more. So the
        # visible mass is a sixth of the sheet high and half of it wide.
        ry = H * rng.uniform(0.83, 0.90)
        # The foot goes just below the border, not far below it. Push it a
        # long way down — the natural way to make sure it is off the paper —
        # and only the plateau of the profile clears the bottom edge: every
        # shelf and both flanks are below the border, and the rock is a slab.
        rh = (H - ry) + H * rng.uniform(0.02, 0.07)
        # 石分三面 needs a *range*: the body of the rock is 淡墨 and must
        # stay there, but if the dark stop of its ramp is diluted to match,
        # the whole mass comes out inside forty levels of grey and reads as
        # a flat piece of card with a line round it. So the pale stop stays
        # where it was and the dark stop is drawn nearly straight from the
        # stone — it is spent only in the band under the crest, in the 皴
        # and in the moss, which is where a painter's dark ink goes.
        rlt = plane_tone(st, "rock", 0.56 * we["k"])[0]
        rdp = plane_tone(st, "rock", 0.95 * we["k"])[1]
        rock_top = rock(cv, rng, rx, rx + rw, ry + rh, rh,
                        rlt, rdp, node_ink, alpha=0.70,
                        moss=0.8 if style_key != "朱竹" else 0.4)
        occ.add(rx + rw * 0.5, (ry + H) * 0.5, max(rw, H - ry) * 0.55)
        ybase = ry + (H - ry) * rng.uniform(0.15, 0.6)

    # --- the culms, plane by plane, back to front -------------------------
    planes = []
    for k, (n0, n1) in sc["culms"]:
        planes.append((k, rng.randint(n0, n1)))
    total = sum(n for _, n in planes) or 1
    xs_all = culm_xs(rng, total, lo, hi, W * 0.052)
    rng.shuffle(xs_all)
    lead_done = False
    tip_shown = False
    tips_by_plane = []
    drift = rng.uniform(-0.085, 0.085)          # 一势 — the grove's own lean
    # 折枝 always breaks. The one-in-five sheet that did not come out as a
    # single upright cane in a scene whose whole subject is a cut branch
    # crossing the paper — a thin vertical against the border with four
    # groups on it, and the scene's name a lie.
    broken = bool(sc.get("broken"))
    # 一势 — and on a 折枝 sheet the 势 is the diagonal, so *everything* on
    # the sheet has to lie along it. Only the principal was being broken
    # before; the pale culms behind it were still being stood up vertically
    # like a grove's, and a branch crossing the paper at forty-five degrees
    # behind two uprights is 忌井字 twice over — it also hands the eye two
    # kinds of plant, one of them tilted hard over while the others are
    # straight, which is exactly the fault it reads as. A 折枝 has no
    # uprights. The companions are the same cut diagonal, slid along the
    # paper and faintly off-parallel.
    br_side = rng.choice((0, 1))
    br_y0 = rng.uniform(0.72, 1.04)             # where the principal enters
    br_y1 = rng.uniform(0.04, 0.32)             # ...and where it leaves

    for pi, (k, n) in enumerate(planes):
        near = k >= 0.9
        plane = "near" if near else ("mid" if k >= 0.5 else "far")
        kk = k * we["k"]
        clt, cdp = plane_tone(st, "culm", kk)
        soft = 0.0 if near else (0.5 if plane == "mid" else 1.4)
        for _ in range(n):
            if not xs_all:
                break
            x = xs_all.pop()
            lead = near and not lead_done
            if lead:
                lead_done = True
            # 主从 — the leader is thicker, darker and more upright; the
            # others are drawn as its company, never as its equals.
            #
            # The width matters more than anything else on this list. A culm
            # is a *line* in this tradition: at paper scale about one part in
            # a hundred of the shorter side, which on a 900-pixel sheet is a
            # stroke some fourteen pixels across. Twice that — the first
            # draft's figure — and the picture becomes a photograph of
            # scaffolding poles, no matter how well the leaves are written.
            w0 = S * rng.uniform(0.0062, 0.0088) * (1.0 if near else
                                                    (0.84 if plane == "mid"
                                                     else 0.68))
            if lead:
                w0 *= sc["lead"] * 1.15
            start_a = -math.pi / 2
            if broken:
                # 折枝 — no ground and no foot: the culm crosses the sheet
                # from an edge, cut by the frame at both ends. The oldest
                # framing there is, and the one that needs the least paper.
                #
                # Aimed at the far edge rather than launched at a free
                # angle. A branch entering low on the left at sixty-six
                # degrees — the steep end of the old range — leaves through
                # the top having crossed two fifths of the width, and the
                # whole lower half of the sheet is then paper the subject
                # never reaches. The two ends of a 折枝 are both the frame:
                # it is a shallow diagonal right across the paper, and
                # fixing where it comes in and where it goes out says that
                # much more directly than guessing at the slope.
                #
                # The principal's own diagonal, shared: the companions get
                # the same entry and exit heights shifted bodily up or down
                # the sheet, which keeps them parallel to it rather than
                # across it.
                sh = 0.0 if near else rng.uniform(-0.40, 0.26)
                x = W * (-0.04 if br_side else 1.04)
                tx = W * (1.04 if br_side else -0.04)
                yb2 = H * (br_y0 + sh)
                ty = H * (br_y1 + sh)
                start_a = (math.atan2(ty - yb2, tx - x)
                           + (0.0 if near else rng.gauss(0.0, 0.055)))
                h = math.hypot(tx - x, ty - yb2) * rng.uniform(1.02, 1.18)
                ybase = yb2
            elif rng.random() < (0.92 if (near and not tip_shown) else 0.40):
                # 出梢 — the culm ends inside the paper, in a crown of its
                # own leaves. At least one culm in a painting must do this:
                # a sheet where every culm runs off both edges has no top
                # and no bottom, and reads as a fragment of wallpaper. The
                # height is chosen so the tip lands in the upper third,
                # where the eye expects the crown to be.
                h = (ybase - H * rng.uniform(0.07, 0.34)) * rng.uniform(0.98,
                                                                        1.02)
                tip_shown = True
            else:
                # 出纸 — and the rest are cut by the frame, which is what
                # says the grove continues past it.
                h = H * rng.uniform(1.06, 1.3) * (1.0 if near else 0.94)
            # 节 — the internodes are spaced by the culm's own *diameter*,
            # seven to eleven of them, which is what bamboo does and what
            # keeps the joints from reading as a ladder.
            nodes = max(3, int(h / (2.0 * w0 * rng.uniform(7.0, 11.0))))
            nodes = min(nodes, 13)
            # 一势之中有参差 — the group has to agree on a direction (a grove
            # of culms fanning every way is a haystack) while no two members
            # stand at exactly the same angle. So the lean is a shared drift
            # plus a private error — and the shared part has to be much the
            # bigger of the two, which is the opposite of what this did at
            # first. With the private error at ±0.17 two neighbours could
            # stand twenty degrees apart; over a culm the height of the
            # sheet that is two hundred pixels of divergence, so one or two
            # members of every grove struck off on their own and crossed
            # their neighbours at a wide angle, which is 忌井字 and reads as
            # a dropped handful of canes. ±0.05 keeps a grove a grove.
            #
            # A gaussian has no end to it, which is the catch: at σ = 0.042
            # most groves are tidy and one in six draws a two-sigma culm
            # five degrees out, and five degrees of *launch* becomes ten of
            # base-to-tip once the bow has worked on it. The complaint is
            # about that one culm, not about the other five, so what is
            # wanted is a distribution with an actual edge. A triangular
            # error — the sum of two uniforms — still favours the middle but
            # cannot produce an outlier at all.
            tilt = drift + (rng.random() + rng.random() - 1.0) * 0.050 \
                * (0.5 if lead else 1.0) \
                + we["sweep"] * 0.20 * we["wind"] * rng.uniform(0.94, 1.06)
            # The bow is shared the same way, and for the same reason: two
            # culms at one angle but with bows differing threefold part
            # company just as visibly as two at different angles. The
            # leader's own allowance is part of the same sum and has to be
            # read the same way: 主从 says the leader stands straighter than
            # its company, but at 0.7 of their bow it stands a third
            # straighter, and in 风 that is the leader leaving the grove
            # rather than leading it. A fifth is enough to be seen.
            bow = we["bow"] * rng.uniform(0.89, 1.12) * we["wind"] \
                * (0.82 if lead else 1.0)
            if broken:
                # a cut branch is never ruled — but a companion bends less
                # than the principal, or the two part company at the ends
                bow *= 2.1 if near else 1.3
            spine = culm_spine(rng, x, ybase, h,
                               start_a + (0.0 if broken else tilt),
                               w0, w0 * rng.uniform(0.52, 0.72), bow, nodes,
                               kink=0.85 if (broken and near) else 0.40)
            a_c = 0.95 if near else (0.72 if plane == "mid" else 0.46)
            draw_culm(cv, rng, spine, clt, cdp, node_ink, a_c, soft=soft,
                      k=kk, dry=0.35 if near else 0.0)
            occ.add_path([(p[0], p[1]) for p in spine], w0 * 1.6, w0, step=2)

            # --- 生枝 — branches off the nodes, alternating sides ---------
            tips = []
            llt, ldp = plane_tone(st, "leaf", kk)
            nb = rng.randint(*sc["twigs"]) + (2 if lead else 0)
            # 竹无叶在下 — the foot of a standing culm is bare, so the lower
            # quarter of it carries no branches. A cut branch has no foot:
            # both its ends are the frame, and it is leafed the whole way. The
            # quarter was being charged to it as well, and because a 折枝
            # branch is drawn a third longer than the sheet that quarter is
            # most of the paper the branch crosses — so every 折枝 came out
            # with its leaves bunched in one half and a long bald stretch
            # running out to the border. The bald end is the half the eye
            # reads as unfinished, not as 留白.
            root = 0.06 if broken else 0.26
            cand = [i for i in range(1, len(spine) - 1)
                    if i >= len(spine) * root
                    and -H * 0.02 < spine[i][1] < H * 1.02
                    and -W * 0.02 < spine[i][0] < W * 1.02]
            if cand:
                sgn = rng.choice((1.0, -1.0))
                picks = sorted(rng.sample(cand, min(nb, len(cand))))
                for j, i in enumerate(picks):
                    bx, by, bw, ba = spine[i]
                    s2 = sgn * (1.0 if j % 2 == 0 else -1.0)
                    # The side is `s2` and stays `s2` — passed in rather
                    # than inferred from the launch angle, so a leaning
                    # culm cannot flip it. The droop and the wind are left
                    # to the twig itself, which applies them to its pitch.
                    a = ba + s2 * rng.uniform(0.55, 1.15)
                    draw_twig(cv, rng, bx + math.cos(ba) * bw * 0.4,
                              by + math.sin(ba) * bw * 0.4, a,
                              S * rng.uniform(0.08, 0.17), max(0.7, bw * 0.52),
                              clt, cdp, a_c, tips, droop=we["droop"],
                              soft=soft, sweep=we["sweep"] * we["wind"],
                              out=s2)
            # 梢 — a culm whose top falls inside the paper must end in leaves.
            # Bare, it is a cut pole; bamboo carries its own crown.
            tx, ty, tw, ta = spine[-1]
            if 0.0 < tx < W and H * 0.03 < ty < H * 0.92:
                # The crown leans. A tip is the one site where the twig's
                # angle gives no side at all — the culm points straight up —
                # so the side is chosen and the group is turned onto it. A
                # crown written symmetrically about a vertical is a
                # weather-vane, and leaves standing on end are 叶朝天.
                #
                # Turned *from the vertical*, not added to the culm's own
                # tip angle: in 风 the bow accumulates over a tall culm
                # until the tip is leaning 40° already, and three quarters
                # of a radian added to that lands straight back on the
                # vertical. It is the pitch that has to be set.
                to = rng.choice((1.0, -1.0))
                tips.append((tx, ty,
                             _aim(_hold(_pitch(ta) + rng.uniform(0.55, 0.95),
                                        -0.95, 0.30), to), tw, to))
            tips_by_plane.append((plane, kk, soft, a_c, llt, ldp, tips, lead))

    # --- 布叶 — leaves last, and chosen so the groups keep their paper ----
    LR = S * 0.105 * sc["leaf"]
    flown = []                  # where a torn-off leaf could have come from
    # 疏可走马，密不透风 — but the 密 is inside a group, never between them.
    # How many groups a sheet carries depends on the size of the sheet and
    # the size of a group and on nothing else — certainly not on how many
    # twigs happened to get drawn, which is what decided it before: every
    # near-plane twig end was given a formula, so a 丛竹 of nine culms came
    # out with fifty groups and went solid across the top. A budget, then,
    # and it is resolution-independent because both terms are areas.
    # 折枝 is the exception the budget needs: the sheet's whole subject is
    # one branch, so the groups that would have been spread over a grove all
    # belong on it, and at the plain budget it comes out as a bare cane.
    total = max(4, int(W * H * sc.get("dense", 1.0) / (LR * LR * 9.0)))
    # 主从 again: the near plane is the subject and gets most of the budget,
    # the back planes are there to say the grove is deep and get a little.
    SHARE = {"near": 0.62, "mid": 0.26, "far": 0.26}
    left = total
    marks = []                  # every group already settled, all planes
    chosen = []
    for plane in ("near", "mid", "far"):
        rs = [r for r in tips_by_plane if r[0] == plane and r[6]]
        if not rs or left <= 0:
            continue
        # Each plane's share, spread over its culms — and at least one group
        # per culm whatever the arithmetic says, because a near culm with no
        # leaves on it at all is not bamboo, it is a pole.
        q = min(left, max(len(rs), int(round(total * SHARE[plane]))))
        per = max(1, int(math.ceil(q / len(rs))))
        for r in rs:
            if left <= 0:
                break
            got, marks = cluster_sites(
                rng, r[6], min(per + (1 if r[7] else 0), left), LR * 0.62,
                sep=0.92 if plane == "near" else 1.0, sheet=(W, H),
                taken=marks)
            left -= len(got)
            chosen.append((r, got))
    # 远近 — drawn back to front whatever order they were chosen in, so a
    # near group always lies over a far one and never the other way round.
    chosen.sort(key=lambda c: c[0][1])
    for (plane, kk, soft, a_c, llt, ldp, tips, lead), got in chosen:
        R = LR * (1.0 if plane == "near" else
                  (0.92 if plane == "mid" else 0.85))
        for x, y, a, w, out, cs, crowd in got:
            # 层次 — a group that lands on top of another is pulled back in
            # tone instead of merging with it. Overlap then reads as depth.
            k2 = kk * (1.0 - 0.30 * crowd)
            lt2, dp2 = plane_tone(st, "leaf", k2)
            leaf_cluster(cv, rng, x, y, a, R, lt2, dp2,
                         a_c * (1.0 - 0.18 * crowd), we, plane=plane,
                         scale=cs * (1.08 if lead else 1.0), soft=soft,
                         k=k2, snow=we.get("snow", 0.0), out=out)
            occ.add(x, y, R * cs * 0.8)
            if plane != "far":
                flown.append((x, y, R * cs))

    # --- 坡草 — grass at the feet, and the ground line --------------------
    if sc["grass"] and rng.random() < sc["grass"]:
        glt, gdp = plane_tone(st, "leaf", 0.55 * we["k"])
        for _ in range(rng.randint(1, 3)):
            gx = rng.uniform(lo, hi)
            gy = (H * rng.uniform(0.90, 0.99) if rock_top is None
                  else H * rng.uniform(0.93, 1.0))
            grass(cv, rng, gx, gy, rng.randint(2, 5), glt, gdp, 0.7, S, we)

    # --- 天候 — the passes that go over the top ---------------------------
    if we.get("rain"):
        rain(cv, rng, S, dilute(I("dan"), 0.55), int(50 * scale) + 24)
    if we.get("snow"):
        snowfall(cv, rng, S, int(70 * scale) + 30)
    if we.get("fly") and flown:
        # 风 — leaves already torn loose and travelling. Scattered one here
        # and one there across empty paper they read as specks of dirt; what
        # makes them read as *wind* is that they go in a train, all lying the
        # same way, leaving from somewhere a leaf could actually have left
        # from. So one group is chosen as the source and the train runs
        # downwind of it.
        flt, fdp = plane_tone(st, "leaf", 0.72 * we["k"])
        wind = 1.0 if we["wind"] > 0 else -1.0
        # ...and from the group that has paper downwind of it. Picking the
        # source at random throws away half the paintings: the group chosen
        # is as often as not the one nearest the edge the wind is blowing
        # toward, the train leaves the sheet on its first step, and nothing
        # is drawn at all.
        flown.sort(key=lambda g: (W - g[0]) if wind > 0 else g[0])
        wx, wy, wr = flown[-1 - rng.randrange(min(2, len(flown)))]
        ca = rng.uniform(-0.34, 0.26)
        dx, dy = math.cos(ca) * wind, math.sin(ca)
        px, py = wx + dx * wr * 1.5, wy + dy * wr * 1.5
        for i in range(rng.randint(*we["fly"]) + 1):
            L = S * rng.uniform(0.052, 0.082) * (1.0 - 0.10 * i)
            px += dx * S * rng.uniform(0.07, 0.14)
            py += dy * S * rng.uniform(0.07, 0.14) - S * rng.uniform(0.0, 0.035)
            if not (S * 0.02 < px < W - S * 0.02 and
                    S * 0.02 < py < H - S * 0.02):
                break
            leaf(cv, rng, px, py,
                 ca + rng.gauss(0.0, 0.16) + (0.0 if wind > 0 else math.pi),
                 L, flt, fdp, rng.uniform(0.42, 0.66) * (1.0 - 0.12 * i),
                 bend=rng.uniform(-0.16, 0.16), k=0.72, soft=0.25)
            occ.add(px, py, L * 0.5)

    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))]) if rng.random() < 0.62 \
        else (we["cn"] + TITLE_B[rng.randrange(len(TITLE_B))])
    inscribe(cv, rng, occ, S, title)
    return title, ground


def age(cv, rng):
    """A last breath of ink dust, as both the other painters do."""
    W, H = cv.w, cv.h
    for _ in range(int(W * H * 0.00009)):
        disc(cv, rng.uniform(0, W), rng.uniform(0, H),
             rng.uniform(0.6, 1.8), I("dan"), rng.uniform(0.05, 0.24))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def paint(path, size=(900, 1300), seed=None, scene=None, weather=None,
          style=None, quiet=False):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    wk = weather or sorted(WEATHER)[rng.randrange(len(WEATHER))]
    yk = style or sorted(STYLES)[rng.randrange(len(STYLES))]
    for key, tbl, what in ((sk, SCENES, "scene"), (wk, WEATHER, "weather"),
                           (yk, STYLES, "style")):
        if key not in tbl:
            raise SystemExit("unknown %s %r; try --list" % (what, key))

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, wk, yk)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-24s %s  %s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk, WEATHER[wk]["cn"],
                 STYLES[yk]["cn"], ground, seed, size[0], size[1],
                 time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese bamboo paintings (墨竹).")
    ap.add_argument("-o", "--out", default="bamboo.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="丛竹|折枝|竹石|新篁")
    ap.add_argument("--weather", help="晴|风|雨|雪|月")
    ap.add_argument("--style", help="墨竹|朱竹|翠竹")
    ap.add_argument("--size", default="900x1300", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for k, v in SCENES.items():
            print("  %-6s %s" % (k, v["en"]))
        print("天候 (weather):")
        for k, v in WEATHER.items():
            print("  %-4s %-8s %s" % (k, v["cn"], v["en"]))
        print("设色 (colour):")
        for k, v in STYLES.items():
            print("  %-6s %s" % (k, v["en"]))
        print("叶诀 (leaf formulas):")
        print("  " + "  ".join("%s(%d叶)" % (k, len(v))
                               for k, v in CLUSTERS.items()))
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("墨竹 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.weather, a.style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
