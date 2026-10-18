#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bird.py — 随机生成中国写意翎毛图 / Random Chinese ink-bird paintings (翎毛).

翎毛 is the oldest 专科 in the tradition after 山水, and the one with the most
explicit 画诀, because a bird is the subject where the viewer already knows
the answer. Nobody can say whether a mountain is drawn correctly. Everybody
can see that a sparrow's head is too small.

The 写意 form this module follows is 齐白石's and 八大山人's, not 吕纪's: a
small bird is **seven marks and five lines**, laid wet, and the painter's own
instructions are a list of which mark goes where —

    一笔点头，两笔成背   the head is one touch, the back is two
    头如卵，背如弓       the head is an egg, the back is a drawn bow
    嘴如钳，眼如豆       the bill is pincers, the eye is a bean
    腹宜虚               the belly is left empty — it is paper, never paint
    翅分三停             the wing is three ranks: 覆羽, 大覆羽, 飞羽
    尾如剪               the tail is scissors — a few flat feathers, fanned,
                         never a plume
    爪要抓枝             the claws must *grip*. A bird resting on a branch
                         instead of holding it is the one fault every viewer
                         catches without being able to name it.
    眼有神               and the eye is where the bird is alive or dead. 八大
                         painted it as a ring with the pupil jammed into one
                         corner — 白眼向人 — and four hundred years later that
                         is still the most expressive mark in the tradition.

The structural problem this module exists to solve is **coherence across
separately-painted masses**. Every sibling painter's subject is one thing:
a fish is one spindle, a crab is one press, a berry is one touch, a branch is
one stroke. A bird is four masses — head, body, wing, tail — that have to
read as one creature, and the three ways that fails are all visible at a
glance: the head sits on the body like a ball on a sack, the wing floats off
the flank as a separate leaf, and the back line changes direction where two
strokes meet. Painting them well individually does not fix any of it.

So nothing here is positioned by hand. `Pose` is one body axis and nine
numbers, and every part — the head's pivot, the wing root, the tail root, the
leg root, the eye, the bill — is read off it, in body-local units, before any
ink is laid. The body's silhouette is computed once as a closed polar outline
and then used **twice**: once as the `rad` of the single `touch` that lays the
whole 没骨 mass, and once as the path of the 背线, the single dark stroke that
runs bill → crown → nape → mantle → rump. Those two cannot disagree, because
they are the same function. That is the module's one idea, and it is the same
idea as pavilion's projection: a subject with parts needs somewhere outside
the parts to keep them honest.

Two things are borrowed rather than rebuilt. The belly is **paper**, which
`touch` gives for free — its `tone` returns coverage as well as shade, so
driving coverage to zero on the ventral flank leaves the sheet untouched
rather than painting it pale. And the bird sits *in front of* the branch it
holds, which under this family's multiply compositing is impossible; so the
body, head, wing and tail each `lift` the paper back out of the sheet before
they are painted, exactly as 梅's blossoms lift the wood out from behind
themselves. The claws are then drawn **after** the wood, over it, which is
what makes them grip.

Usage
-----
    python3 bird.py                              # one random painting
    python3 bird.py -n 6                         # six of them
    python3 bird.py --scene 孤禽 --species 八哥
    python3 bird.py --style 赭墨 --air 雪霁 -s 7
    python3 bird.py --size 900x1300 -o scroll.png
    python3 bird.py --list                       # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Occupancy, BrushFont, clamp01, disc, mix, rgb,
                    paint_column, paint_seal, TIANGAN, DIZHI)
# 一家之墨 — the same ink stone, the same 兑水 rule, the same full-sheet veil
# and the same 留白 moon as 竹, 兰, 葡萄, 蟹 and 梅.
from bamboo import I, dilute, veil, moon, snowfall
from orchid import sweep                      # a brush dragged along a line
from grape import touch                       # one touch: polar rim, ink field
from crab import press                        # one side-loaded 侧锋 press
# The perch is a plum branch in all but name, so it *is* one: 梅's wood is the
# family's only 鹿角 wood, and a bird needs exactly that — something that
# breaks direction at a joint and is thick enough at the near end to be
# gripped. `Reserve` is the occlusion engine; `drag` the tapering-stroke
# wrapper over `sweep` that most linework here wants.
from mei import (Reserve, drag, wood_path, bough, cun, moss, blot, flyback,
                 snow_on, _turn)

TAU = math.tau


# --------------------------------------------------------------------------
# 设色 — the four schools.
#
# 翎毛 splits on exactly one question: whether the bird is ink or feathers.
# 水墨 is the orthodox answer and the others are departures that keep the ink
# drawing underneath and add a *breath* of colour over it (淡彩 proper), which
# is why every entry carries the same `plume` ramp shape and differs mostly in
# where its `accent` lands. Painting a bird in its field-guide colours is
# 标本, not 写意: the sparrow below is browner than ink and nothing like as
# brown as a sparrow.
#
# `plume` are the two stops of the plumage ramp, pale flank and loaded core;
# `accent` a single hue washed over the mantle, the flank and the wing
# coverts at `accent_a`; `bill` and `foot` the horn and the leg; `leaf` the
# foliage on the perch, which has to belong to the same sheet.
# --------------------------------------------------------------------------

STYLES = {
    "水墨": dict(cn="水墨", en="plain ink — 墨分五色, no colour at all",
                 plume=("#8A8276", "#1F1B18"), accent=None, accent_a=0.0,
                 bill="#1C1917", foot="#3A352E",
                 leaf=("#9A9484", "#3A3730"), wood=("#8E8678", "#241F1B")),
    "淡彩": dict(cn="淡彩", en="light colour over an ink drawing (淡彩)",
                 plume=("#8E8678", "#241F1B"), accent="#A8703E", accent_a=0.20,
                 bill="#2A241E", foot="#5C4026",
                 leaf=("#A4AE88", "#47582F"), wood=("#8A8276", "#262220")),
    "赭墨": dict(cn="赭墨", en="ochre and ink — 赭石 on the back (麻雀本色)",
                 plume=("#A08A72", "#33251B"), accent="#8C5A30", accent_a=0.30,
                 bill="#2E241C", foot="#69472A",
                 leaf=("#AFA377", "#5A4C28"), wood=("#95866F", "#2C2219")),
    "花青": dict(cn="花青", en="indigo and ink — 花青 on the wing (翠羽)",
                 plume=("#808E98", "#1C242E"), accent="#2C5C78", accent_a=0.30,
                 bill="#22262A", foot="#4C4230",
                 leaf=("#8FA294", "#2E4438"), wood=("#8A8A82", "#23231F")),
}


# --------------------------------------------------------------------------
# 禽 — the four birds.
#
# All four are the same construction with different numbers, which is the
# point: the difference between a sparrow and a kingfisher is not anatomy, it
# is five ratios, and a generator that cannot say which five has not
# understood either bird. Lengths below are fractions — `body` of the sheet's
# short side, everything else of `body` — except `head`, which is a fraction
# of the body *length* and is the single most sensitive number in the file.
# The eye's tolerance for the head of a known animal is about ten per cent;
# 游鱼's report measured it and 翎毛 is worse, because a bird's head carries
# the eye and the bill and a viewer reads all three together.
#
# `stout` is the body's depth over its length — a kingfisher is a brick and a
# bulbul is a knife. `bill` is length over head diameter and `taper` how fast
# it closes: a conical finch bill is short and wide, a kingfisher's is longer
# than its own skull and barely narrows. `crest` raises the crown feathers,
# `nape` lifts a pale patch out of the hood (白头翁 is named for it), `cap`
# darkens the crown and the nape, and `tone` is how dark the whole bird runs
# against the style's ramp.
#
# `cap` earns its place because of what a finished sheet looked like without
# it. Every value on the bird except the bill, the eye and the tail landed
# between a third and two thirds of the scale: a correctly drawn, correctly
# graded, *pale* animal, which at arm's length is a grey plush toy on a
# branch. Nearly every small bird a painter paints has a dark head, and the
# dark head is the thing the eye reads first — before the bill, before the
# posture. 墨分五色 is not a property of a wash, it is a property of the
# sheet, and the sheet needs its 浓墨 where the subject keeps it.
#
# `belly` is how far up the flank the paper is allowed to come — 腹宜虚 is a
# rule about *small brown birds*, and applying it to a myna gives you a black
# bird with a hole in it. One number, and 八哥 stays inked almost to the vent
# while 麻雀 keeps its whole underside as sheet.
#
# `fan` is half the angle the tail opens through, and it was set from the
# ornithology rather than from the brush: a perched bulbul really does hold
# its tail shut to within a tenth of a radian. Drawn, that is five presses
# 0.88 of a body long sharing one centreline at alpha 0.70, and five
# multiplies of the same near-black ink over the same pixels is not a closed
# tail, it is a **solid black dagger** with one straight edge — measured off
# a finished sheet, the darkest and hardest mark in the module, and the thing
# an ablation finally pinned down after the primaries had been blamed for it
# twice. 尾如剪 is a rule about *edges*: what says tail is the paper between
# the blades, so the fan has to be opened until the brush can get between
# them, and the ordering between the four species is what the table is
# really for. (`_tail` then pays for the opening with one base mark, so the
# fan is still one object rather than a comb.)
# --------------------------------------------------------------------------

BIRDS = {
    "麻雀": dict(cn="麻雀", en="tree sparrow — round, short-tailed, 齐白石's",
                 body=(0.172, 0.206), stout=0.80, head=0.50,
                 bill=0.42, taper=0.72, gape=0.30,
                 tail=0.54, tail_n=4, fan=0.30, notch=0.18,
                 wing=0.66, prim=4, leg=0.17, crest=0.05, nape=0.0,
                 cap=0.58, tone=0.84, cheek=0.55, belly=1.00, season="秋"),
    "八哥": dict(cn="八哥", en="crested myna — all black, 八大山人's bird",
                 body=(0.198, 0.232), stout=0.73, head=0.44,
                 bill=0.56, taper=0.80, gape=0.26,
                 tail=0.66, tail_n=5, fan=0.23, notch=0.06,
                 wing=0.74, prim=5, leg=0.21, crest=0.80, nape=0.0,
                 cap=0.0, tone=1.00, cheek=0.0, belly=0.40, season="冬"),
    "白头翁": dict(cn="白头翁", en="light-vented bulbul — slim, long-tailed",
                   body=(0.178, 0.212), stout=0.70, head=0.46,
                   bill=0.46, taper=0.74, gape=0.28,
                   tail=0.88, tail_n=5, fan=0.18, notch=0.04,
                   wing=0.62, prim=4, leg=0.17, crest=0.22, nape=0.72,
                   cap=0.84, tone=0.80, cheek=0.30, belly=0.84, season="春"),
    # 翠鸟 is the one species whose dials were set from the bird and not
    # from the *painting* of it. A real kingfisher's tail is three tenths
    # of its body and it is genuinely stubby — but three blades that short
    # painted at the palest tone in the table do not read as a short tail,
    # they read as a wart on the rump, which is what nine sheets of 双栖
    # showed. 0.46 is still the shortest of the four by a wide margin and
    # it is unmistakably a tail. The tone goes up for the same reason: at
    # 0.76 the bird was the one pale sheet in the set.
    "翠鸟": dict(cn="翠鸟", en="kingfisher — big head, dagger bill, short tail",
                 body=(0.158, 0.188), stout=0.84, head=0.56,
                 bill=1.05, taper=0.94, gape=0.18,
                 tail=0.46, tail_n=4, fan=0.24, notch=0.0,
                 wing=0.56, prim=4, leg=0.11, crest=0.18, nape=0.0,
                 cap=0.70, tone=0.86, cheek=0.62, belly=0.80, season="夏"),
}


# --------------------------------------------------------------------------
# 章法 — the four framings.
#
# `birds` is how many; `entry` where the wood comes in; `perch` the band of
# the sheet the bird is allowed to sit in, measured in sheet heights; `lean`
# the body's tilt off horizontal (head up is positive) and `hunch` how far
# the neck is pulled in. `glare` is 八大's eye.
#
# 孤禽 is the one that is not a variation. It is a whole composition and a
# whole temperament: one hunched bird, low, on a stump, with four fifths of
# the sheet left as nothing at all. It is also the scene that tests whether
# 留白 in this family is real, because there is almost no ink on it to hide
# behind.
# --------------------------------------------------------------------------

SCENES = {
    "栖枝": dict(cn="栖枝", en="one bird on a bough across the sheet",
                 birds=1, entry="side", perch=(0.44, 0.64), w0=(0.009, 0.014),
                 L=(0.74, 0.96), segs=(3, 4), twigs=(3, 5), leaves=1.0,
                 lean=(0.16, 0.36), hunch=(0.20, 0.45), glare=0.0, old=0.45),
    "双栖": dict(cn="双栖", en="a pair on one bough — 呼应, never parallel",
                 birds=2, entry="side", perch=(0.42, 0.62), w0=(0.010, 0.015),
                 L=(0.80, 1.00), segs=(3, 5), twigs=(3, 5), leaves=0.9,
                 lean=(0.14, 0.32), hunch=(0.18, 0.40), glare=0.0, old=0.50),
    "孤禽": dict(cn="孤禽", en="八大 — one hunched bird on a stump, 留白",
                 birds=1, entry="foot", perch=(0.66, 0.80), w0=(0.026, 0.038),
                 L=(0.30, 0.44), segs=(2, 3), twigs=(0, 0), leaves=0.0,
                 lean=(0.16, 0.38), hunch=(0.70, 1.00), glare=1.0, old=1.00),
    "俯枝": dict(cn="俯枝", en="a branch down from the top edge, the bird high",
                 birds=1, entry="top", perch=(0.32, 0.52), w0=(0.009, 0.013),
                 L=(0.72, 0.94), segs=(3, 4), twigs=(3, 6), leaves=1.0,
                 lean=(-0.06, 0.14), hunch=(0.25, 0.50), glare=0.0, old=0.35),
}


# --------------------------------------------------------------------------
# 态 — what the bird is *doing*, which is a different question from where it
# is sitting.
#
# The dials to do this with were all there — `turn` swings the head over the
# shoulder, `dip` nods it, `hunch` draws the neck in, `tail_drop` cocks the
# tail — and the composer was reading five of them out of five independent
# uniforms. That is not variety, it is noise: independent small jitters on
# five axes average to the middle of all five, so nine sheets came out as
# nine copies of one level, mildly hunched, straight-ahead bird with slightly
# different numbers. A living posture is *correlated* — a bird that looks
# down also lowers its breast and lifts its tail, a bird that calls stretches
# its neck and raises its bill together — and correlations cannot come out of
# independent draws at any width.
#
# So the attitudes are named and the dials move in the groups they belong to.
# `turn` and `dip` are absolute (they are the head's own two freedoms);
# `hunch`, `lean`, `fluff` are **offsets** on what the scene and the weather
# already decided, so 孤禽 stays hunched and 雪霁 stays puffed whatever the
# bird is doing. `w` is how often it comes up against the others.
#
# Sign note, because it is the opposite of what the name suggests: `dip` is
# added to the bill's body-local polar angle, and +x is the *tail*, so a
# positive `dip` lifts the bill and a negative one drops it.
# --------------------------------------------------------------------------

MOODS = {
    "平栖": dict(cn="平栖", en="at rest, level — the default perched bird",
                 w=1.00, turn=(0.00, 0.00), dip=(-0.10, 0.20),
                 hunch=(0.00, 0.10), lean=(0.00, 0.00),
                 tail=(0.12, 0.30), fluff=1.00),
    "回首": dict(cn="回首", en="looking back over the shoulder — the liveliest",
                 w=0.52, turn=(0.70, 0.96), dip=(-0.12, 0.22),
                 hunch=(-0.10, 0.04), lean=(0.02, 0.12),
                 tail=(0.04, 0.22), fluff=0.98),
    "仰首": dict(cn="仰首", en="neck out, bill up — 鸣, about to call",
                 w=0.36, turn=(0.00, 0.16), dip=(0.46, 0.80),
                 hunch=(-0.30, -0.14), lean=(0.04, 0.18),
                 tail=(0.18, 0.42), fluff=0.88),
    "俯啄": dict(cn="俯啄", en="head down, watching the ground",
                 w=0.30, turn=(0.00, 0.12), dip=(-0.74, -0.44),
                 hunch=(-0.22, -0.06), lean=(-0.18, -0.04),
                 tail=(-0.14, 0.06), fluff=0.92),
    "缩颈": dict(cn="缩颈", en="head drawn in, feathers up — a cold bird",
                 w=0.44, turn=(0.00, 0.00), dip=(-0.06, 0.18),
                 hunch=(0.26, 0.46), lean=(-0.06, 0.04),
                 tail=(0.16, 0.38), fluff=1.20),
    "侧顾": dict(cn="侧顾", en="half a turn — looking out of the sheet",
                 w=0.30, turn=(0.24, 0.44), dip=(-0.16, 0.24),
                 hunch=(-0.06, 0.08), lean=(0.00, 0.10),
                 tail=(0.10, 0.30), fluff=1.02),
}


# 风致 — one air for the whole sheet, as every painter here has one weather.
# `fluff` is the single number that says how cold it is: a bird in the cold
# puffs its contour feathers and pulls its head in, so one dial widens the
# silhouette, softens its rim and deepens the hunch at once. That is a fact
# about birds and it does more for the picture than any amount of weather
# painted around them.
AIR = {
    "春晴": dict(cn="春晴", en="clear spring — the bird is sleek and alert",
                 fluff=0.18, hunch=0.00, wet=1.00, dry=0.28, k=1.00,
                 leaves=1.00, lean=0.00),
    "清秋": dict(cn="清秋", en="autumn — thin leaves, dry brush",
                 fluff=0.34, hunch=0.10, wet=0.88, dry=0.44, k=1.00,
                 leaves=0.45, lean=0.04),
    "疏雨": dict(cn="疏雨", en="light rain — 缩颈, everything softened",
                 fluff=0.62, hunch=0.30, wet=1.28, dry=0.12, k=0.92,
                 leaves=0.70, lean=0.02, veil=("#8E96A0", 0.085)),
    "雪霁": dict(cn="雪霁", en="after snow — the bird is a ball of feathers",
                 fluff=1.00, hunch=0.46, wet=1.10, dry=0.20, k=0.90,
                 leaves=0.08, lean=0.00, veil=("#9AA2AC", 0.075), snow=1.0),
}


# --------------------------------------------------------------------------
# 势 — the pose, and the one idea in this file.
#
# A bird is four masses and they have to agree. The agreement is not something
# you can add afterwards by drawing carefully; it has to exist before the
# first mark, as a frame the marks are read out of. So every length below is
# in **body-local units** — origin at the centre of the body, +x from the head
# toward the tail, +y toward the belly — and `L2S` is the only place the sheet
# is mentioned. Change `lean` and the whole bird tilts, bill, claws, tail and
# the dark edge down its back, because none of them know where they are except
# through this class.
#
# The sign gymnastics are worth one paragraph, because they are the only part
# that is fiddly. A bird faces left or right; mirroring it is one flag,
# `side`. A body-local direction at polar angle ψ comes out on the sheet at
# `th + side·ψ`, and `touch` and `Reserve.lift` both want their outline as a
# function of *their* local angle φ — so every radius function here is written
# in body-local ψ and handed over as `lambda φ: R(side·φ)`. Getting that
# backwards mirrors the bird's silhouette against its own limbs, which looks
# exactly like a bird and is wrong in a way that takes an hour to see.
# --------------------------------------------------------------------------

def _ease(t):
    """`clamp01` with the corners taken off.

    Every tonal field on this bird is a clamped linear ramp — 背如弓 is
    `(0.58 - Y) / 1.24` and 腹宜虚 is a waterline — and a clamped ramp is
    not smooth where it saturates. The value is continuous there and its
    *slope* is not, which on a mass as evenly shaded as a flank the eye
    reads as a **fold**: measured at eleven diameters on all four species,
    a ruled straight crease running from under the gape down across the
    throat, and a second one over the crown. Nothing is drawing it. It is
    the line `Y = 0.64 - 1.22` seen edge-on.

    Smoothstep lands the ramp flat at both ends, so the wash arrives at
    the spine and at the belly without a seam. It is the same curve in the
    middle to within a few percent, so no tone has to be retuned.
    """
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def _polar(poly, na=192):
    """Radius table of a star-shaped polygon about the origin.

    `poly` is a closed ring of (x, y) in body-local pixels. Returns `na`
    radii at equally spaced angles, which is what `touch` and `lift` both
    consume. A polygon is a far easier thing to *design* a bird's outline
    in than a Fourier series, and a radius table is a far easier thing to
    render; this is the half-page that lets both be true.
    """
    out = [0.0] * na
    n = len(poly)
    for k in range(na):
        a = k * TAU / na
        dx, dy = math.cos(a), math.sin(a)
        best = 0.0
        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]
            ex, ey = x1 - x0, y1 - y0
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            # t·d == edge start + s·e, by Cramer on the 2x2
            s = (dy * x0 - dx * y0) / den
            # Closed on both ends *and* slack by an epsilon. A ray that
            # passes through a vertex meets two edges at s = 1 and s = 0,
            # and in floating point it can land at 1 + 1e-16 on the first
            # and -1e-16 on the second — missing the polygon entirely and
            # leaving a 0 in the table. See the repair below for what that
            # draws.
            if s < -1e-9 or s > 1.0 + 1e-9:
                continue
            t = (x0 * ey - ex * y0) / den
            if t > best:
                best = t
        out[k] = best
    # 补漏 — a zero in this table is never a shape, it is always a miss.
    # The polygon is closed and contains the origin, so every ray must hit
    # it; one that does not has fallen through the crack at a vertex. Left
    # alone, the three smoothing passes below spread that single zero into
    # a **V-shaped notch about seven samples wide and a third of the
    # radius deep** — and what that draws, on a 翠鸟's big head at eight
    # diameters, is a cleft in the skull with two straight walls meeting
    # in a point, as if the bird had been painted in two overlapping
    # strokes. It survived six ablations (背线, 喉线, 冠, 嘴, 翅, 丝毛) and
    # a two-way ablation of the two masses, because it is not a mark: it
    # is a hole in the outline all three consumers share. So heal it here,
    # where the outline is built, rather than anywhere it is used.
    miss = [k for k in range(na) if out[k] <= 0.0]
    if miss and len(miss) < na:
        for k in miss:
            a = b = None
            for j in range(1, na):
                if a is None and out[(k - j) % na] > 0.0:
                    a = (j, out[(k - j) % na])
                if b is None and out[(k + j) % na] > 0.0:
                    b = (j, out[(k + j) % na])
                if a and b:
                    break
            out[k] = (a[1] * b[0] + b[1] * a[0]) / (a[0] + b[0])
    # 圆 — and the polygon is not. Thirteen vertices sampled by ray-casting
    # give thirteen dead-straight chords, and a 没骨 mass painted through
    # them has visible flats on its back that no amount of feathering
    # hides. Three circular [1,2,1] passes cost nothing and are the whole
    # reason the outline can go on being written as a readable polygon.
    for _ in range(3):
        out = [(out[k - 1] + 2.0 * out[k] + out[(k + 1) % na]) * 0.25
               for k in range(na)]
    return out


def _at(tbl, a):
    """Interpolate a radius table at an arbitrary angle."""
    na = len(tbl)
    f = (a % TAU) * na / TAU
    i = int(f)
    g = f - i
    if i >= na:
        i -= na
    j = i + 1
    if j >= na:
        j = 0
    return tbl[i] + (tbl[j] - tbl[i]) * g


def _smooth(pts, rounds=2):
    """Pull the kinks out of a polyline without moving its ends.

    The 背线 is assembled from two arcs that belong to two different
    shapes — the head and the body — and they meet at the nape with a
    step in them. A bird has a nape notch, so the step is not wrong; a
    *corner* is, because 一笔贯气 means the stroke was not lifted.
    """
    for _ in range(rounds):
        out = [pts[0]]
        for i in range(1, len(pts) - 1):
            a, b, c = pts[i - 1], pts[i], pts[i + 1]
            out.append(((a[0] + 2.0 * b[0] + c[0]) * 0.25,
                        (a[1] + 2.0 * b[1] + c[1]) * 0.25))
        out.append(pts[-1])
        pts = out
    return pts


def _body_poly(Lb, d, fl):
    """背如弓, 腹宜虚 — the body outline, in body-local pixels.

    Thirteen points and not a curve among them, because the radius table
    smooths it anyway and a polygon can be *read*.

    The first version put the top of the dorsal bow in the middle of the
    back and necked the front down to a throat, which is what a body looks
    like when you draw it on its own — and it produced, exactly, the fault
    this module was written to avoid: a ball on a sack, with a pale wedge
    of nothing between the head and the shoulder. A perched bird is not
    built that way. Its back is **highest at the shoulder**, immediately
    behind the skull, and falls all the way to the tail from there; its
    breast juts *forward and down* under the bill, so the front of the
    silhouette is the chest, not a neck. Both of those are what make the
    head look attached, and neither is visible until the head is on.

    `fl` is the cold — a bird raises its contour feathers and gets rounder,
    which is one multiplier on the depth and a little length at the rump.
    """
    dy = d * (1.0 + 0.15 * fl)
    dx = 1.0 + 0.05 * fl
    # The rear half is a *wedge*. A rump as deep as the shoulder closes the
    # outline into an egg, and an egg is the shape this module keeps being
    # accused of however well the ink inside it is graded — because the
    # silhouette is read before any of the ink is. A perched bird narrows
    # steadily from the shoulder to the tail root and its belly is a
    # shallow curve, not a hemisphere.
    pts = [
        (-0.42 * dx, -0.48),         # 肩 — the top of the bow, behind the head
        (-0.18 * dx, -0.50),
        ( 0.08 * dx, -0.42),
        ( 0.31 * dx, -0.28),
        ( 0.47 * dx, -0.12),         # rump
        ( 0.53 * dx,  0.03),         # tail root
        ( 0.44 * dx,  0.19),         # vent
        ( 0.23 * dx,  0.35),
        (-0.05 * dx,  0.45),         # the belly, deepest forward of centre
        (-0.31 * dx,  0.48),         # 胸 — the breast, low and forward
        (-0.49 * dx,  0.37),
        (-0.57 * dx,  0.14),
        (-0.55 * dx, -0.18),         # 喉 — rising to meet the head
    ]
    return [(x * Lb, y * dy * Lb) for x, y in pts]


class Pose:
    """Where every part of one bird is, before any of them is painted."""

    # Deliberately free of `rng`: a Pose is a pure function of its arguments,
    # so the composer can build a throwaway one at the origin to ask where the
    # legs come out, and then build the real one where that answer says to put
    # it, without the two differing by a draw.
    def __init__(self, sp, cx, cy, Lb, face=1, lean=0.18, hunch=0.3,
                 turn=0.0, dip=0.0, fluff=0.2, tail_drop=0.22):
        self.sp = sp
        self.side = 1 if face >= 0 else -1
        self.th = lean if self.side > 0 else math.pi - lean
        self.cth, self.sth = math.cos(self.th), math.sin(self.th)
        self.cx, self.cy = cx, cy
        self.Lb = Lb
        self.hunch, self.turn, self.fluff = hunch, turn, fluff
        self.dip = dip
        d = sp["stout"]
        # 蓬毛 raises the contour feathers; it does not inflate the bird.
        # At 0.24 a 雪霁 kingfisher came out as deep as it was long, which
        # is a ball with a bill in it and no amount of good drawing on top
        # rescues the proportion.
        self.depth = d * (1.0 + 0.15 * fluff) * Lb

        # ---- the body ----------------------------------------------------
        self.poly = _body_poly(Lb, d, fluff)
        self.Rb = _polar(self.poly)
        self.rad_body = self._mirror(self.Rb)

        # ---- the head ----------------------------------------------------
        # 头如卵. The one number the eye actually audits, so it is a plain
        # fraction of the body length and nothing is allowed to scale it
        # except the species table.
        self.hd = sp["head"] * Lb
        R0 = self.hd * 0.5
        self.R0 = R0
        cr = sp["crest"]

        # 回首 — and this is the one thing the class had flatly wrong.
        #
        # `turn` used to be added straight into the bill's *body-local
        # bearing*, which is the angle the whole head frame is built on, so
        # a glance over the shoulder came out as the skull being **rolled in
        # the picture plane**: at full 回首 the crown ended up pointing
        # down and forward, 153 degrees round, and the bird read as a dead
        # one hanging by its feet. The comment four paragraphs down already
        # said what it was supposed to be — a rotation about the bird's
        # *vertical* axis, which in a flat projection leaves the skull
        # upright and swings the bill round it — and the code did not
        # implement a word of it.
        #
        # A yaw in a flat projection is three separate facts and they have
        # to be kept apart, because each one governs different marks:
        #
        # * the skull's **silhouette** shortens along the bill axis and
        #   stops at the head's own width — a head seen end-on is not a
        #   line, it is as wide as the bird's skull is wide (`hsq`);
        # * **which end of the egg is pointed** follows the bill all the
        #   way round, so at a full turn the taper is at the *back* (`tap`);
        # * anything **on the midline** — the bill, the culmen, the gape —
        #   foreshortens by the honest signed cosine and reverses with it
        #   (`hc`), while anything **off** it — the eye, the cheek patch —
        #   swings sideways by its own offset (`hs`) and changes which of
        #   the bird's two the viewer is looking at.
        #
        # Nonlinear in `turn` for the same reason the old code was: a bird
        # foreshortens first and swings last, so half a turn is a glance
        # and not a profile reversed. The exponent and the scale are solved
        # from the two ends of the mood table rather than chosen: 侧顾 is to
        # top out looking out of the sheet at 60 degrees and 回首 is to
        # reach 150, and the band between 85 and 100 degrees is **kept
        # empty on purpose** — a head square on to the viewer has its bill
        # pointing at the eye of the painter and one visible eye on the
        # midline, which is a cyclops however honestly it was projected.
        yaw = turn ** 1.18 * 2.75
        self.yaw = yaw
        self.hc = math.cos(yaw)                       # midline foreshortening
        self.hs = math.sin(yaw)                       # lateral swing
        WID = 0.78                                    # skull width / length
        self.hsq = math.hypot(self.hc, WID * self.hs)
        tap = self.hc / self.hsq

        # 头如卵, projected. ψ is measured in the head's own frame: 0 at
        # the bill, +π/2 at the throat, -π/2 at the crown, π at the nape.
        # An egg is narrow at the bill end and broad at the nape, so the
        # first harmonic is negative — and signed by `tap`, so that when the
        # bird has looked right round the point of the egg has gone round
        # with the bill.
        # The numbers are the whole of 头如卵 and the old ones did not say
        # it. A first harmonic of 0.13 against a second of 0.05 makes a
        # shape 2.10 R0 long and 1.90 deep — a ratio of 1.10, which is a
        # **circle** with a bill stuck on it, and it was the single loudest
        # thing wrong with the bird at six diameters. A passerine skull in
        # profile runs about 1.25, and it is not an egg laid on its side
        # either: the broad end is the crown and nape, the narrow end is
        # the chin, so the long axis is **tilted**, point down and forward.
        # Hence the shift `d`, which rides on the same signed `tap` so that
        # the whole shape mirrors correctly when the bird has looked round.
        d = 0.28 * tap
        def egg(ps):
            return R0 * (1.0 - 0.19 * tap * math.cos(ps - d)
                         + 0.11 * math.cos(2.0 * ps) + 0.05 * fluff)

        # Squash along the bill axis and re-measure, rather than scaling the
        # radius: a radius table of an affinely squashed shape is not the
        # squashed radius table, and `_polar` already knows how to measure a
        # polygon. Everything downstream — the `lift` that cuts the hole,
        # the `touch` that fills it, the 背线, the jaw, the cap — reads this
        # one table, so the yawed head is the yawed head in all of them.
        nh = 96
        hpoly = []
        for i in range(nh):
            ps = i * TAU / nh
            r = egg(ps)
            hpoly.append((math.cos(ps) * r * self.hsq, math.sin(ps) * r))
        Rh = _polar(hpoly)
        if cr > 0.01:
            # The crest is on the midline, so it does not foreshorten — it
            # only *travels*: forward of the crown at rest, over the crown at
            # a half turn, behind it once the bird has looked round. Which is
            # one `atan2` on the same signed taper.
            for ps0, amp, wid in ((-0.95, 0.34, 0.46), (-1.95, 0.14, 0.70)):
                ctr = math.atan2(math.sin(ps0), tap * math.cos(ps0))
                na = len(Rh)
                for i in range(na):
                    e = ((i * TAU / na - ctr + math.pi) % TAU - math.pi) / wid
                    Rh[i] += R0 * amp * cr * math.exp(-e * e)
        self.Rh = Rh
        self._rhead = lambda ps: _at(Rh, ps)
        self.r_crown = _at(Rh, -0.5 * math.pi)
        # `H2S` places head-ψ at `hth - side*ps`, so a touch rotated to `hth`
        # asks for ψ = -side*f. Mirroring it the body's way — the body's own
        # `LD` is `th + side*ps` — put the crest under the chin and floated
        # the 背线 a finger's width clear of the crown it is supposed to be
        # the edge of. One sign, and it is not the same sign as the body's.
        self.rad_head = lambda f: _at(Rh, -self.side * f)

        # 嘴往哪里去 — the bill's direction in body-local polar. Straight
        # ahead is π; `dip` noses it down, `turn` swings the whole head round
        # over the shoulder, which is 回首 and the liveliest thing a perched
        # bird does.
        ba0 = math.pi - 0.20
        # 嘴不至胸. These two used to add at full slope, and the sum is not a
        # posture any bird holds: 俯啄 draws `dip` to -0.8 and is also
        # allowed a fifth of a `turn`, and a fifth of 2.85 is another 0.57
        # radian of nod, which put a 八哥's bill eighty degrees below
        # horizontal — a black bean hanging inside its own breast, with the
        # head's leading edge a smooth curve above it. Two corrections:
        #
        # * the nod is **capped**, because a neck has a stop in it;
        # * the turn's pull on the bill is **a power, not a slope**. 回首 is
        #   a rotation about the body's vertical axis, and in a side view
        #   such a rotation barely moves the bill until it is nearly
        #   complete — the bill foreshortens first and swings last. Linear
        #   in `turn` meant every half-hearted glance also nosed the bird
        #   at the floor.
        nod = max(-0.66, min(0.80, dip))
        dip = self.dip = nod
        # 侧首微倾 — and a *little* roll, now that the yaw is no longer
        # being drawn as one. A bird that turns its head cocks it; nine
        # degrees at the half turn is what reads as attention rather than as
        # a specimen mounted straight. It goes into `ba` and not into `hth`
        # so that `head_in_body` keeps agreeing with the head it measures.
        self.ba = ba0 + nod + 0.13 * self.hs
        # 缩颈 — where the skull sits, and the one measurement that decides
        # whether this is a bird or a snowman. It is given relative to the
        # body's own front, in units of the head's own radius, so it holds
        # for a kingfisher and a bulbul without a second table: a shade
        # forward of the breast, high enough that the crown clears the
        # shoulder, and pulled back and down as the bird hunches. The head
        # has to *overlap* the body by most of its back half — a bird has a
        # neck and you cannot see it.
        hx = -0.50 * Lb - R0 * (0.06 + 0.44 * (1.0 - hunch)) \
            + turn * R0 * 0.26
        # The height is not a free constant. It is set by where the *crown*
        # has to come out, and the crown has to come out above the back —
        # by a hair on a bird with its neck drawn in, by a third of a skull
        # on one standing up. Naming the centre instead, as this did, works
        # at one `stout` and fails at the rest: a 0.84 kingfisher in snow
        # grew a body deep enough to swallow the skull, the crown came out
        # *below* the shoulder, and the result reads as two balls in a bag
        # with a bill on the front one. Name the silhouette, solve for the
        # centre, and the proportion survives every row of the table.
        clear = R0 * (0.16 + 0.42 * (1.0 - hunch))
        # 伸颈回顾 — and the clearance is not constant in `turn`.
        #
        # Past the half turn the bill swings round and points back *along*
        # the bird, and on a head sitting at its resting height that line
        # runs straight through the mantle. Which is honest projection and
        # unreadable ink: what it draws is a dark leaf lying on the back
        # with no visible root, and no amount of 留白 round it turns it
        # into a bill, because the fault is not contrast — the mark is in
        # the wrong *place*. No bird looks over its shoulder from a resting
        # neck either; it stretches up first, which is the whole character
        # of the pose. So the crown is asked for another half skull of air
        # once the turn is past the point where the bill reverses, and the
        # bill clears the back the way it does on a live bird.
        craned = clamp01((turn - 0.46) / 0.34)
        clear += R0 * 0.58 * craned
        hy = -(0.50 * self.depth + clear) + self.r_crown

        # 颈 — and the head turns about it, not about its own middle.
        #
        # The two head freedoms are not the same kind of motion and treating
        # them alike is what broke every attitude but the level one. `turn`
        # is 回首: a rotation about the bird's *vertical* axis, which in a
        # flat projection leaves the skull where it is and swings the bill
        # round it. `dip` is a nod, about the transverse axis, and a nod
        # pivots at the neck — so a bird looking down does not spin its head
        # on the spot like a gun turret, it drops the skull down and back.
        #
        # Rotating in place gave 俯啄 a head still sitting high on the
        # shoulders with a bill pointing at the floor, and 理羽 a head
        # standing straight up with a dagger laid across the breast, which
        # is not a bird preening, it is a bird being stabbed. One pivot and
        # both postures come right, because the pivot is the thing that was
        # actually missing from the anatomy.
        px_ = hx - math.cos(ba0) * R0 * 0.78
        py_ = hy - math.sin(ba0) * R0 * 0.78
        cd, sd = math.cos(dip), math.sin(dip)
        vx, vy = hx - px_, hy - py_
        hx = px_ + vx * cd - vy * sd
        hy = py_ + vx * sd + vy * cd
        self.neck_l = (px_, py_)
        self.hc_l = (hx, hy)
        self.hx, self.hy = self.L2S(hx, hy)
        self.hth = self.th + self.side * self.ba
        # The head frame as two sheet vectors: `eu` along the bill, `ev`
        # toward the throat. `H2S` is exactly `hx + u·eu + v·ev` written in
        # polar, and `HP` is the same thing written in Cartesian so that the
        # yaw can scale the two axes by different amounts.
        self.eux, self.euy = math.cos(self.hth), math.sin(self.hth)
        self.evx, self.evy = self.side * self.euy, -self.side * self.eux

        # ---- the bill ----------------------------------------------------
        bl = sp["bill"] * self.hd
        self.bill_len = bl
        # A bill is on the midline, so it foreshortens by the honest cosine
        # and reverses past the half turn. Floored at a sixth so that a bird
        # caught square-on still has something to eat with; no mood in the
        # table reaches that far, so the floor never actually binds — it is
        # there to keep `atan2(bill - head)` from being noise.
        self.kbill = math.copysign(max(0.17, abs(self.hc)), self.hc or 1.0)
        self.bill_base = self.HP(-0.10, R0 * 0.86, k=self.kbill)
        self.bill_tip = self.HP(0.0, R0 * 0.80 + bl, k=self.kbill)
        self.gape = self.HP(0.40, R0 * 0.80, k=self.kbill)
        # Where the bird is *looking*, in sheet angles. 点睛 reads off this
        # and not off `hth`, because at 回首 the two point opposite ways and
        # a pupil shoved toward the bill's old bearing is a bird whose eye
        # has not been told its head moved.
        self.look = math.atan2(self.bill_tip[1] - self.hy,
                               self.bill_tip[0] - self.hx)

        # ---- the eye -----------------------------------------------------
        # 眼如豆, and it goes where a bird's eye goes: forward of the centre
        # of the skull and above the gape line, never in the middle of the
        # head, which is what makes a drawn bird look like a toy.
        self.eye_r = R0 * (0.175 + 0.03 * (1.0 - sp["stout"]))
        # And a bird has **two**, which only matters once the head can turn.
        # The eye is the one feature far enough off the midline for its own
        # lateral offset to dominate: as the head comes round, the near eye
        # swings toward the tail and over the limb, and the far one comes up
        # past the bill. Drawing the near one regardless put the eye of a
        # 回首 bulbul on the back of its skull. `HPV` picks whichever of the
        # pair is actually facing the viewer, which is two multiplies.
        # Back and up from where it was. 眼在额下 — but the number that
        # forced the move was not anatomy, it was clearance: the eye ring
        # and the base of the bill wedge are each about a fifth of the
        # skull across, and at ψ = -0.56 the two overlapped by a few
        # pixels on every species. What that draws is a bird with its eye
        # welded to its beak, and the join reads as a blot rather than as
        # two marks. Solve for the gap, not for the position.
        self.ex, self.ey = self.HPV(-0.74, R0 * 0.44, R0 * 0.60)

        # ---- wing, tail, legs --------------------------------------------
        # 翅 — a folded wing lies *along* the body, and the only honest way
        # to say so is to name both of its ends in body-local units and let
        # the angle fall out. Naming a direction and a length instead is
        # what put the primaries a body-depth below the belly, hanging off
        # the bird like a dropped fan: 0.26 radians is a small number until
        # it is multiplied by a wing.
        self.wing_l = (-0.30 * Lb, -0.20 * self.depth)
        # The tip is where the *primaries* will end up, so it has to stay
        # above the waterline: at 0.15 of the depth the flight feathers came
        # out hanging below the vent on bare paper, two dark sausages with
        # nothing behind them, reading as something the bird was carrying.
        self.wing_tip_l = ((-0.30 + 1.12 * sp["wing"]) * Lb,
                           0.03 * self.depth)
        self.wing_root = self.L2S(*self.wing_l)
        self.wing_len = math.hypot(self.wing_tip_l[0] - self.wing_l[0],
                                   self.wing_tip_l[1] - self.wing_l[1])
        self.wing_ps = math.atan2(self.wing_tip_l[1] - self.wing_l[1],
                                  self.wing_tip_l[0] - self.wing_l[0])
        # 尾根 sits *above* the axis, inside the ink, not out on the
        # waterline where the coverage field has already given up.
        self.tail_l = (0.46 * Lb, -0.10 * self.depth)
        self.tail_root = self.L2S(*self.tail_l)
        self.tail_len = sp["tail"] * Lb
        self.tail_ps = tail_drop
        self.leg_len = sp["leg"] * Lb
        # 腿根 — where the tarsus leaves the feathers, and it has to be a
        # point the ink actually reaches. On the outline (u≈1) it is below
        # the coverage waterline, so the leg started on bare paper a body
        # depth under the bird. `_stand` seats the whole creature off this
        # one point, so it is held here and read by `_foot` rather than
        # guessed twice.
        # 0.46, not 0.38. Geometrically either is inside the bird; in
        # *ink* they are not. 腹宜虚 fades the belly out well before the
        # contour, so a tarsus started at 0.38 begins on bare paper and the
        # creature reads as mounted on stilts. See `_thigh`, which covers
        # the rest of the distance with feathers rather than with geometry.
        self.leg_l = (-0.02 * Lb, 0.46 * self.depth)

        # ---- the two silhouette paths every stroke has to respect ---------
        self.back = self._back_path()
        self.front = self._front_path()
        self.jaw = self._jaw_path()
        self.reach = max(Lb * 0.62 + self.hd,
                         self.tail_len + Lb * 0.5, self.wing_len)

    # -- frames ------------------------------------------------------------

    def L2S(self, bx, by):
        """Body-local pixels -> sheet pixels."""
        s = self.side
        return (self.cx + bx * self.cth - by * s * self.sth,
                self.cy + bx * self.sth + by * s * self.cth)

    def LD(self, ps):
        """Sheet angle of the body-local polar direction ψ."""
        return self.th + self.side * ps

    def H2S(self, ps, r):
        """Head-frame (ψ, radius) -> sheet pixels. ψ=0 is the bill.

        Note the **minus**. The head frame's reference direction is the
        bill, which points the opposite way along the creature from the
        body frame's +x, so sharing `LD`'s sign would make +ψ ventral on
        the body and dorsal on the head. It did, for one round: the crown
        came out pale, the crest grew under the chin and the eye sat below
        the gape, and every one of those was this one character.
        """
        a = self.hth - self.side * ps
        return (self.hx + math.cos(a) * r, self.hy + math.sin(a) * r)

    def HD(self, ps):
        """Sheet angle of the head-frame direction ψ."""
        return self.hth - self.side * ps

    def HP(self, ps, r, lat=0.0, k=None):
        """A point *on the head*, carried through the 回首 yaw.

        `H2S` maps the head's **outline**, whose radii already come out of
        the projected table `Rh`. This maps everything that is not the
        outline — the bill, the gape, the eye, a patch of cheek — and those
        have to be projected here instead, because they are named at rest.

        `ps`, `r` are the feature's place in the head's own frame at rest;
        `lat` is how far it sits **off the midline**, positive toward the
        viewer, in the same units. A bill has `lat = 0` and foreshortens to
        nothing when the bird looks straight out of the sheet; an eye does
        not, and swings the other way instead.
        """
        c = self.hc if k is None else k
        u = math.cos(ps) * r * c - lat * self.hs
        v = math.sin(ps) * r
        return (self.hx + u * self.eux + v * self.evx,
                self.hy + u * self.euy + v * self.evy)

    def HPV(self, ps, r, lat, k=None):
        """`HP`, for a feature the bird has one of on each side.

        Returns whichever of the pair is nearer the viewer. The depth of a
        feature at (`ps`, `r`, ±`lat`) after a yaw of `yaw` is
        `r·cosψ·sin(yaw) ± lat·cos(yaw)`, and the test is which of those
        two is larger — no occlusion machinery, because a head has exactly
        two of these and only one of them is ever drawn.
        """
        a = math.cos(ps) * r * self.hs
        return self.HP(ps, r, lat if a + lat * self.hc
                       >= a - lat * self.hc else -lat, k)

    def _mirror(self, tbl):
        s = self.side
        return lambda f: _at(tbl, s * f)

    def head_in_body(self, ps, r):
        """How deep a head-frame point (ψ, r) lies inside the body outline.

        0 outside, ramping to 1 a fifth of a body-length in. This is the
        number that makes four masses read as one animal, and it replaces
        every constant that was previously guessing at it. A 没骨 head is
        laid *over* a wet body and the two run together; a head `touch`
        with its own coverage and its own 水线 rim, laid over a body that
        already has both, draws a complete second outline *inside* the
        first — a bubble in a bubble, with a hard arc across the shoulder.
        No tone fixes it, because the fault is not a tone: it is an edge
        where the hand never lifted the brush.

        Measuring it instead of fading on a fixed angle matters because
        the head *moves*: 缩颈 buries the nape entirely, 回首 swings it
        clear over the mantle, and the same constant cannot be right for
        both. Geometry is right for both, and costs one `atan2`.
        """
        a = self.ba - ps
        bx = self.hc_l[0] + math.cos(a) * r
        by = self.hc_l[1] + math.sin(a) * r
        d = math.hypot(bx, by)
        R = _at(self.Rb, math.atan2(by, bx))
        return clamp01((R - d) / (0.20 * self.Lb))

    def on_body(self, ps, u=1.0):
        """A point on (u=1) or inside (u<1) the body outline at ψ."""
        r = _at(self.Rb, ps) * u
        return self.L2S(math.cos(ps) * r, math.sin(ps) * r)

    # -- the silhouette ----------------------------------------------------

    def _hand_over(self, ps):
        """Body-local polar angle of the head-outline point at head-ψ.

        The hand-over from the head's arc to the body's. Hard-coding it
        worked at one value of `hunch` and put a hook in the back line at
        every other, because the head moves with the hunch and the body
        does not: the mantle would start *in front of* where the crown
        stopped and the stroke doubled back on itself. Measuring it costs
        four lines and is right at every pose.
        """
        a = self.ba - ps                          # head-ψ -> body-local dir
        r = self._rhead(ps)
        return math.atan2(self.hc_l[1] + math.sin(a) * r,
                          self.hc_l[0] + math.cos(a) * r)

    def _back_path(self):
        """背线 — bill base over the crown and the nape to the tail root.

        One stroke, and it is the same outline the body and head masses are
        painted from, sampled rather than redrawn. The two cannot drift
        apart because there is only one of them.
        """
        pts = []
        n, e0, e1 = 11, -1.02, -2.52
        for i in range(n + 1):
            ps = e0 + (e1 - e0) * i / n           # bill side -> nape
            r = self._rhead(ps)
            # Only where the head *is* the silhouette. Once the head can
            # nod and turn, its crown-to-nape arc spends part of its length
            # buried in the shoulder, and drawing it there lays a dark line
            # across the middle of the bird — a loop of wire over the back,
            # which is the exact fault 墨梅 §9 names: a line whose far end
            # has nothing to sit against is read as an object. The burial
            # test already exists for the ink; the line gets it too.
            if self.head_in_body(ps, r * 0.90) > 0.22:
                continue
            pts.append(self.H2S(ps, r * 0.90))
        if len(pts) < 3:
            pts = []
        a0 = min(-0.70, self._hand_over(e1) + 0.06)
        if self.turn > 0.45 or abs(self.dip) > 0.42:
            # 回首 / 俯啄: the head is off the body's own axis and there is
            # no continuous dorsal line to draw. Joining the crown to the
            # mantle anyway runs the stroke straight across the bird —
            # a loop of wire laid over the breast — because the two arcs
            # no longer meet anywhere. Two strokes, as the hand would.
            self.back2 = self._mantle(-2.56, -0.16)
            return _smooth(pts, 1)
        self.back2 = None
        if not pts:
            self.back2 = self._mantle(-2.56, -0.16)
            return []
        pts += self._mantle(a0, -0.16)
        return _smooth(pts, 2)

    def _in_head(self, bx, by, u=0.98):
        """Is a body-local point inside the head's own outline?"""
        dx, dy = bx - self.hc_l[0], by - self.hc_l[1]
        return (math.hypot(dx, dy)
                < self._rhead(self.ba - math.atan2(dy, dx)) * u)

    def _mantle(self, a0, a1):
        # Nineteen points and smoothed, not fourteen points raw. `back2` —
        # the standalone mantle line a 仰首 or 回首 bird gets instead of one
        # continuous 背线 — was the only path in the module that reached
        # `sweep` without going through `_smooth`, and at a body length of
        # 180 pixels its segments are fifteen pixels of dead straight line
        # apiece. At seven diameters that is unmistakable: a thin grey
        # **chevron** laid across the cheek, two ruled strokes meeting at a
        # corner, the single most mechanical mark on the sheet.
        #
        # And it is trimmed at the head. The arc starts up and forward of
        # the body's centre, which on a bird with its neck out is *behind
        # the skull* — so the first two or three points are inside the head
        # outline and the line begins in the middle of the face. Same test
        # `_back_path` already applies to its own samples, same reason:
        # 墨不到处笔不到, and a line whose end has nothing to sit against
        # reads as an object rather than as an edge.
        out, n = [], 18
        lead = True
        for i in range(n + 1):
            ps = a0 + (a1 - a0) * i / n
            r = _at(self.Rb, ps)
            bx, by = math.cos(ps) * r, math.sin(ps) * r
            if lead:
                if self._in_head(bx, by):
                    continue
                lead = False
            out.append(self.L2S(bx, by))
        return _smooth(out, 2) if len(out) >= 3 else out

    def _jaw_path(self):
        """喉线 — the fold where the throat feathers meet the breast.

        Not part of the silhouette, and that is the point. On a 缩颈 bird
        the skull is four fifths buried in the shoulders, so the union
        outline of head and body runs straight past it: geometrically
        honest, and it leaves a sheet with an eye and a bill stuck on an
        almond. A painter draws the jaw anyway. It is one short mark and
        it is the whole difference between a hunched bird and a lozenge.

        It is drawn only where it is *interior*: it starts where the head
        has gone under the body outline at all, and stops where it has
        gone right under. Outside that band it would either double the
        silhouette `_front_path` already draws, or vanish into the breast.
        The band has to be wide — on a 缩颈 八哥 the throat is already 42%
        buried at the gape — which is the whole reason the first version
        of this drew nothing on exactly the birds that needed it.
        """
        pts = []
        for i in range(21):
            ps = 0.45 + 1.75 * i / 20.0
            r = self._rhead(ps) * 0.97
            b = self.head_in_body(ps, r)
            if b < 0.03 or b > 0.92:
                if pts:
                    break
                continue
            pts.append(self.H2S(ps, r))
        return _smooth(pts, 1) if len(pts) >= 3 else []

    def _front_path(self):
        """The throat and the top of the breast, and then it stops.

        腹宜虚, and the first version did not take it seriously enough. It
        ran the chin round the whole breast to the belly — a correct
        contour, drawn where there is no ink, so it read as a wire hanging
        below the bird. The lower line has to **die inside the painted
        mass**: it goes as far as the ink does and no further, because a
        line whose far end has nothing to sit against is a line the eye
        reads as an object.
        """
        pts = []
        for i in range(6):
            ps = 0.34 + 0.68 * i / 5.0            # chin -> under the jaw
            r = self._rhead(ps)
            if self.head_in_body(ps, r * 0.99) > 0.22:
                continue
            pts.append(self.H2S(ps, r * 0.99))
        a0 = self._hand_over(1.02) - 0.05
        body = []
        for i in range(6):
            ps = a0 - 0.22 * i / 5.0              # throat -> upper breast
            r = _at(self.Rb, ps)
            body.append(self.L2S(math.cos(ps) * r, math.sin(ps) * r))
        # The chin and the breast are one stroke only while they are next
        # to each other. A nodding head takes its jaw away from the throat,
        # and bridging the gap draws a straight wire from the bill down
        # across the bird to the chest.
        if pts and math.hypot(pts[-1][0] - body[0][0],
                              pts[-1][1] - body[0][1]) > self.R0 * 0.55:
            return _smooth(body, 2) if len(body) > 2 else []
        pts += body
        return _smooth(pts, 2) if len(pts) > 2 else []


# --------------------------------------------------------------------------
# 七笔 — the marks.
#
# Order is not decoration here, it is the drawing. 写意翎毛 goes
# 头、背、翅、尾、嘴、眼、爪, and two of those positions are structural
# rather than traditional: the masses all `lift` the paper out from behind
# themselves *before* they are painted, so the branch stops showing through
# the bird; and the claws are drawn dead last, over the wood, which is the
# only reason they read as gripping it rather than resting on it.
# --------------------------------------------------------------------------

def _ramp(st, sp, k, pale=0.55):
    """The two stops of this bird's plumage ramp.

    墨分五色 means the five are *far apart*. The first version read the
    species' `tone` as a dilution of the dark stop and as a floor under the
    pale one, which is the arithmetic of a single wash: a 0.76 bird came out
    with a ramp running from a mid tan to a mid brown, and every mark drawn
    between them landed in the same sixth of the value scale. The sheet then
    reads as a grey animal in fog however correct its drawing is, and no
    amount of extra line rescues it, because the fault is that the darkest
    ink on the bird is not dark.

    So `tone` now moves the two stops *apart* instead of together: it is a
    floor under the loaded end and a ceiling over the pale end. 八哥 at 1.0
    is unchanged — it was always black — and every other bird gains most of
    a value scale.
    """
    tn = clamp01(sp["tone"] * k)
    return (dilute(rgb(st["plume"][0]), clamp01(pale * (0.52 + 0.48 * tn))),
            dilute(rgb(st["plume"][1]), clamp01(0.46 + 0.54 * tn)))


def _mass(cv, rng, res, po, st, air, k):
    """头与背 — the head and the body, each one touch, belly left as paper.

    This is the whole reason `Pose` computes an outline instead of a list
    of parts. `rad_body` is handed to `lift` and to `touch` unchanged, so
    the hole in the sheet and the ink that fills it are the same shape to
    the pixel, and the 背线 drawn later runs along that same curve. Three
    marks that cannot disagree.
    """
    sp = po.sp
    lo, hi = _ramp(st, sp, k)
    # 头重 — on every bird in the table the crown and the nape are the
    # darkest plumage on the animal, and painting the head off the body's
    # own ramp is what made the skull read as a lighter ball stuck to a
    # darker sack. One notch down the same ramp, not a second palette.
    hlo, hhi = _ramp(st, sp, min(1.0, k * 1.14), pale=0.48)
    wet = air["wet"]
    ft = 1.5 * wet + 0.5 * po.fluff
    # 焦墨 — 八哥 is the one bird in the table that is *black*, and it was
    # coming out the same mid grey as the sparrow. The reason is that
    # `tone` was only ever spent on the two stops of the ramp: the dark
    # stop was already ink, and the field almost never asks for it. On a
    # mid flank `w` is 0.47 and the shade comes out 0.66 — correct for a
    # brown bird graded from spine to belly, and wrong in kind for a
    # myna, whose value structure is not a ramp at all. A black bird is
    # black everywhere and its lights are 留白 and 飞白, not pale plumage.
    # So for the top fifth of the `tone` scale the whole field is pushed
    # toward its loaded end, which leaves 墨分五色 intact *inside* the
    # black instead of spreading it from black to tan.
    # A **gamma**, not a lerp toward black. The first version was
    # `1 - (1-sh)·(1 - 0.72·blk)`, an affine push, and affine is exactly
    # the wrong shape: it moves the *bottom* of the range as hard as the
    # top, so a myna's lightest value went from 0 to 0.72 and the bird
    # came out a flat dark lump with no 留白 in it — which is a different
    # way of failing 墨分五色, not a fix for it. A gamma pins 0 and 1 and
    # bows everything between them, so the breast and the belly keep
    # their paper and only the mid tones go to ink.
    blk = clamp01((sp["tone"] - 0.80) / 0.20) * k

    def sink(sh):
        if sh <= 0.0:
            return 0.0
        return sh ** (1.0 - 0.56 * blk)

    side = po.side
    Rb0 = po.Rb

    # 留白 first: the branch behind the bird is lifted out, not painted
    # over. Under a multiply composite there is no other way to be in front.
    #
    # But the hole has to be the shape of the *ink*, not of the silhouette.
    # Lifting the whole outline restores clean paper under the belly — and
    # 腹宜虚 means no ink will ever be laid there, so what the sheet gets is
    # a bright hard-rimmed balloon hanging under the bird, cleaner than the
    # ground around it and lighter than the 做旧 that follows. It was the
    # single most damaging mark in the module and it was not a brush stroke
    # at all. So the lift is tapered to the same waterline the coverage
    # field uses, and the 没骨 mass goes on being opaque where it is opaque.
    def lift_body(f):
        ps = side * f
        return _at(Rb0, ps) * (0.56 + 0.44 * clamp01(
            (0.84 - math.sin(ps)) / 0.80))

    def lift_head(f):
        ps = -side * f
        return po._rhead(ps) * (0.66 + 0.34 * clamp01(
            (0.92 - math.sin(ps)) / 0.75))

    res.lift(cv, po.cx, po.cy, lift_body, rot=po.th, alpha=0.94,
             feather=ft * 1.5, pad=0.6)
    res.lift(cv, po.hx, po.hy, lift_head, rot=po.hth, alpha=0.95,
             feather=ft * 1.1, pad=0.5)

    ph = rng.uniform(0, TAU)
    # The one coordinate both fields are written in. `touch` hands its tone
    # function a polar pair (u, φ) and the first two versions of this file
    # used it directly — which is why the first produced a dark ring round a
    # pale core and the second a flat lens cut through the middle of the
    # bird. Value on a bird is not polar in anything: it is *height on the
    # flank*, measured straight down the body's own y axis. So convert, and
    # write both fields in (X, Y) where X runs -1 at the breast to +1 at the
    # tail root and Y runs -1 along the spine to +1 at the deepest belly.
    Rb, half, hlen = po.Rb, po.depth * 0.5, po.Lb * 0.52
    paper = 0.55 * (1.0 - sp["belly"])

    def body_tone(u, f):
        ps = side * f
        r = _at(Rb, ps) * u
        X = math.cos(ps) * r / hlen
        Y = math.sin(ps) * r / half
        # 背如弓: loaded along the spine, giving out down the flank, with a
        # little extra behind the shoulder where a 没骨 bird carries its
        # weight, and a 水线 at the very rim where the brush pooled.
        w = _ease((0.58 - Y) / 1.24)
        # The ripple is multiplied by u², and the exponent is not taste.
        # `touch` samples its field in polar coordinates, so *any* term
        # that depends on the angle alone is constant along a ray — which
        # near the centre, where 216 angular bins meet in one pixel, comes
        # out as a starburst of faint spokes converging on the middle of
        # the bird. It only became visible when `cover` rose, and it is a
        # sampling artefact of a mark that has no centre in the first
        # place: 水纹 belongs at the rim, where the brush pooled.
        sh = (0.21 + 0.79 * w ** 0.74) * (0.84 + 0.16 * u) \
            + 0.10 * w * clamp01(-X) + 0.13 * w * u ** 7 \
            + 0.05 * math.sin(5.3 * ps + ph) * u * u
        sh = sink(sh)
        # 腹宜虚. Coverage, not shade: at zero the sheet is not touched at
        # all, so the belly keeps the ground's own grain and foxing instead
        # of wearing a pale grey copy of it. The waterline is a *slope*, not
        # a level — it runs right down under the breast and climbs above the
        # axis at the vent, which is where a small bird's white actually is.
        cov = _ease(((0.70 + paper) - 0.30 * X - Y) / 0.45) ** 0.85
        return clamp01(sh), cov

    # `cover` is the other half of 墨分五色 and it was the half holding the
    # bird back. At 0.42 the mantle is a glaze: most of what the eye sees
    # there is still the 米黄 ground showing through, so the darkest ink on
    # the sheet arrives on the paper already halfway to the paper's own
    # value. 没骨 is not a glaze — the loaded brush *puts the ink down*. It
    # is only safe to say so now the lift above is shaped like the ink,
    # because an opaque mass over a silhouette-shaped hole is exactly the
    # bright balloon that lift was cut to avoid.
    touch(cv, po.cx, po.cy, po.rad_body, body_tone, lo, hi, alpha=0.93,
          feather=ft, rot=po.th, cover=0.62, na=216, nr=18)

    rh = po._rhead

    def head_tone(u, f):
        # Same field in the head's own frame, where +x is the bill and +y is
        # the throat. The hood goes over the crown and down past the eye; the
        # chin is paper.
        ps = -side * f
        X = math.cos(ps) * u
        Y = math.sin(ps) * u
        w = _ease((0.64 - Y) / 1.22)
        # 一笔点头 — and the brush was not lifted between the head and the
        # back. Where this sample is already inside the body the head gives
        # up its own rim and most of its own coverage, so the two masses run
        # together the way wet ink does; where it is the silhouette it keeps
        # everything. `bur` is 0 on the crown, face and chin at every pose
        # and 1 deep in the shoulder at every pose, which is why 缩颈 and
        # 回首 both come out right off one expression.
        bur = po.head_in_body(ps, rh(ps) * u)
        sh = sink((0.32 + 0.68 * w ** 0.70) * (0.86 + 0.14 * u)
                  + 0.13 * w * u ** 7 * (1.0 - bur))
        # 颏下虚 — *pale*, not bare, and that distinction is worth a
        # paragraph because getting it wrong was the other loud fault at
        # six diameters. The body can afford a coverage field that reaches
        # zero: 腹宜虚 across a mass the length of the bird reads as a
        # belly giving out into paper. The same field across the head is a
        # straight waterline laid over a disc 70 pixels wide, and what it
        # draws is a **wedge bitten out of the skull** — a hard chord from
        # the gape to the nape with the branch showing through it. Ablated
        # down to the two masses alone, that bite was the crescent seam
        # that three rounds of work had been blaming on the 喉线 and on the
        # 背线, neither of which turned out to be drawing it at all.
        #
        # So the ramp is nearly twice as long and the field keeps a floor:
        # the chin is the palest ink on the animal and it is still ink.
        # The shade field already makes it pale — `w` is 0 down there — so
        # nothing is lost but the hole.
        cov = _ease(((1.02 + paper * 0.55) - 0.26 * X - Y) / 0.72) ** 0.85
        return clamp01(sh), (0.26 + 0.74 * cov) * (1.0 - 0.80 * bur)

    touch(cv, po.hx, po.hy, po.rad_head, head_tone, hlo, hhi, alpha=0.94,
          feather=ft * 0.85, rot=po.hth, cover=0.66, na=192, nr=16)

    # 分羽 — two or three side-loaded strokes laid back along the mantle,
    # inside the mass that is already there.
    #
    # Everything above this line is a *field*: a smooth function of height
    # on the flank, correct in every value and recognisably not painted by
    # a hand. A 没骨 bird is laid in with a handful of loaded strokes and
    # the eye reads the bird partly from their edges — where one stroke
    # laps the last, where the brush ran dry, which way the hairs went.
    # Without them the sheet carries a photograph of a bird's tonality,
    # which is the one way this family can be wrong while measuring right.
    #
    # They are strokes, so they obey crab's law: the mass is already
    # painted, these are marks *inside* it, at an alpha that cannot build
    # a second silhouette, and they stop short of the waterline because
    # below it there is nothing for them to lie on (墨不到处笔不到).
    nm = 2 if po.Lb < 86.0 else 3
    for i in range(nm):
        t = (i + 0.5) / nm
        ps0 = -2.30 + 0.30 * t                      # 肩 — behind the skull
        ps1 = -0.46 - 0.34 * t                      # 腰 — short of the rump
        u = 0.94 - 0.30 * t                         # and each one lower
        x0, y0 = po.on_body(ps0, u)
        x1, y1 = po.on_body(ps1, u)
        L = math.hypot(x1 - x0, y1 - y0)
        if L < po.Lb * 0.22:
            continue
        sa_ = math.atan2(y1 - y0, x1 - x0)
        press(cv, (x0 + x1) * 0.5, (y0 + y1) * 0.5, L * 1.02,
              po.depth * (0.26 - 0.05 * t), sa_, lo, hi,
              load=0.74 - 0.14 * t, grad=0.52, flank=side,
              rim=0.10, tip=0.46, tail=0.52, splay=0.26, bulge=0.16,
              streak=0.16 + 0.30 * air["dry"], bare=0.06 + 0.06 * t,
              ragged=0.07, alpha=0.20 - 0.05 * t,
              phase=rng.uniform(0, TAU))

    _simao(cv, rng, po, st, air, k, lo, hi)

    # 赭石一抹 — the accent, if the school has one. One wash over the mantle
    # and the flank and nowhere else: a bird coloured all over is a specimen
    # plate, and this family's 淡彩 means a breath of colour on an ink
    # drawing, not a painting in colour.
    acc = st["accent"]
    if acc and st["accent_a"] > 0.01:
        ac = rgb(acc)
        al = mix(ac, (255.0, 255.0, 255.0), 0.55)

        def acc_tone(u, f):
            ps = side * f
            dors = -math.sin(ps)
            fwd = -math.cos(ps)
            w = clamp01((dors + 0.20) / 0.90) * clamp01(0.70 - 0.55 * fwd)
            return 0.30 + 0.70 * u, w * (0.55 + 0.45 * u)

        touch(cv, po.cx, po.cy, po.rad_body, acc_tone, al, ac,
              alpha=st["accent_a"], feather=ft * 1.3, rot=po.th,
              cover=0.0, na=144, nr=12)


def _simao(cv, rng, po, st, air, k, lo, hi):
    """丝毛 — the hair strokes that turn a wash into plumage.

    Everything in `_mass` is a *field*, and a field is a photograph. The
    two masses measure right in every value, the waterline is in the right
    place and the hood sits where a hood sits — and at three diameters the
    bird is a smooth airbrushed dome, because nothing on it was made by a
    hand holding a brush. The 分羽 presses above are the right idea at a
    tenth of the needed dose: three strokes at alpha 0.2 across a mass two
    hundred pixels long disappear into it.

    What a 没骨 bird actually carries is 丝毛 — several dozen short fine
    strokes, each a hair's width, each lying the way the contour feathers
    lie. Three things make them read as feathers rather than as scratches:

    * **They flow.** Plumage has a lay, and the lay is one field. The
      first version took it to be the body outline's own tangent, which
      is wrong in a way worth recording: a tangent at constant depth is a
      *circle about the body's centre*, so forty of them drew forty
      concentric contour rings — a topographic map of a bird. Feathers do
      not go round a bird, they go **down** it. The lay radiates from one
      source just ahead of and above the shoulder, which is where the
      neck feathers come over: straight back along the mantle, down and
      back over the breast, back and out along the flank, and no stroke
      anywhere running against its neighbour.
    * **They stop at the waterline.** The same coverage field `body_tone`
      uses gates the alpha, so no hair is drawn on bare paper below the
      belly. 墨不到处笔不到 — a hair stroke lying on nothing is a whisker.
    * **They are too faint to see one of.** Each is alpha ~0.1 on ink that
      is already there; what the eye reads is the *grain* of forty of
      them, not any one. Anything strong enough to be seen individually
      builds a second silhouette, which is crab's law all over again.
    """
    side = po.side
    Rb = po.Rb
    half, hlen = po.depth * 0.5, po.Lb * 0.52
    paper = 0.55 * (1.0 - po.sp["belly"])
    # Dry air means a splayed, thirsty brush: fewer hairs, more of them
    # visible. Wet air means the stroke sinks into the wash it lies on.
    dry = air["dry"]
    n = max(22, int(po.Lb * (0.86 + 0.34 * dry)))
    ln = po.Lb * (0.135 + 0.065 * dry)
    hw = max(0.40, po.Lb * 0.0082)
    base = (0.20 + 0.14 * dry) * (0.70 + 0.50 * k)
    for _ in range(n):
        ps = rng.uniform(-math.pi, math.pi)
        u = 0.22 + 0.72 * math.sqrt(rng.random())
        r = _at(Rb, ps) * u
        X = math.cos(ps) * r / hlen
        Y = math.sin(ps) * r / half
        cov = clamp01(((0.70 + paper) - 0.30 * X - Y) / 0.45)
        if cov < 0.22:
            continue
        w = clamp01((0.58 - Y) / 1.24)
        a = base * cov * (0.34 + 0.66 * w)
        if a < 0.012:
            continue
        # the lay: away from a source at the base of the neck
        cx0 = math.cos(ps) * r
        cy0 = math.sin(ps) * r
        th = math.atan2(cy0 + half * 0.90, cx0 + hlen * 1.15) \
            + rng.uniform(-0.15, 0.15)
        L = ln * rng.uniform(0.62, 1.34)
        # a hair bows away from the body the way a hair does
        bx, by = -math.sin(th) * L * 0.09, math.cos(th) * L * 0.09
        pts = []
        for j in range(4):
            t = j / 3.0 - 0.5
            b = (0.25 - t * t) * 4.0
            lx = cx0 + math.cos(th) * L * t + bx * b
            ly = cy0 + math.sin(th) * L * t + by * b
            pts.append(po.L2S(lx, ly))
        drag(cv, pts, hw * rng.uniform(0.72, 1.25), hw * 0.34,
             lo, hi, alpha=a, d0=0.42 + 0.30 * w, d1=0.88,
             core=0.30, feather=0.85,
             dry=0.30 * dry, dry_ph=rng.uniform(0, TAU))


def _cap(cv, po, st, k):
    """头顶 — the dark crown and nape.

    One more `touch` in the head's own frame, over the mass that is already
    there, and it is the cheapest vividness in the module: a small bird is
    read head-first, and a head that is the same value as the flank gives
    the eye nothing to read.

    Two things keep it from being a helmet. It is written as *coverage*, so
    where the dial is low the sheet is not touched at all rather than
    tinted; and the boundary is a **line in the head's own Cartesian
    frame**, not a function of the polar angle. The first version used the
    angle — `dorsal + a share of rearward` — and a function of the angle is
    constant along a ray, so multiplying it by `u` to kill the starburst at
    the centre turned the whole cap into a *rim*: one dark arc riding the
    top of the skull like a hatband, which is worse than no cap at all. A
    cap is a region, and the region is everything above a line that passes
    a little over the eye and tips down toward the nape.
    """
    sp = po.sp
    cap = sp["cap"]
    if cap < 0.05:
        return
    # Not `_ramp`. That function spends the species' `tone` dial on how far
    # apart to pull the two stops of the *plumage*, and at 白头翁's 0.80 its
    # loaded end comes out at 73 of 255 — which, laid over a flank already
    # sitting at 120, lands the crown at 76. Measured, that is a 44-level
    # separation and it reads as a shadow rather than as a hood. 头顶 is not
    # plumage graded one notch darker, it is the one place on a small bird
    # where the painter puts 焦墨 straight from the stone, so the dark stop
    # here is the ink itself, barely let down, and only the amount of paper
    # left in it scales with the species.
    hi = dilute(rgb(st["plume"][1]), clamp01(0.74 + 0.26 * sp["tone"] * k))
    lo = dilute(rgb(st["plume"][0]), 0.40)
    side = po.side
    rh = po._rhead

    def tone(u, f):
        ps = -side * f
        X = math.cos(ps) * u                      # +1 at the bill
        Y = math.sin(ps) * u                      # +1 at the throat
        # Above a line that clears the eye (which sits at about X 0.36,
        # Y -0.22) and tips down behind, so the hood runs on into the
        # mantle at the nape instead of stopping in mid-skull.
        # ...and the line is *ruffled*. A hood whose hem is a clean curve
        # reads as a painted-on mask however soft the ramp across it,
        # because feathers do not end on a curve -- they interlock. Two
        # octaves of wobble on the boundary, a twentieth of the skull
        # each, turn the hem into 参差 without moving the region.
        w = clamp01((-0.08 - Y - 0.34 * X
                     + 0.055 * math.sin(3.1 * ps + 1.7)
                     + 0.030 * math.sin(7.3 * ps - 0.6)) / 0.40) ** 0.85
        # and it gives up where the skull is inside the body, exactly as
        # `head_tone` does — 墨不到处笔不到, and a 缩颈 bird would otherwise
        # wear a dark band across its own shoulder.
        bur = po.head_in_body(ps, rh(ps) * u)
        return 0.30 + 0.70 * w, w * cap * (0.88 + 0.12 * u) * (1.0 - bur)

    # `cover` high, and that is the point of the mark. A hood laid on at
    # the module's usual 0.5 is a *glaze* over ink that is already there,
    # and a glaze over mid grey is mid grey: the first working version
    # moved the crown by 43 levels, which measures as a cap and reads as a
    # smudge. 浓墨 has to actually land.
    touch(cv, po.hx, po.hy, po.rad_head, tone, lo, hi, alpha=0.95,
          feather=po.R0 * 0.14, rot=po.hth, cover=0.86, na=168, nr=18)


def _patch(cv, res, po, ps, r, alpha=0.88):
    """A patch of plumage lifted back to paper — 麻雀's cheek, 白头翁's nape.

    The same `lift` as everything else. White cannot be painted onto a
    米黄 ground under multiply; the only white this family has is the sheet,
    so a white patch is a *hole*, cut after the hood has been laid.
    """
    # A cheek or a nape is as far off the midline as an eye is, so it
    # travels the same way and the bird has two of it: at 回首 the patch
    # the viewer sees is the one on the *other* side of the head.
    x, y = po.HPV(ps, po.R0 * 0.58, po.R0 * 0.50)
    # Feathered wide, lifted well short of a full restore, and *long* — a
    # patch of plumage lies along the head the way the feathers in it do.
    # A round one at alpha 0.9 with a tight edge is not plumage, it is a
    # hole punched in the bird: on a 翠鸟 it came out as a white amoeba at
    # the jaw that cut the bill off the head. 粉白 on silk is a shade of
    # the paper, and it has a grain.
    rot = po.HD(ps) + math.pi * 0.5
    res.lift(cv, x, y, lambda f: r * (1.30 - 0.42 * abs(math.cos(f))),
             rot=rot, alpha=alpha, feather=r * 1.0)


def _wing(cv, rng, res, po, st, air, k):
    """翅分三停 — coverts, greater coverts, flight feathers.

    A folded wing is not a shape, it is three ranks of different shapes
    stacked back along the flank, and leaving out the middle rank is what
    makes a painted wing read as a leaf stuck to the bird. The covert mass
    is one press (crab's law: a mass is one stroke, everything else is
    line drawn inside it); the primaries are presses too, long and narrow
    and each pointed, because a flight feather *is* one stroke of a brush
    that was lifted.
    """
    sp, side = po.sp, po.side
    lo, hi = _ramp(st, sp, k, pale=0.50)
    dk = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k * 1.05))
    wl = po.wing_len
    a = po.LD(po.wing_ps)
    ca, sa = math.cos(a), math.sin(a)
    rx, ry = po.wing_root
    dry = air["dry"]

    # 覆羽 — the mass, and *how far dorsal* is the whole of it. The wing
    # root sits at a fifth of the depth above the axis and the mass is
    # about a third of the depth wide, so seating it a hair above the root
    # left its upper edge an eighth of a depth short of the back — a pale
    # crescent of gap between the wing and the 背线, the whole length of
    # the bird. What that draws is a dark leaf laid *on* the bird rather
    # than a wing folded *into* it, and it was the last mark on the animal
    # still reading as a separate object. A folded wing has no gap: its
    # upper edge **is** the back from the shoulder to halfway down. So the
    # offset is solved for instead of chosen — root + offset + half-width
    # lands on -0.50 of the depth, which is where `_body_poly` puts 肩.
    off = -side * 0.5 * math.pi
    L = wl * 0.80
    W = po.depth * 0.37
    lift_w = po.depth * 0.50 - po.depth * 0.20 - W * 0.5
    mx = rx + ca * L * 0.46 + math.cos(a + off) * lift_w
    my = ry + sa * L * 0.46 + math.sin(a + off) * lift_w
    press(cv, mx, my, L, W, a, lo, hi, load=0.86, grad=0.40, flank=side,
          rim=0.02, splay=0.34, tip=0.44, tail=0.30, bulge=0.20,
          streak=0.04 + 0.14 * dry, bare=0.04, ragged=0.05,
          alpha=0.40, phase=rng.uniform(0, TAU))

    # 飞羽 — the primaries, from under the back of the covert mass. They are
    # a *bundle*, not a fan: a folded wing stacks its flight feathers almost
    # on top of one another and the whole group ends in one point near the
    # tail root. Opening them even a fifth of a radian each turns the bird
    # into a bird holding a comb.
    #
    # Fanning them *along* the wing and nothing else is what made the bundle
    # a dagger. Four presses of near-black ink at alpha 0.58, aimed within a
    # tenth of a radian of one another and all sharing a centreline, do not
    # read as four feathers: they multiply into one solid wedge with a hard
    # straight upper edge and no interior at all — the single darkest mark on
    # the sheet, pointing out of the bird like a blade. A folded wing stacks
    # its primaries like a dealt hand of cards, so the feathers are stepped
    # **across** the wing as well as turned along it, from a little dorsal at
    # the short inner one to well ventral at the long outer one. That one
    # offset is what puts a hairline of paper between each blade and the
    # next, and the hairlines are the whole mark: 墨不到处笔不到.
    #
    # The alpha comes down with it. Separated strokes no longer have to be
    # dark individually to make a dark bundle, and the pairwise overlaps are
    # now narrow enough that where two do cross, the crossing reads as the
    # shadow under an overlapping feather instead of as more wedge.
    # And they were far too **short**, which is the same arithmetic mistake
    # as the 翎管 whiskers read backwards. A press with `tip=0.74, tail=0.42`
    # keeps only the band from 0.17·fl to 0.59·fl of its nominal length — it
    # throws away three fifths of what it is given — so `fl = 0.45·wl` put a
    # 0.19·wl mark on the flank. Measured against the finished sheet that is
    # a fifth of the body: not a rank of flight feathers but a **barcode**,
    # a patch of fine parallel hatching stuck on the side of the animal with
    # nothing leading into it and nothing leading out. The folded primaries
    # of a small bird run from mid-flank to the tail root and they are the
    # second-longest thing on it. So the two ends are named and `fl` is
    # solved for, exactly as the shaft below is: the mark is to start at
    # 0.34·wl and reach the wing tip.
    n = sp["prim"]
    bx = rx + ca * wl * 0.09 + math.cos(a + off) * po.depth * 0.05
    by = ry + sa * wl * 0.09 + math.sin(a + off) * po.depth * 0.05
    #
    # Two things were still wrong with the row, and both of them are about
    # the **tips**, which on a folded wing are the crispest marks the bird
    # has — the one place the brush was lifted cleanly off the paper.
    #
    # The first is 擦. `press` modulates coverage by the *across*-stroke
    # coordinate only, which is correct where the mark is wide: the hairs
    # run out in lines along the drag. At the leading end `tip=0.74` has
    # pulled the radius in to a quarter of it, so the same three or four
    # streak bands now span two or three pixels — and what they print is
    # not 飞白, it is the feather sawn lengthwise into separate pale
    # slivers with square ends. Measured at nine diameters: four of them
    # off the back of every wing, stepping down in a regular staircase,
    # reading as a frayed rope end. Half the streak, and the tip closes.
    #
    # The second is that the staircase is *arithmetic*. `fl`, `fw`, `fa`
    # and `step` are all exact linear functions of `i`, so the five blades
    # are five equally spaced copies — which is a comb, drawn correctly.
    # 参差 is the oldest instruction in the 翎毛 诀 and it costs four
    # `uniform`s. The jitter is deliberately small (a few percent of each
    # dial): enough that no two gaps match, not enough to open the bundle.
    for i in range(n):
        t = i / max(1, n - 1.0)
        fl = wl * (1.46 + 0.22 * t) * rng.uniform(0.955, 1.055)
        fw = po.depth * (0.124 - 0.018 * t) * rng.uniform(0.90, 1.10)
        fa = a + side * (0.015 + 0.105 * t + rng.uniform(-0.016, 0.016))
        step = po.depth * (0.022 - 0.056 * t + rng.uniform(-0.006, 0.006))
        cx = bx + math.cos(fa) * fl * 0.46 + math.cos(a + off) * step
        cy = by + math.sin(fa) * fl * 0.46 + math.sin(a + off) * step
        press(cv, cx, cy, fl, fw, fa, lo, dk, load=0.80 - 0.06 * t,
              grad=0.44, flank=side, rim=0.18, tip=0.74, tail=0.42,
              splay=0.16, bulge=0.10, streak=0.06 + 0.13 * dry,
              bare=0.05, ragged=0.04, alpha=0.54,
              phase=rng.uniform(0, TAU))
        # 翎管 — the shaft. One pale line down a dark feather, and it is the
        # only thing that says "feather" rather than "stroke".
        #
        # It has to end where the *mark* ends, and the mark is much shorter
        # than `fl`: `tip=0.74` cuts three quarters off the leading radius,
        # so a press centred at 0.46·fl with a half-length of 0.5·fl reaches
        # only 0.59·fl. Running the shaft to 0.88 of the nominal length put
        # a pale hairline a third of a feather long out on bare paper beyond
        # every primary and every tail blade — the wire whiskers trailing
        # off the back of every bird on every sheet in the module. Solve for
        # the reach instead of guessing it.
        if fl > 9.0:
            end = 0.46 + 0.5 * (1.0 - 0.74) - 0.06
            sx = bx + math.cos(a + off) * step
            sy = by + math.sin(a + off) * step
            p0 = (sx + math.cos(fa) * fl * 0.23, sy + math.sin(fa) * fl * 0.23)
            p1 = (sx + math.cos(fa) * fl * end, sy + math.sin(fa) * fl * end)
            drag(cv, [p0, p1], max(0.45, fw * 0.12), 0.3,
                 dilute(hi, 0.34), dilute(hi, 0.58), alpha=0.17)

    # 大覆羽 — the middle rank. Without it the covert mass and the flight
    # feathers are two unrelated objects that happen to touch.
    #
    # These used to be drawn straight **across** the wing: three pointed
    # dashes a quarter of the body depth long, hanging in a row out of the
    # bottom of the covert mass. At a sparrow's scale that is not a rank of
    # feathers, it is a set of **teeth**, and it was the single ugliest mark
    # on the bird — proved by ablation, because deleting this loop and
    # changing nothing else took the comb out of the middle of the animal.
    #
    # A rank of coverts is a row of overlapping *tips*. So the mark that
    # says so runs back **along** the wing at a shallow angle to it, is a
    # third as long, sits inside the mass instead of below it, and is laid
    # at an alpha that cannot build an edge of its own (crab's law: these
    # are lines inside a mass that is already painted).
    # Straight dashes could not do it either, for the same reason the
    # primaries could not: a tip is *curved*, and a row of straight marks all
    # at one angle is a row of straight marks at one angle. Each one is now a
    # three-point arc bowed toward the tail, they run from the shoulder to
    # where the primaries emerge so the row leads the eye from the one into
    # the other, and they sit on the mass's own ventral boundary — a third of
    # the depth below the back, which is where `lift_w` put its lower edge —
    # instead of on the axis, where they floated on bare flank like stitches.
    edge = -po.depth * 0.05
    for i in range(4):
        t = 0.26 + 0.13 * i
        d = po.depth * (0.086 - 0.008 * i)
        ox = rx + ca * wl * t + math.cos(a + off) * edge
        oy = ry + sa * wl * t + math.sin(a + off) * edge
        qa = a + off * 0.52
        q = []
        for u in (-0.55, 0.0, 0.58):
            # the bow: the middle of the arc lags toward the head, so the
            # tip of the covert points back along the wing
            bowx = ox - ca * d * 0.34 * (1.0 - u * u)
            bowy = oy - sa * d * 0.34 * (1.0 - u * u)
            q.append((bowx + math.cos(qa) * d * u,
                      bowy + math.sin(qa) * d * u))
        drag(cv, q, max(0.55, po.depth * 0.030), 0.3,
             dilute(hi, 0.50), dilute(dk, 0.78), alpha=0.22,
             d0=0.74, d1=0.20)


def _tail(cv, rng, res, po, st, air, k):
    """尾如剪 — a few flat feathers, fanned, and no plume anywhere.

    Drawn outermost first so the central pair lands on top, which is the
    order the feathers actually overlap. The whole fan `lift`s once rather
    than per feather: a tail is one object passing in front of the wood,
    and lifting each blade separately leaves pale seams between them.
    """
    sp, side = po.sp, po.side
    lo, hi = _ramp(st, sp, k, pale=0.48)
    dk = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k))
    n = sp["tail_n"]
    tl, fan = po.tail_len, sp["fan"]
    ta = po.LD(po.tail_ps)
    tx, ty = po.tail_root
    mid = (n - 1) * 0.5

    # The hole has to be the shape of the thing that fills it. A radius
    # function only loosely related to the fan leaves a disc of lifted
    # paper round the tail — a grey moon behind the bird, and worse on a
    # branch, because the wood vanishes into it.
    # A blade's width off the **body depth** is right until the tail is
    # short. 翠鸟's is three tenths of its body against 翠鸟's deep chest,
    # which gave blades 45 pixels long and 25 wide — ovals, three of them,
    # fanned a quarter radian: a grey thumbprint on the rump, and the
    # 铺底 under it came out wider than it was long and so was back to
    # being the **disc** the per-blade split exists to prevent. A tail
    # feather is a long thin thing in every species; what varies is how
    # many and how far they reach, not the aspect of one blade. So the
    # width is also capped against the tail's own length, which binds on
    # 翠鸟 (31 → 17 pixels) and clears the other three.
    bw = min(po.depth * 0.175, tl * 0.38)

    # 一笔铺底 — the fan is laid as *one* wet mark first and the blades are
    # drawn into it. Without it the tail is five separate presses with the
    # sheet showing between them, which at any size reads as a worn-out
    # brush and at small size as a black comb nailed to the bird. It is also
    # what pays for opening `fan`: the blades can now be drawn pale enough
    # not to stack into a wedge, because the mark that makes the tail *one
    # thing* is no longer their own overlap.
    # And it covers the **root half only**. Running it the whole length of
    # the fan and the whole width of it was the obvious reading of 铺底 and
    # it draws a paddle: one clean ellipse as wide as it is long, with the
    # blades striating its middle and a bare rim of it showing all round
    # them. The blades have to own the silhouette — they are what the tail
    # *is* — so the base mark stops where they take over, and its job is
    # only to keep the five of them from reading as five separate strokes
    # where they leave the rump.
    #
    # It is also *n* marks and not one, which is the repair of the same
    # fault one level down. A single press spanning the fan is `bw +
    # 0.8·tl·sin(fan)` wide against `0.58·tl` long, and on any open fan
    # those two numbers are equal — the mark is a **disc**, and what it
    # draws is a grey moon rising behind the bird with the wood dissolving
    # into it. A press cannot draw a wedge (the 虾蟹 lesson), so the wedge
    # is built *out of* presses: one along each blade, all rooted at the
    # same point, merged into one wet mark at the rump and parting with
    # the blades beyond it.
    for j in range(n):
        fa = ta + side * ((j - mid) / (mid or 1.0)) * fan
        press(cv, tx + math.cos(fa) * tl * 0.22, ty + math.sin(fa) * tl * 0.22,
              tl * 0.52, bw * 0.95, fa, lo, hi,
              load=0.70, grad=0.40, flank=side, rim=0.04, tip=0.34, tail=0.46,
              splay=0.18, bulge=0.10, bare=0.05,
              streak=0.10 + 0.18 * air["dry"], ragged=0.06,
              alpha=0.20, phase=rng.uniform(0, TAU))

    order = sorted(range(n), key=lambda j: -abs(j - mid))
    for j in order:
        e = (j - mid) / (mid or 1.0)
        fa = ta + side * e * fan
        # 剪尾 — `notch` lengthens the outer feathers and shortens the
        # middle ones, which is a forked tail; at zero the fan is square.
        # 参差 — and the lengths are staggered. Five blades cut to the same
        # length by the same `tip` end flush on one smooth convex curve, and
        # what that draws is a **leaf**: a pointed oval with parallel veins
        # in it. A tail ends in five ends.
        fl = (tl * (1.0 + sp["notch"] * (abs(e) - 0.5) * 0.7)
              * rng.uniform(0.90, 1.10))
        fw = bw * (0.82 - 0.08 * abs(e))
        cx = tx + math.cos(fa) * fl * 0.38
        cy = ty + math.sin(fa) * fl * 0.38
        press(cv, cx, cy, fl, fw, fa, lo, dk,
              load=0.80 - 0.10 * abs(e), grad=0.46, flank=side,
              # 根虚 — `tail` was 0.12, so five blades arrived at the
              # rump at very nearly full width and stacked there. Five
              # times alpha 0.66 on one spot is a hard black comma pinned
              # to the bird, and no amount of work on the *ends* of the
              # tail fixes what is wrong at its root.
              rim=0.18, tip=0.60, tail=0.34, splay=0.10,
              streak=0.07 + 0.13 * air["dry"], bare=0.03,
              ragged=0.04 + 0.05 * rng.random(),
              alpha=0.66, phase=rng.uniform(0, TAU))
        if fl > 12.0:
            # Same arithmetic as the primaries' shaft: `tip=0.52` here, and
            # the press is centred at 0.38·fl, so the blade ends at 0.62·fl
            # and the shaft has to stop short of that.
            end = 0.38 + 0.5 * (1.0 - 0.60) - 0.06
            p0 = (tx + math.cos(fa) * fl * 0.06, ty + math.sin(fa) * fl * 0.06)
            p1 = (tx + math.cos(fa) * fl * end, ty + math.sin(fa) * fl * end)
            drag(cv, [p0, p1], max(0.45, fw * 0.16), 0.3,
                 dilute(hi, 0.40), dilute(hi, 0.62), alpha=0.26)

    # 尾上覆羽 — the short feathers that cover the root. Without them the
    # fan is a separate object pinned to the rump.
    for i in range(3):
        e = (i - 1) * 0.5
        fa = ta + side * e * fan * 1.4
        fl = tl * 0.46
        cx = tx + math.cos(fa) * fl * 0.18 - math.cos(ta) * tl * 0.14
        cy = ty + math.sin(fa) * fl * 0.18 - math.sin(ta) * tl * 0.14
        press(cv, cx, cy, fl, po.depth * 0.085, fa, lo, hi, load=0.68,
              grad=0.55, flank=side, tip=0.58, tail=0.20, alpha=0.30,
              phase=rng.uniform(0, TAU))


def _bone(cv, rng, po, st, air, k):
    """背线 — one stroke, bill to tail root, and the breast line that stops.

    Sampled off the same outline the masses were painted from, so it is a
    dark *edge* rather than a contour drawn round a shape: it lies exactly
    where the side-loaded brush would have pooled. Low alpha and a wide
    soft nib — the lesson 墨梅 paid for twice is that a thin even line at
    high alpha reads as a pen however Chinese the shape under it is.
    """
    sp = po.sp
    dk = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k))
    pale = dilute(rgb(st["plume"][0]), 0.75)
    # Scaled off the body *length*, not its depth: depth now carries the
    # cold, and tying the line weight to it gave a puffed bird a 20-pixel
    # grey band round it, which is an outline however softly it is drawn.
    w = max(0.7, po.Lb * 0.016)
    dry = 0.18 + 0.44 * air["dry"]

    def run(pts, w0, w1, a0, a1, alpha):
        n = len(pts) - 1
        if n < 1:
            return
        hws = [max(0.45, w0 + (w1 - w0) * (i / n) ** 1.2) for i in range(n + 1)]
        dens = [clamp01(a0 + (a1 - a0) * i / n) for i in range(n + 1)]
        sweep(cv, pts, hws, dens, pale, dk, alpha, dry=dry, dry_k=2.0,
              dry_ph=rng.uniform(0, TAU), core=0.34)

    if po.back:
        run(po.back, w * 0.72, w * 1.05, 0.58, 0.46, 0.66)
    if po.back2:
        run(po.back2, w * 1.05, w * 0.80, 0.86, 0.52, 0.60)
    # 喉胸 — and then nothing. The belly has no line.
    if po.front:
        run(po.front, w * 0.60, w * 0.95, 0.80, 0.34, 0.44)
    if po.jaw:
        run(po.jaw, w * 0.46, w * 0.26, 0.76, 0.14, 0.42)


def _silk(cv, rng, po, st, k, n=16):
    """丝毛 — the short strokes where the dark back gives out.

    They go on the boundary and nowhere else: a bird drawn with feather
    strokes all over is a drawing of feathers, and the places the eye
    actually looks for them are the flank, the thigh and the fluff under
    the breast of a cold bird.
    """
    sp, side = po.sp, po.side
    c0 = dilute(rgb(st["plume"][0]), 0.52)
    c1 = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k * 0.70))
    for _ in range(n):
        # Only on the waterline, and the waterline is not a guess — it is
        # where `body_tone`'s coverage crosses zero, which is ψ ≈ 0.33 at
        # the rump and ψ ≈ 2.60 at the breast. The first version scattered
        # these over the belly, where there is no ink at all, and a dozen
        # pale hairs hanging off bare paper is the single ugliest thing
        # this module has produced.
        if rng.random() < 0.55:
            ps = rng.uniform(-0.26, 0.30)
        else:
            ps = rng.uniform(2.56, 3.30)
        u = rng.uniform(0.82, 1.00)
        x0, y0 = po.on_body(ps, u)
        # 毛向后下 — contour feathers lie back along the bird, so the stroke
        # leaves at a shallow angle to the outline and never radially.
        a = po.LD(ps) + side * rng.uniform(1.05, 1.45)
        # Short. 丝毛 is a *tuft* at the edge of the ink — a dozen strokes
        # a tenth of a body long, laid side by side at a regular angle,
        # stop being feathers and become a comb drawn on the bird, and on
        # the breast, where they are pale marks on a pale field, a rib
        # cage. Length is the whole difference and it costs nothing.
        L = po.Lb * rng.uniform(0.026, 0.052) * (1.0 + 0.6 * po.fluff)
        x1 = x0 + math.cos(a) * L
        y1 = y0 + math.sin(a) * L
        drag(cv, [(x0, y0), ((x0 + x1) * 0.5, (y0 + y1) * 0.5 - L * 0.08),
                  (x1, y1)],
             max(0.40, po.Lb * 0.012), 0.25, c0, c1,
             alpha=rng.uniform(0.12, 0.26), d0=0.75, d1=0.14)


def _bill(cv, rng, res, po, st, k):
    """嘴如钳 — a wedge that comes to a point, split down the middle.

    It was a `press`, and a press cannot draw this mark. Every silhouette
    dial `press` owns modulates a *radius* about the mark's own centre, so
    the widest place on a press is always its middle — and a bill is widest
    where it leaves the feathers and narrowest at the point. Asking `tip`
    for the point only pulls the leading radius in, which blunts the end
    instead of sharpening it, and (the same arithmetic that cost two rounds
    on the primaries) also stops the mark a quarter of its length short of
    where the tip actually is. Measured at ten diameters the result was a
    black lozenge with rounded ends and a grey hairline floating off one
    corner — a bean, not a pair of pincers.

    So it is a `sweep`: a brush dragged from the feather line to the point
    with its half-width run down to nothing. Three marks, and in this order,
    because that is the order a brush does it in — the wedge, the 口缝 that
    makes it two mandibles, and the 上喙 ridge over the top.
    """
    sp = po.sp
    R0, bl = po.R0, po.bill_len
    hc = rgb(st["bill"])
    pale = mix(hc, (255.0, 255.0, 255.0), 0.34)
    tp = sp["taper"]
    g = sp["gape"]
    kb = po.kbill
    tip = po.HP(0.0, R0 * 0.76 + bl, k=kb)
    up = po.HP(-g * 1.10, R0 * 0.80, k=kb)
    dn = po.HP(g * 1.25, R0 * 0.74, k=kb)
    mx, my = (up[0] + dn[0]) * 0.5, (up[1] + dn[1]) * 0.5
    L = math.hypot(tip[0] - mx, tip[1] - my)
    if L < 1.5:
        return
    W = math.hypot(dn[0] - up[0], dn[1] - up[1]) * (0.52 + 0.48 * (1.0 - tp))
    # The base is measured off the **gape**, which is right for the three
    # short-billed species and silently wrong for the one long-billed one.
    # 翠鸟's gape is the narrowest in the table (0.18) and its bill is two
    # and a half times the longest of the others, so the two dials fight:
    # measured at 平栖, a base half-width of 3.9 pixels carrying a bill 82
    # pixels long — an aspect of **ten to one**, falling under two pixels
    # for the outer third. That is not a dagger, it is a mosquito's
    # proboscis, and it is what the 双栖 sheet showed: two birds with
    # drawn-in hairlines curving off their faces.
    #
    # A bill has a *proportion*. 嘴如钳 is a pair of pincers, and pincers
    # are stout for their length whatever the species; the gape only says
    # how far they open. So the base is floored at a fraction of the bill's
    # own length, and only the long bill ever notices: the floor binds on
    # 翠鸟 (3.9 → 8.8) and clears the other three by two pixels or more.
    hw0 = max(0.60, W * 0.56, L * (0.070 + 0.040 * tp))
    # Two aspects, and the difference between them is a bug that hid inside
    # the fix above. `asp` is the proportion of the mark as it will be
    # *drawn*; `aspr` is the proportion the species' own dials asked for.
    # They differ only where the floor binds — and the floor binds exactly
    # on the bill whose shape the taper exponent below was raised to fix.
    # Feeding the floored aspect back into that exponent cancels most of
    # the raise (翠鸟: 10.4 → 4.7, so 1.15 → 0.88, back to concave), which
    # is why the kingfisher still came out as a black bead with a wire off
    # it. Anything describing the *shape the bill wants* reads `aspr`;
    # anything describing the mark actually on the paper reads `asp`.
    aspr = L / max(1.0, 2.0 * max(0.60, W * 0.56))
    asp = L / max(1.0, 2.0 * hw0)
    # 嘴向下微曲 — the midline droops a little toward the gape, which is
    # what stops a wedge reading as a thorn. Measured along the head's own
    # ventral axis so it keeps working when the bird dips, turns or is
    # drawn facing the other way. A *fixed fraction of the length*, though,
    # is a fixed fraction of a much longer line on a dagger bill, and what
    # keeps a sparrow's bill from being a thorn turns a kingfisher's into a
    # scythe. The sag is a property of the stroke's stoutness, so it is
    # tied to the aspect and the long bill comes out nearly straight.
    # How stout the mark is, 0 at a dagger and 1 at a pair of pincers.
    # Three separate dials turn out to be the same dial — see below.
    stout = clamp01((5.0 - asp) / 2.2)
    bk = 0.085 * min(1.0, 2.6 / max(1.6, asp))
    bowx, bowy = po.evx * L * bk, po.evy * L * bk
    # Where the midline crosses out of the cheek — the feather line, in the
    # midline's own parameter. Tested against the same `Rh` the head was
    # painted from and the same `Rh` the 留白 block below reads, so the
    # three marks agree about where the face ends.
    tr = 0.0
    for i in range(1, 25):
        t = i / 24.0
        x = mx + (tip[0] - mx) * t
        y = my + (tip[1] - my) * t
        ang = math.atan2(y - po.hy, x - po.hx)
        if (math.hypot(x - po.hx, y - po.hy)
                >= _at(po.Rh, po.side * (po.hth - ang)) * 0.985):
            tr = t
            break
    n = 7
    mid, hws, dens = [], [], []
    for i in range(n):
        t = i / (n - 1.0)
        b = 4.0 * t * (1.0 - t)
        mid.append((mx + (tip[0] - mx) * t + bowx * b,
                    my + (tip[1] - my) * t + bowy * b))
        # Below 1, not above: a wedge whose half-width falls faster than
        # linear is thin for most of its length and reads as a scimitar.
        # 嘴如钳 is a stout mark that happens to end in a point.
        #
        # ...for a *short* bill. An exponent under 1 is concave — most of
        # the width is spent in the first third — and over a short mark
        # that just rounds the shoulders, which is what it was chosen for.
        # Over a long one it draws a blob with a wire off it, and no base
        # width fixes that, because the fault is the shape of the fall and
        # not its start. A long bill wants the fall to be *straight*: a
        # true wedge, full depth at the face, dead even to the point. So
        # the exponent rises with the aspect and reaches linear-and-a-bit
        # exactly where the mark gets long enough for the difference to be
        # visible.
        # 根虚 — and here it is the difference between a bill and a bead.
        # The wedge starts at the midpoint of the gape, which is a fifth of
        # a head radius *inside* the cheek, and `sweep` ends every stroke in
        # a round cap. On a short bill that cap is small and buried; on
        # 翠鸟's, floored to seven pixels of half-width, it is a fourteen
        # pixel disc of the darkest ink on the sheet sitting on the lore,
        # fatter than the wedge it feeds — measured at ten diameters, a
        # black ball glued to the face with a dagger coming out from behind
        # it. A bill is widest where it *leaves the feathers*, not where it
        # is rooted in them, so the half-width is ramped up to full over the
        # stretch that is still under the cheek and the root goes in thin.
        root = 1.0 if tr <= 1e-3 else clamp01(t / tr)
        hws.append(max(0.42, hw0 * (0.34 + 0.66 * root)
                       * (1.0 - t) ** min(1.15, 0.70
                                          + 0.09 * max(0.0, aspr - 2.6))))
        dens.append((0.62 + 0.36 * t) * stout + (0.86 + 0.14 * t) * (1.0 - stout))
    # 留白 round the part that is off the skull.
    #
    # 回首 swings the bill back over the bird's own mantle, and a black
    # wedge drawn onto a dark back is not a bill — it is a hole punched in
    # the shoulder, which is exactly what it looked like at six diameters.
    # Multiply compositing cannot put anything in front of anything (the
    # 墨梅 lesson), so the paper is lifted out along the stretch of the
    # midline that lies *outside the head outline* and the wedge is laid
    # into the gap. The test is one `atan2` against the same `Rh` the head
    # was painted from, so it is the same edge in both marks.
    #
    # But *outside the head* is only half the test, and the half that is
    # wrong on its own. On a 侧顾 bird — an eighth of a turn, no more — a
    # couple of samples at the bill's root fall a hair outside `Rh` while
    # still sitting squarely on the cheek, and each one lifts a bright nick
    # there: measured at 88 pixels, 101 levels, right where the bill meets
    # the face. What it draws is a bill *detached from the head*, which is
    # a worse fault than the one the block exists to fix and appears on far
    # more sheets. The lift is only ever wanted where the bill is lying
    # over something **else the bird is made of**, so the second half of
    # the test is the body outline, measured in body-local coordinates
    # against the same `Rb`. In profile the bill leaves the skull into bare
    # paper, every sample fails both halves, and the block costs nothing.
    s, cth, sth = po.side, po.cth, po.sth
    m = 15
    for i in range(m):
        t = i / (m - 1.0)
        j = t * (n - 1.0)
        a0 = int(j)
        a1 = min(n - 1, a0 + 1)
        g = j - a0
        x = mid[a0][0] + (mid[a1][0] - mid[a0][0]) * g
        y = mid[a0][1] + (mid[a1][1] - mid[a0][1]) * g
        w = (hws[a0] + (hws[a1] - hws[a0]) * g) * 1.16 + 0.45
        ang = math.atan2(y - po.hy, x - po.hx)
        rr = _at(po.Rh, po.side * (po.hth - ang))
        if math.hypot(x - po.hx, y - po.hy) < rr * 0.94:
            continue
        dx, dy = x - po.cx, y - po.cy
        bx = dx * cth + dy * sth
        by = s * (dy * cth - dx * sth)
        # Faded by how far in it is, so the 留白 dies at the mantle's edge
        # instead of stopping on a line of its own.
        bur = clamp01((_at(po.Rb, math.atan2(by, bx)) - math.hypot(bx, by))
                      / (0.10 * po.Lb))
        if bur <= 0.02:
            continue
        res.lift(cv, x, y, lambda f, w=w: w, alpha=0.66 * bur,
                 feather=1.1, na=28)
    # 墨分五色 across the width is a property of a **wide** mark.
    #
    # `core` pales the rim because a round brush carries its load in the
    # middle, and `side` tilts the load onto one flank. Both read correctly
    # on a sparrow's bill, where the wedge is seven pixels of half-width at
    # the face and the rim is a fraction of it. On a kingfisher's they are
    # a disaster, and for a reason worth writing down: **as a stroke tapers
    # to nothing, every pixel of it becomes rim.** Past the half-way point
    # the half-width is under the feather distance, `lat` is pinned at 1
    # everywhere across, and the whole outer bill is multiplied by
    # `1 - core`. What that draws — measured at eight diameters on a 俯枝
    # 翠鸟 — is a black blob at the gape with a *hollow outlined* prong
    # running out of it, dark along both edges and pale down the middle:
    # a bird with its beak open. The same taper is why the 上喙 ridge has
    # to go: a hard line laid along the top of a mark two pixels wide is
    # not a ridge on the bill, it is a second edge beside it.
    #
    # So all three are tied to the one number. A stout bill keeps the
    # modelling it was tuned for; a dagger is laid flat and solid, which
    # is how a dagger bill is painted anyway.
    sweep(cv, mid, hws, dens, pale, hc, 0.95, core=0.22 * stout,
          side=0.30 * po.side * stout, feather=0.8)
    # 口缝 — the join of the two mandibles, and the one mark that turns a
    # wedge into a bill. It runs *inside* the wedge, not along its edge.
    j0, j1 = int(n * 0.18), n - 1
    drag(cv, mid[j0:j1 + 1], max(0.40, hw0 * 0.17), 0.3, pale, hc,
         alpha=0.52, d0=0.84, d1=0.30)
    # 上喙 — the culmen, the one hard line on the bird, run along the top
    # edge of the wedge rather than beside it. Stopped short of the point:
    # a ridge line carried right to the tip crosses the gape line there and
    # draws a little cross-hair on the end of every bill.
    ux, uy = -po.evx, -po.evy
    cul = [(mid[i][0] + ux * hws[i] * 0.80, mid[i][1] + uy * hws[i] * 0.80)
           for i in range(n - 1)]
    drag(cv, cul, max(0.45, hw0 * 0.20), 0.4, pale, hc,
         alpha=0.42 * (0.22 + 0.78 * stout), d0=0.54, d1=0.72)
    # 口角 — and the gape runs *back* from the corner into the cheek
    # feathers and dies there. Running it forward along the wedge only
    # draws the wedge's own lower edge a second time, in a thinner line.
    cor = po.HP(g * 1.20, R0 * 0.86, k=kb)
    into = po.HP(g * 0.60 + 0.70, R0 * 0.80)
    drag(cv, [mid[j0], cor, into], max(0.40, hw0 * 0.16), 0.28, pale, hc,
         alpha=0.40, d0=0.78, d1=0.10)


def _eye(cv, rng, res, po, st, k, glare=0.0, gaze=0.0):
    """眼有神 — a bean of paper, a ring round it, and a pupil off centre.

    The pupil's offset is the bird's whole expression and it costs two
    lines. 八大's version squares the ring and jams the pupil into the top
    corner — 白眼向人, the look that made a bird a political statement —
    and that is one extra term in the radius plus a bigger offset.
    """
    er = po.eye_r
    # 点睛 is the last and the darkest mark on the animal, and it was being
    # laid in the same thinned ink as the plumage. Measured off a finished
    # sheet at twelve diameters: the rim came out at 55, the pupil at 70 and
    # the skull around them at 110 — three values inside a sixth of the
    # scale, with the *rim* the darkest of the three. What that draws is a
    # washer. An eye is a bead of 焦墨 with a rim round it and one spark of
    # paper in it, and the bead has to be the darkest thing on the bird or
    # the bird has no look in it; the rim is the pale one, because it is
    # eyelid, not pupil.
    pup = rgb(st["plume"][1])
    ink = dilute(pup, clamp01(0.40 + 0.28 * k))
    pale = dilute(rgb(st["plume"][0]), 0.60)

    if glare > 0.3:
        def rad(f):
            ps = po.side * f
            c, s = abs(math.cos(ps)), abs(math.sin(ps))
            q = (c ** 5.0 + s ** 5.0) ** -0.2
            return er * (1.0 + 0.20 * glare) * (1.0 + 0.58 * glare * (q - 1.0))
    else:
        def rad(f):
            # 眼如豆 — a bean, along the head, not a drawn circle.
            return er * (1.0 + 0.13 * math.cos(2.0 * po.side * f))

    # The sclera is a *sliver*, not a saucer. Lifting a disc 1.18 times the
    # eye back to clean paper and then dropping a small pupil in the middle
    # of it draws the one thing a brush painter never draws: a white ball
    # with a dot on it. On a dark head it is the brightest object on the
    # sheet and the bird turns into a toy. 八大 is the limiting case and
    # proves the rule — 白眼向人 is a *crescent* of paper left at one corner
    # of a square rim, visible precisely because the rest of the eye is
    # ink. So the hole shrinks to the eye itself and gives up a third of
    # its opacity, and the pupil grows to fill most of what is left.
    # And the paper under it is a **crescent**, not an annulus. Lifting
    # the full bean and then dropping a concentric pupil in it leaves a
    # ring of bright sheet all the way round the bead, which at ten
    # diameters is a bullseye — the one shape that makes a painted bird
    # look like a toy, and the thing 白眼向人 is the proof of: 八大 leaves
    # paper at *one corner* of the eye and inks the rest. So the hole is
    # smaller than the bead that goes in it and the bead is pushed off
    # centre, which leaves the sheet showing on one side only.
    # `feather` here is a *radius* in pixels, and at half the eye it
    # restored paper for half an eye's width outside the hole in every
    # direction -- a bright ring all the way round the bead, which is the
    # bullseye again by another route. The crescent has to have an edge on
    # the far side too: the sclera meets the lid, it does not dissolve into
    # the skull. One pixel of feather, and no more.
    # And the hole is **inside the lid**. `0.88 + 0.16 * glare` put it at
    # 1.04 of the ring's own radius on a 白眼 bird, so the paper came out
    # past the eyelid on every side: a white ball sitting proud of the
    # skull with a dot on it — a googly eye, which is the one thing worse
    # than the bullseye it replaced. 眼眶 is the outer edge of the eye and
    # nothing lighter than the skull may appear outside it.
    res.lift(cv, po.ex, po.ey, lambda f: rad(f) * (0.90 - 0.06 * glare),
             rot=po.hth, alpha=0.56 + 0.30 * glare,
             feather=max(0.5, er * 0.16))
    ring = [(po.ex + math.cos(po.hth + po.side * (i * TAU / 26)) * rad(i * TAU / 26),
             po.ey + math.sin(po.hth + po.side * (i * TAU / 26)) * rad(i * TAU / 26))
            for i in range(27)]
    w = max(0.45, er * (0.16 + 0.14 * glare))
    sweep(cv, ring, [w] * 27, [0.86] * 27, pale, ink, 0.80 + 0.18 * glare,
          core=0.26)
    # 瞳 — forward and up by default; hard into the corner when glaring.
    # *Hard* into it: the whole of 白眼向人 is that the bead is not in the
    # middle of the eye. A small pupil jammed against the upper rim leaves
    # the paper as one large angular region, which is the look; a large
    # pupil near the middle leaves a ring, which is a doll.
    # The direction is built as a **screen vector**, not as an angle off
    # `look` plus a multiple of `side`. The angle form was wrong twice for
    # the same reason: it needs the sign of `side`, the sign of the yaw and
    # the sign of screen y all to agree, and at 回首 — the one pose where
    # the projected bill reverses — they do not. Measured on a 孤禽 麻雀
    # looking back: the 白眼 bead came out in the eye's lower *front*
    # corner, a bird staring at its own feet, which is the exact opposite
    # of the one thing the mark exists to say. Two unit vectors added
    # cannot get this wrong, because neither of them has a sign to get
    # wrong: away from the bill (or along it, for a merely alert bird),
    # plus straight up the sheet.
    bx, by = math.cos(po.look), math.sin(po.look)
    if glare > 0.3:
        dx, dy = -bx * 0.56, -by * 0.56 - 1.00      # 白眼向人 — up and away
    else:
        dx, dy = bx * 0.80, by * 0.80 - 0.78        # alert — along the bill
    pa = math.atan2(dy, dx) + po.side * gaze
    pr = er * (0.80 - 0.26 * glare)
    # And it is **inside the lid**. `d` was a flat multiple of the eye, so
    # on a 白眼 bird the bead's centre sat further out than the sclera's
    # own edge and three quarters of the pupil landed on the eyelid: what
    # that draws is not a pupil in a corner, it is a square notch bitten
    # out of the ring, with the paper left over reading as a bare white
    # blob. 八大's bead *touches* the rim and stops. So the offset is
    # solved from the lid's radius in the direction the bead is going —
    # the same `rad` the ring is drawn from, so the two cannot disagree.
    lid = rad(po.side * (pa - po.hth)) * (0.90 - 0.06 * glare)
    # A fifth of an eye of offset is not an offset. The bead ended up very
    # nearly concentric and the paper came round it through two hundred
    # degrees, which at three diameters is the bullseye again — a target,
    # not a look. A third of an eye puts the crescent decisively on one
    # side, where 高光 and 留白 belong on a sphere lit from above.
    d = min(er * (0.64 * glare + 0.34 - 0.13 * glare),
            max(0.0, lid - pr * (0.80 if glare > 0.3 else 0.62)))
    px = po.ex + math.cos(pa) * d
    py = po.ey + math.sin(pa) * d
    disc(cv, px, py, pr, pup, 0.98, soft=0.18)
    disc(cv, px, py, er * (0.52 - 0.14 * glare), pup, 0.98)
    # 高光 — one spark of paper on the far side of the bead, lifted rather
    # than painted, because under a multiply composite there is no white.
    # It is two pixels across on a sparrow and it is the difference between
    # an eye and a drilled hole. 白眼 does not get one: there the paper is
    # already the larger half of the eye and a second highlight inside the
    # pupil only muddles the crescent that *is* the expression.
    if glare <= 0.3:
        # On the *same* side as the pupil's offset, not the far side. The
        # far side is where the sclera already shows, so a spark there
        # merges with it and the two together draw a comma. Inside the bead,
        # up and forward, with the paper crescent opposite: that is an eye.
        gx = px + math.cos(pa) * er * 0.30
        gy = py + math.sin(pa) * er * 0.30
        res.lift(cv, gx, gy, lambda f: er * 0.17, alpha=0.85,
                 feather=max(0.4, er * 0.16))


# --------------------------------------------------------------------------
# 爪 — the grip, and why it is the last thing drawn on the sheet.
#
# Every painted bird that looks wrong and cannot be faulted is standing on
# its branch rather than holding it. The difference is not anatomy, it is
# **order**: the toes have to be in front of the wood. This family composites
# by multiply, so being in front is not something draw order grants — a mark
# laid later still lets everything under it print through. But a *dark* mark
# on pale wood is the one case where multiply tells the truth, because
# multiplying dark by anything is dark. So the bird's masses go behind by
# lifting the wood out of themselves, and the claws go in front by being
# drawn last and nearly black. Two opposite techniques on the same creature,
# each used where it is honest.
#
# The toes themselves wrap a circle. The branch has a half-width at the perch
# node; treat it as a cylinder seen side-on, put its axis at `perch - n·hw`,
# and walk each toe round that circle by its own arc. That is the whole
# model, and it is why the grip survives a change of branch thickness: a
# sparrow on a twig closes its foot right round, the same sparrow on a trunk
# barely curls, and neither case is special-cased anywhere below.
# --------------------------------------------------------------------------

def _foot(cv, rng, po, st, k, ax, ay, ta, hw, near=1.0):
    """One foot: 跗 (the tarsus), three toes forward, one back, four claws."""
    side = po.side
    fc = rgb(st["foot"])
    pale = mix(fc, (255.0, 255.0, 255.0), 0.34 + 0.14 * (1.0 - near))
    dk = mix(fc, (0.0, 0.0, 0.0), 0.30)
    # Off the body *length*. Tied to depth, a puffed sparrow grew a leg two
    # centimetres thick — the cold makes a bird rounder, not stouter-legged.
    aw = max(0.8, po.Lb * (0.0140 + 0.0050 * near))

    # 跗 — from the belly feathers down to the ankle, kinked back once. A
    # bird's tarsus is not a straight post and drawing it as one is the
    # second-commonest reason a perched bird looks propped up.
    r0 = po.L2S(*po.leg_l)                        # out of the belly feathers
    if near < 0.5:
        # 远足 — only the bottom of it shows. Drawn full length it is a
        # second leg of equal length splayed off the same hip, and a pale
        # one, so it renders as a wire triangle beside the real leg.
        r0 = (r0[0] + (ax - r0[0]) * 0.46, r0[1] + (ay - r0[1]) * 0.46)
    # 跗 kinks back by a fraction of the *branch's* radius, which on a
    # short-legged bird astride a fat bough is most of the leg: 翠鸟's
    # tarsus is a ninth of its body and came out a sharp dogleg, two
    # straight pieces meeting at a corner halfway down. The kink is a
    # property of the leg, so it is capped against the leg's own span.
    #
    # ...and a cap alone is only half of it. Tied to the branch, the kink
    # is also *absent* on a thin twig — `hw * 0.30` on a six-pixel bough is
    # under two pixels, so the near leg of a bird on a willow wand is a
    # ruled post, which is the fault this line was written to prevent. The
    # lean back is a property of the leg at both ends: floored against the
    # leg's own span so it is always there, capped against it so it never
    # becomes a dogleg.
    span_ = math.hypot(ax - r0[0], ay - r0[1])
    kink = min(span_ * 0.17, max(span_ * 0.075, hw * 0.30))
    midx = (r0[0] + ax) * 0.5 - math.cos(ta) * side * kink
    midy = (r0[1] + ay) * 0.5
    # 根虚 — and `drag` cannot say it, which is why this one stroke is a
    # `sweep`. A tarsus drawn from `leg_l` downward begins with the round
    # cap every capsule brush ends in: a blunt dark stud sitting *on* the
    # belly, full width, full density, at the one join in the bird where
    # the feathers are supposed to swallow the leg. `drag` ramps width and
    # density together from one end to the other, so it cannot be fat in
    # the middle and faint at the top. Four nodes and an explicit pair of
    # ramps can: a thin, almost dry stub pushed a sixth of a body depth up
    # *inside* the belly, widening to the full tarsus where the ink leaves
    # the feathers, then tapering to the ankle as before.
    #
    # The root node has to be measured from **`r0`**, not from the hip. For
    # the near leg they are the same point; for the far one `r0` has
    # already been walked 46 % of the way down the leg (远足 — only the
    # bottom shows), and anchoring the stub at `po.leg_l` puts it back,
    # which is the whole shortening undone. That is what the sheet showed:
    # two stilts of equal length splayed off one hip, the back one pale
    # ochre, reading as a bamboo cane leaning against the bird.
    if near > 0.5:
        up = po.L2S(po.leg_l[0] - po.Lb * 0.010,
                    po.leg_l[1] - po.depth * 0.17)
    else:
        d_ = max(1e-6, span_)
        up = (r0[0] - (ax - r0[0]) / d_ * po.depth * 0.05,
              r0[1] - (ay - r0[1]) / d_ * po.depth * 0.05)
    w = aw * (1.30 if near > 0.5 else 1.02)
    # 上粗下细 — the ankle is the thin end. Held at 0.85 the tarsus came out
    # a ribbon with two parallel edges, which is a strut and not a leg.
    #
    # `core` has to come down for this one stroke, and the reason is the
    # same one that made the bill a bead: a round brush carries its load in
    # the middle, so `sweep` pulls the rim back toward 淡 — and on a mark
    # two pixels wide *every* pixel is rim. On a 620 px sheet the tarsus is
    # `Lb * 0.019` ≈ 2 px, and at the default `core=0.30` with the densities
    # below it printed a pale thread with the toes — `drag`, alpha 0.94,
    # no rim pullback — hanging off the bottom of it as a dark knot. The
    # foot gripped and the leg did not reach it. A leg is a *bone*: narrow,
    # but as dark as the claw it carries.
    sweep(cv, [up, r0, (midx, midy), (ax, ay)],
          [w * 0.62, w, w * 0.84, aw * 0.60],
          [0.08, 0.62, 0.84, 0.98],
          pale if near < 0.5 else fc, fc if near < 0.5 else dk,
          0.92 * (0.52 + 0.48 * near), core=0.12)
    # 鳞 — two or three scale ticks across it, which is all the detail a
    # 写意 leg gets and all it needs.
    #
    # Evenly spaced, dead straight, and spanning the full width, three of
    # them turn the tarsus into a *ladder* — which at twelve diameters is
    # exactly what it looked like. 鳞 in 写意 is a flick of the brush: it
    # sits across part of the leg, not from edge to edge, and no two are
    # the same length or quite parallel.
    pa0 = math.atan2(ax - r0[0], -(ay - r0[1]))
    for i in range(1 + 2 * (near > 0.5)):
        t = 0.38 + 0.21 * i + rng.uniform(-0.045, 0.045)
        sx = r0[0] + (ax - r0[0]) * t
        sy = r0[1] + (ay - r0[1]) * t
        pa = pa0 + rng.uniform(-0.30, 0.30)
        hl = aw * rng.uniform(0.52, 0.86)
        off = rng.uniform(-0.25, 0.25) * aw
        cx3 = sx + math.cos(pa) * off
        cy3 = sy + math.sin(pa) * off
        drag(cv, [(cx3 - math.cos(pa) * hl, cy3 - math.sin(pa) * hl),
                  (cx3 + math.cos(pa) * hl, cy3 + math.sin(pa) * hl)],
             aw * 0.30, aw * 0.24, pale, dk,
             alpha=rng.uniform(0.24, 0.38) * near)

    if near < 0.5:
        # 远足 stops at the wood — but not in mid-air. One dark tick where
        # the hidden toes close, or the far leg ends as a cut-off post.
        drag(cv, [(ax - math.cos(ta) * hw * 0.55, ay - math.sin(ta) * hw * 0.22),
                  (ax + math.cos(ta) * hw * 0.70, ay + math.sin(ta) * hw * 0.28)],
             aw * 0.62, aw * 0.30, fc, dk, alpha=0.46, d0=0.55, d1=0.95)
        return

    # The cylinder the toes close around.
    nx, ny = math.sin(ta), -math.cos(ta)          # one of the two normals
    if (ax - po.cx) * nx + (ay - po.cy) * ny < 0.0:
        nx, ny = -nx, -ny                         # ...the one *away* from it
    ox, oy = ax + nx * hw, ay + ny * hw           # the axis, inside the wood
    a0 = math.atan2(ay - oy, ax - ox)
    # Which way round is forward? Toward wherever the bill points.
    fa = po.LD(math.pi)
    sg = 1.0 if (-math.sin(a0) * math.cos(fa)
                 + math.cos(a0) * math.sin(fa)) > 0.0 else -1.0

    # 三前一后 — three toes forward, one behind. The middle one is longest
    # and the outer two are splayed along the branch, which is the only
    # thing in the foot that says the branch is round.
    #
    # A toe has a *length*, not an arc. The first version gave each one a
    # fixed sweep of the cylinder, which means a bird on a twig gets a foot
    # the size of the twig — four marks three pixels long, invisible, and
    # the sheet shows a bird hovering over its own perch. So: spend the
    # toe's length on wrapping as far as it will go, cap the wrap at a bit
    # over a right angle, and let whatever length is left run on tangent.
    # Thin twig: the toes close right round it and the claws meet beneath.
    # Thick bough: the same toes lie flat along the top and barely curl.
    # Nothing is special-cased; the budget does it.
    toes = ((0.74, 0.62, 0.86), (1.00, 0.04, 1.00), (0.84, -0.52, 0.90),
            (-0.80, -0.10, 0.82))
    unit = po.Lb * 0.125
    for span, lat, wk in toes:
        TL = unit * abs(span)
        # The cap used to be 1.15 rad — a right angle and a bit — and on
        # a twig thinner than the toe is long that spends two thirds of
        # the toe's length on the straight `run`, which leaves the wood
        # at the branch's shoulder and dangles below it in bare paper.
        # Four spider legs under a bird. But the opposite cap is just as
        # wrong: at 2.3 rad every toe closes right round and the four of
        # them draw a dark *hoop* with the branch inside it, which is
        # anatomically defensible and reads as a bracelet. A painted foot
        # shows the toes over the near shoulder with the tips just curling
        # under, so: a little over a right angle for the front three, less
        # for the hallux, and damp whatever length is left rather than
        # letting it hang off into bare paper.
        cap = 1.62 if span > 0.0 else 1.18
        arc = min(cap, TL / max(1.2, hw))         # how much of it wraps
        run = max(0.0, TL - arc * hw) * 0.34      # and how much runs on
        sgn = sg * (1.0 if span > 0.0 else -1.0)
        # 参差 — splay measured off the *toe*, not off the branch. Tied to
        # `hw` the three front toes of a bird on a twig land within two
        # pixels of each other and print as one fat toe.
        spl = max(hw * 0.70, unit * 0.44)
        lx, ly = math.cos(ta) * lat * spl, math.sin(ta) * lat * spl
        pts = []
        for i in range(5):
            t = i / 4.0
            a = a0 + sgn * arc * t
            r = hw * (1.04 - 0.04 * t)
            pts.append((ox + math.cos(a) * r + lx * t,
                        oy + math.sin(a) * r + ly * t))
        if run > hw * 0.15:
            # tangent at the end of the wrap, carrying on round and down
            a = a0 + sgn * arc
            tgx, tgy = -math.sin(a) * sgn, math.cos(a) * sgn
            ex, ey = pts[-1]
            for t in (0.55, 1.0):
                pts.append((ex + tgx * run * t - math.cos(a) * run * t * 0.26,
                            ey + tgy * run * t - math.sin(a) * run * t * 0.26))
        drag(cv, pts, aw * 0.96 * wk, aw * 0.38, pale, mix(fc, dk, 0.28),
             alpha=0.94, d0=0.60, d1=0.94)
        # 爪 — the claw, hooked off the end of the toe and into the wood.
        p1, p0 = pts[-1], pts[-2]
        ca2 = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) + sgn * 0.85
        cl = max(2.4, unit * 0.34)
        cx2, cy2 = p1[0] + math.cos(ca2) * cl, p1[1] + math.sin(ca2) * cl
        mx2 = p1[0] + math.cos(ca2 - sgn * 0.45) * cl * 0.55
        my2 = p1[1] + math.sin(ca2 - sgn * 0.45) * cl * 0.55
        drag(cv, [p1, (mx2, my2), (cx2, cy2)], aw * 0.52, 0.3, fc, dk,
             alpha=0.96, d0=0.80, d1=1.0)


def _thigh(cv, rng, po, st, k):
    """胫羽 — the feathered thigh the tarsus comes out of.

    Geometry says the leg starts inside the bird. Ink says otherwise: the
    belly is 虚 by design, so at the depth a leg leaves the body there is
    almost no ink left, and the tarsus begins on bare paper a finger below
    the last grey. That gap is the commonest fault in a painted perching
    bird and the hardest one to name — the creature reads as *mounted*,
    on two tan stilts, and nothing about the stilts is wrong.

    So a dozen short strokes in body ink, laid down and back over the top
    of the leg after the leg is drawn. They are the same marks as `_silk`
    and deliberately so: this is 丝毛 at the one place anatomy puts a tuft.
    """
    sp, side = po.sp, po.side
    c0 = dilute(rgb(st["plume"][0]), 0.46)
    c1 = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k * 0.60))
    lx, ly = po.leg_l
    # Narrow. The first dose used a half-width of a twentieth of the
    # body depth against a length of a twelfth of its *length*, which is a
    # mark very nearly as wide as it is long — fourteen of those over the
    # 留白 of the belly is not a tuft, it is a stain.
    hw0 = max(0.50, po.depth * 0.021)
    # 墨不到处笔不到, again, and this is the place it cost the most. The fan
    # was sampled from a span of the belly and given a length of up to an
    # eighth of the body, with no test of where it *ended* — so a third of
    # the strokes ran out through the contour and hung in the bare paper
    # beside the leg. Pale, soft, vertical, and twice as wide as the tarsus
    # they were supposed to be hiding the root of: at nine diameters the
    # bird is standing in a column of smoke. A tuft is not a mist. It is
    # short, it is as dark as the belly it grows out of, and every hair of
    # it is *on the bird*. So both ends are tested against the same `Rb`
    # the body was painted from — the start well inside, the tip allowed a
    # couple of percent past the edge and no more — and the stroke is
    # shortened until that is true rather than being dropped, because a
    # fan with a third of its strokes missing is a fan with a hole in it.
    for _ in range(int(11 + 9 * po.fluff)):
        bx = lx + po.Lb * rng.uniform(-0.052, 0.034)
        by = ly - po.depth * rng.uniform(0.06, 0.30)
        if math.hypot(bx, by) > _at(po.Rb, math.atan2(by, bx)) * 0.90:
            continue
        ps = math.pi * 0.5 - rng.uniform(0.05, 0.58)
        cs, sn = math.cos(ps), math.sin(ps)
        L = po.Lb * rng.uniform(0.050, 0.098)
        for _ in range(5):                        # walk it back to the edge
            ex, ey = bx + cs * L, by + sn * L
            if math.hypot(ex, ey) <= _at(po.Rb, math.atan2(ey, ex)) * 1.02:
                break
            L *= 0.74
        if L < po.Lb * 0.022:
            continue
        a = po.LD(ps)
        x0, y0 = po.L2S(bx, by)
        drag(cv, [(x0, y0),
                  (x0 + math.cos(a) * L * 0.55 - math.sin(a) * L * 0.07 * side,
                   y0 + math.sin(a) * L * 0.55 + math.cos(a) * L * 0.07 * side),
                  (x0 + math.cos(a) * L, y0 + math.sin(a) * L)],
             hw0 * rng.uniform(0.70, 1.20), hw0 * 0.24, c0, c1,
             alpha=rng.uniform(0.30, 0.58), d0=0.40, d1=0.95)


def _legs(cv, rng, po, st, k, ax, ay, ta, hw):
    """Both feet. The far one is a pale stub behind the near one.

    Drawing two complete feet side by side is what makes a painted bird
    stand like a chicken: the far leg is almost entirely hidden by the
    belly feathers and whatever of it shows is half a tone paler. One line
    of difference, and it is the difference between two legs and a pair.
    """
    side = po.side
    # Far enough along the branch to *show*. A fiftieth of the body length
    # is two or three pixels, and pushed half a branch-radius down behind
    # the wood on top of that the far leg was not pale, it was absent —
    # eleven strokes a sheet costing runtime and drawing nothing. A pair of
    # feet reads as a pair when the back one is visible past the front one
    # and plainly weaker, so: a step of a twentieth of the body along the
    # bough, and only a token push behind it.
    off = math.cos(ta) * po.Lb * 0.050, math.sin(ta) * po.Lb * 0.050
    nx, ny = math.sin(ta), -math.cos(ta)
    if ny < 0.0:
        nx, ny = -nx, -ny                         # into the wood
    _foot(cv, rng, po, st, k,
          ax + off[0] * side + nx * hw * 0.30,
          ay + off[1] * side + ny * hw * 0.30, ta, hw, near=0.40)
    _foot(cv, rng, po, st, k, ax - off[0] * side, ay - off[1] * side,
          ta, hw, near=1.0)
    _thigh(cv, rng, po, st, k)


# --------------------------------------------------------------------------
# 一禽 — the whole bird, in the order the hand paints it.
# --------------------------------------------------------------------------

def bird(cv, rng, res, po, st, air, k, perch=None, glare=0.0):
    """头、背、翅、尾、嘴、眼、爪 — and the claws last, over the wood."""
    sp = po.sp
    _mass(cv, rng, res, po, st, air, k)
    # 头顶 — before the patches, because 白头翁's white nape is a hole cut in
    # the hood and the hood has to exist first.
    _cap(cv, po, st, k)
    # 颊 / 枕 — the pale patches, cut out of the hood before anything is
    # drawn over it. 麻雀 is a cheek spot and 白头翁 is the nape it is named
    # for; both are holes in the ink, not paint.
    if sp["cheek"] > 0.05:
        _patch(cv, res, po, 1.62, po.R0 * (0.20 + 0.13 * sp["cheek"]),
               alpha=0.34 + 0.20 * sp["cheek"])
    if sp["nape"] > 0.05:
        _patch(cv, res, po, -2.05, po.R0 * (0.24 + 0.20 * sp["nape"]),
               alpha=0.52 * sp["nape"] + 0.12)
    _wing(cv, rng, res, po, st, air, k)
    _tail(cv, rng, res, po, st, air, k)
    _bone(cv, rng, po, st, air, k)
    _silk(cv, rng, po, st, k,
          n=int(9 + 11 * po.fluff + 5 * max(0.0, po.Lb / 110.0 - 1.0)))
    # 麻雀 wears its name: 斑. A dozen dark ticks on the mantle and the
    # covert mass, and the sparrow stops being a generic small bird.
    if st["accent"] and sp["cn"] == "麻雀":
        dk = dilute(rgb(st["plume"][1]), clamp01(sp["tone"] * k))
        for _ in range(10):
            ps = rng.uniform(-2.55, -0.60)
            x0, y0 = po.on_body(ps, rng.uniform(0.52, 0.88))
            a = po.LD(ps) + po.side * 1.25
            L = po.Lb * rng.uniform(0.035, 0.065)
            drag(cv, [(x0, y0), (x0 + math.cos(a) * L, y0 + math.sin(a) * L)],
                 max(0.45, po.depth * 0.038), 0.3,
                 dilute(dk, 0.55), dk, alpha=rng.uniform(0.26, 0.48))
    _bill(cv, rng, res, po, st, k)
    _eye(cv, rng, res, po, st, k, glare=glare,
         gaze=rng.uniform(-0.12, 0.22))
    if perch is not None:
        _legs(cv, rng, po, st, k, *perch)


# --------------------------------------------------------------------------
# 枝 — the perch.
#
# Borrowed whole from 墨梅, because 梅's wood is the family's only 鹿角 wood
# and a perch wants exactly that: straight runs that break direction at a
# joint, thick enough at the near end to be held. Writing a second wood
# generator here would have produced a worse one and left two to maintain.
# What this module adds is only the question 梅 never has to ask — *which
# point of this tangle can be stood on* — and the leaves, since a plum sheet
# has none and a bird on bare wood in April is an odd picture.
# --------------------------------------------------------------------------

def _entry(rng, sc, W, H, S, side):
    """Where the wood comes in and which way it grows."""
    e = sc["entry"]
    L = S * rng.uniform(*sc["L"])
    if e == "side":
        y = H * rng.uniform(*sc["perch"]) + S * rng.uniform(0.02, 0.09)
        if side > 0:
            x, a0 = -S * 0.03, rng.uniform(-0.30, 0.08)
        else:
            x, a0 = W + S * 0.03, math.pi - rng.uniform(-0.30, 0.08)
        return x, y, a0, L, a0, 0.30
    if e == "top":
        x = W * (rng.uniform(0.16, 0.38) if side > 0
                 else rng.uniform(0.62, 0.84))
        a0 = math.pi * 0.5 - side * rng.uniform(0.34, 0.76)
        return (x, -S * 0.03, a0, L,
                math.pi * 0.5 - side * rng.uniform(0.55, 1.00), 0.55)
    # 孤禽 never gets here: its perch is a rock, not wood, and `compose`
    # calls `_stump` instead of `wood_path`. Kept honest rather than silent.
    raise AssertionError("entry %r has no wood" % e)



def _stump(cv, rng, res, st, air, k, x0, W, H, S, old=1.0):
    """八大's stump — a short broad rock, not a bough stood on its end.

    The 孤禽 perch used to be the generic 梅 bough with ``entry="foot"``, and
    that was wrong in a way only visible once the bird got good enough to
    look past it. A bough is a *round tapering* thing drawn as a chain of
    knots, so stood upright it reads as a vertical string of beads; and its
    皴 — sixty short strokes scattered over a column eight times taller than
    it is wide — reads as wire mesh. 八大 paints no post. He paints a short
    broad rock, broken flat across the top, steep on one flank and long on
    the other, running off the bottom edge, with the bird on the break.

    The three things a stump needs that a bough cannot give are all shape,
    not ink:

    * it is **wide** — one part in three of its height, not one in eight;
    * its edges are **straight between kinks**, because 石不宜圆: a rock is
      facets and a lump is a potato. Both flanks are piecewise-linear
      random walks, drawn independently, so the mass is never symmetric;
    * it is **cut off** at the top instead of tapering away, because that
      flat is the thing the claws grip. 爪要抓枝 needs something to抓.

    Everything inside it then follows 坡石's own order — mass first, 勾
    second a little inside the wash (an outline that agrees with a fill to
    the pixel is what makes a shape look cut out of paper), 皴 third and
    only along the facets, 苔 last and only on the crest.

    Returns ``(pts, hws, seat)``: the axis and its half-widths, the same
    contract `bough` has so that `blot`, `flyback` and `moss` go on working
    unchanged, plus the point on the break the bird is to stand on.
    """
    # 宽胜于高. The first version of this was 0.30-0.40 S tall and 0.09 S
    # wide, and it came out a plank: anything above about 1:1.4 reads as a
    # post however it is faceted, because height is the first thing the eye
    # measures. A stump is a *stub* — wider than half its own height.
    Lh = S * rng.uniform(0.20, 0.28)
    ybot = H + S * 0.04
    ytop = H - Lh
    span = ybot - ytop
    wb = S * rng.uniform(0.130, 0.170)
    wt = wb * rng.uniform(0.62, 0.80)
    # 石有势 — the mass leans one way and its axis bows the other, which is
    # the whole difference between a rock that stands and a rock that sits.
    # Both small: a tall mass can be bent and stay a rock, a squat one bends
    # into a banana at a tenth of the amount.
    drift = rng.uniform(0.07, 0.18) * Lh * rng.choice((1.0, -1.0))
    bow = rng.uniform(-0.05, 0.05) * Lh

    n = 56

    def facets():
        """A piecewise-linear multiplier on the half-width: straight runs
        with hard corners between them, which is what a rock's edge is."""
        ks = [rng.uniform(-0.22, 0.13) for _ in range(rng.randint(3, 5) + 1)]
        out = []
        for i in range(n + 1):
            t = i / n * (len(ks) - 1)
            j = min(len(ks) - 2, int(t))
            f = t - j
            out.append(ks[j] + (ks[j + 1] - ks[j]) * f)
        return out

    fL, fR = facets(), facets()
    pts, hws, edge = [], [], []
    for i in range(n + 1):
        t = i / n                                  # 0 at the paper's edge
        ax = x0 + drift * t * t + bow * math.sin(math.pi * t)
        hw = wb + (wt - wb) * t ** 1.2
        pts.append((ax, ybot - span * t))
        hws.append(hw)
        edge.append((ax - hw * (1.0 + fL[i]), ax + hw * (1.0 + fR[i])))

    # 断口 — the break. Not level (a level one is a sawn plank), notched
    # once, and falling away at both corners so the crest has shoulders.
    brk = gb.Noise(rng, 9, 3)
    tilt = rng.uniform(-0.055, 0.055)
    jag = rng.uniform(0.035, 0.085)
    nt, nw, na = rng.uniform(0.2, 0.8), rng.uniform(0.06, 0.13), \
        rng.uniform(0.02, 0.06)

    def top_y(u):
        y = (tilt * (2.0 * u - 1.0) + (brk.at(u, 0.5) - 0.5) * jag
             + na * math.exp(-((u - nt) / nw) ** 2)
             + 0.38 * max(0.0, abs(2.0 * u - 1.0) - 0.74) ** 2)
        return ytop + Lh * y

    # ---- the mass. 石分三面: a loaded flank, a face, and a lit side that is
    # ---- mostly paper. 墨不到处笔不到 — below the threshold nothing is laid.
    lt = dilute(rgb(st["wood"][0]), clamp01(0.50 + 0.42 * k))
    dp = dilute(rgb(st["wood"][1]), clamp01(0.44 + 0.56 * k))
    big = gb.Noise(rng, 5, 5)
    grn = gb.Noise(rng, 19, 17)
    lx = gb.LIGHT[0]
    # 石分三面 — and the division between the faces is a *line on the rock*
    # with corners in it, not a gradient across it. A smooth dark-to-light
    # ramp is a cylinder at every setting of its width, which is how the
    # second pass of this came out a grey drainpipe; a broken boundary is
    # what makes the same wash read as two planes meeting at an edge.
    nb = rng.randint(3, 5)
    bks = [rng.uniform(0.30, 0.64) for _ in range(nb + 1)]

    def plane_u(t):
        f = t * nb
        j = min(nb - 1, int(f))
        return bks[j] + (bks[j + 1] - bks[j]) * (f - j)
    dry = air["dry"]
    # 墨不到处笔不到, and on this mass it cannot be asserted from geometry.
    # The lit face of a rock is *mostly paper* by construction — `a` falls
    # to nothing out there — so a 皴 stroke placed by `u` alone has no way
    # of knowing whether there is any ink under it. Three of them on the
    # 孤禽 sheet came out as pale wavy scratches hanging in the blank below
    # the stump's shoulder, which is the exact fault the 崖兰 cliff cost a
    # rebuild over. So the wash writes down where it went: a coarse
    # (t, u) grid of the strongest alpha laid in each cell, filled as the
    # mass is painted and read back by every stroke that follows. Two
    # integer ops a pixel, and it is the *measured* ink rather than a
    # second model of it, so the two cannot drift apart.
    NT, NU = 28, 18
    ink = [[0.0] * NU for _ in range(NT)]
    buf, cwid, chgt = cv.buf, cv.w, cv.h
    y0 = max(0, int(ytop - Lh * 0.14))
    for py in range(y0, min(chgt, int(ybot) + 1)):
        t = clamp01((ybot - py) / span)
        # Interpolated, not nearest: 57 axis samples over 250 rows means
        # four rows share an edge value, and four-pixel steps down a
        # near-vertical edge are a staircase however well the wash is
        # feathered.
        f = t * n
        i = min(n - 1, int(f))
        g = f - i
        xl = edge[i][0] + (edge[i + 1][0] - edge[i][0]) * g
        xr = edge[i][1] + (edge[i + 1][1] - edge[i][1]) * g
        wid = xr - xl
        if wid < 2.0:
            continue
        pu = plane_u(t)
        for px in range(max(0, int(xl)), min(cwid, int(xr) + 1)):
            u = clamp01((px + 0.5 - xl) / wid)
            ty = top_y(u)
            if py + 0.5 < ty:
                continue
            su = u if lx > 0.0 else 1.0 - u        # 0 at the shaded flank
            g1 = big.at(u, t) - 0.5
            g2 = grn.at(u * 0.93 + 0.04, t * 0.97) - 0.5
            # Three or five kinks in the plane boundary are still three or
            # five straight runs, and over a stump a tenth of the sheet
            # tall that is a ruled diagonal with a soft edge — a cylinder
            # again. The wash's own fine grain is already on the rock and
            # already the right scale, so the boundary is pushed about by
            # it: same noise, so the break in the plane and the mottle on
            # the face cannot disagree, and the edge wavers the way a wet
            # wash drying against a facet does.
            dark = clamp01((pu + g1 * 0.09 + g2 * 0.05 - su) / 0.075)
            rim = (math.exp(-(su / 0.06) ** 2)
                   + 0.45 * math.exp(-((1.0 - su) / 0.05) ** 2))
            crest = math.exp(-((py - ty) / max(2.5, Lh * 0.11)) ** 2)
            sh = (0.15 + 0.50 * dark + 0.14 * (1.0 - su) ** 2
                  + 0.20 * rim + 0.34 * crest + g1 * 0.32 + g2 * 0.18)
            # 飞白 on the long lit face, as skipped paper rather than wash
            a = clamp01((sh - 0.14) / 0.24) * (
                1.0 - (0.28 + 0.52 * dry) * (1.0 - dark)
                * clamp01((g2 + 0.16) / 0.36))
            e = min(px + 0.5 - xl, xr - px - 0.5, py + 0.5 - ty)
            if e < 2.4:
                a *= max(0.0, e / 2.4)
            a *= 0.94
            if a <= 0.012:
                continue
            gi = int(t * (NT - 1))
            gj = int(u * (NU - 1))
            if a > ink[gi][gj]:
                ink[gi][gj] = a
            c = mix(lt, dp, clamp01(sh))
            idx = (py * cwid + px) * 3
            for ch in range(3):
                o = buf[idx + ch]
                buf[idx + ch] = int(o + (o * c[ch] * 0.00392 - o) * a)

    def inked(u, t):
        """How much ink the wash left at (u, t). 0 means bare paper."""
        return ink[min(NT - 1, max(0, int(clamp01(t) * (NT - 1))))][
            min(NU - 1, max(0, int(clamp01(u) * (NU - 1))))]

    # ---- 勾 — the two flanks and the break, each in two or three pulls with
    # ---- gaps, and set a little *inside* the wash.
    gl = dilute(rgb(st["wood"][1]), clamp01(0.52 + 0.40 * k))
    gh = dilute(rgb(st["wood"][1]), clamp01(0.74 + 0.26 * k))
    lw = max(0.9, S * 0.0026)

    # 石不宜圆 gave the silhouette its corners, and on the **lit** flank —
    # where the wash has fallen away to nothing and the line is all there
    # is — those corners came back as a drawn polygon: measured at five
    # diameters on the 孤禽 sheet, three ruled segments of even width
    # meeting at two sharp vertices, standing on bare paper. A rock's
    # facet is straight in the large. A brush is never straight in the
    # small, and it never holds one width for the length of a facet. So
    # the contour gets the three things that separate 勾 from a pen:
    #
    #  * **一波三折** — two incommensurable sines across the run, at an
    #    amplitude of a line width or two, then the same two smoothing
    #    taps the 皴 get. The facet stays where it was; the vertex turns
    #    from a polygon corner into a brush turn.
    #  * **提按** — the hand bears down in the middle of a pull and lifts
    #    at both ends. `drag` ramps width monotonically from one end to
    #    the other, so a pull of it is a wedge and three of them in a row
    #    are three wedges; only an explicit per-node profile can be thin,
    #    fat, thin. (The same reason the tarsus had to leave `drag`.)
    #  * **飞白** — the rock's 皴 are dry and its outline was wet, which is
    #    two different brushes on one stone.
    def wave(line, amp, lam):
        ph = rng.uniform(0, TAU)
        m_ = max(1, len(line) - 1)
        out = [(x + amp * (math.sin(i / m_ * lam + ph)
                           + 0.5 * math.sin(i / m_ * lam * 2.3 - ph * 1.7)), y)
               for i, (x, y) in enumerate(line)]
        for _ in range(2):
            out = ([out[0]]
                   + [((out[q - 1][0] + 2.0 * out[q][0] + out[q + 1][0]) * 0.25,
                       out[q][1]) for q in range(1, len(out) - 1)]
                   + [out[-1]])
        return out

    #
    # How hard the brush lifts, and how wide the gaps are, turn out to be
    # one dial and not two. At a full 提按 — thin at both ends of every
    # pull — plus the gaps the old `drag` version used, the lit flank came
    # out as three fat detached **commas** hanging in bare paper: every
    # pull now tapers away before the next begins, so nothing joins and
    # there is no edge, only marks. A broken brush line still has to read
    # as *one line* broken, which means the lift is modest and the gaps
    # are small; 断 is a breath, not a space.
    def pull(line, w):
        line = wave(line, w * rng.uniform(0.5, 1.0), rng.uniform(5.0, 11.0))
        m = len(line)
        j = 0
        while j < m - 4:
            seg = min(m - j, rng.randint(max(7, m // 3), max(10, m // 2)))
            sub = line[j:j + seg]
            q = len(sub) - 1
            hwl = [w * (0.62 + 0.70 * math.sin(math.pi * (i / q)) ** 0.55)
                   * rng.uniform(0.92, 1.08) for i in range(q + 1)]
            dens = [clamp01(0.40 + 0.54 * math.sin(math.pi * (i / q)) ** 0.45)
                    for i in range(q + 1)]
            sweep(cv, sub, hwl, dens, gl, gh,
                  alpha=rng.uniform(0.44, 0.74), core=0.24,
                  dry=0.26 + 0.44 * dry, dry_k=rng.uniform(1.6, 3.2),
                  dry_ph=rng.uniform(0, TAU))
            j += seg + rng.randint(1, max(2, m // 16))

    left = [(edge[i][0] + hws[i] * 0.05, pts[i][1]) for i in range(n + 1)]
    right = [(edge[i][1] - hws[i] * 0.05, pts[i][1]) for i in range(n + 1)]
    pull(left if lx > 0.0 else right, lw * 1.50)    # the shaded flank, heavier
    pull(right if lx > 0.0 else left, lw * 1.05)
    m = 26
    crestline = []
    sxl, sxr = edge[n]
    for i in range(m + 1):
        u = 0.03 + 0.94 * i / m
        crestline.append((sxl + (sxr - sxl) * u, top_y(u) + lw * 0.9))
    pull(crestline, lw * 1.15)

    # ---- 皴 — a few long ones *along* the facets, not a scatter of short
    # ---- ones. A rock's texture runs with its planes or it is hatching.
    for _ in range(int(5 + 4 * old)):
        t0 = rng.uniform(0.04, 0.50)
        t1 = min(0.84, t0 + rng.uniform(0.24, 0.46))
        u0 = (rng.betavariate(1.5, 2.6) if lx > 0.0
              else 1.0 - rng.betavariate(1.5, 2.6))
        # 皴随石势 — they run *with* the facet, all of them, so they read as
        # the rock's own planes. Letting them fan was what made the first
        # pass look like a scatter of worms.
        # Held to ±0.07 over a stroke a third of the rock tall, six of
        # them come out a *comb*: parallel, full length, and at 0.34–0.60
        # of the darkest ink they are the loudest thing on a mass that is
        # supposed to be a wash. 皴随石势 means they lie along the plane,
        # not that they are copies of one another.
        u1 = clamp01(u0 + rng.uniform(-0.15, 0.15))
        line = []
        for q in range(11):
            f = q / 10.0
            tt = t0 + (t1 - t0) * f
            i = min(n, int(tt * n))
            xl, xr = edge[i]
            uu = u0 + (u1 - u0) * f
            line.append((xl + (xr - xl) * clamp01(uu + rng.gauss(0, 0.006)),
                         pts[i][1]))
        # 一波三折, not lightning. Independent jitter on every sample of a
        # short line is a zigzag at the sample spacing; two smoothing taps
        # turn the same noise into the waver a dragged brush actually has.
        for _ in range(2):
            line = [line[0]] + [((line[q - 1][0] + 2.0 * line[q][0]
                                  + line[q + 1][0]) * 0.25,
                                 line[q][1])
                                for q in range(1, len(line) - 1)] + [line[-1]]
        if inked(u0, (t0 + t1) * 0.5) < 0.18:
            continue
        # 皴有断 — in two pulls with a gap, like every other line on the
        # rock. One unbroken drag the height of a facet is a scored
        # groove; the same ink lifted once in the middle is a texture.
        w_a, w_b = lw * rng.uniform(0.75, 1.15), lw * rng.uniform(0.40, 0.80)
        al = rng.uniform(0.22, 0.44)
        if rng.random() < 0.62 and len(line) > 7:
            c = rng.randint(3, 6)
            gp = rng.randint(1, 2)
            parts = [line[:c], line[c + gp:]]
        else:
            parts = [line]
        for q, seg in enumerate(parts):
            if len(seg) < 2:
                continue
            drag(cv, seg, w_a if q == 0 else w_b, w_b,
                 gl, gh, alpha=al * (1.0 if q == 0 else 0.78),
                 d0=0.34, d1=0.12,
                 taper=0.9, dry=0.40 + 0.40 * dry, dry_k=rng.uniform(2.0, 4.0),
                 dry_ph=rng.uniform(0, TAU))

    # ---- 淡皴 — and the lit face is not bare paper.
    #
    # The loop above draws its 皴 from `betavariate(1.5, 2.6)`, whose mean
    # is 0.37, so every stroke lands on the loaded flank or just past it
    # and the long lit side of the rock gets *nothing*: a pale slab with a
    # contour round it, which is the one mark on a 孤禽 sheet that reads as
    # plastic. 石分三面 does not mean one face is drawn and two are washed.
    # The lit face takes 皴 too — it simply takes it **淡**: half the ink,
    # twice the dryness, shorter, and broken into two or three pulls with
    # gaps, so what lands is the memory of a texture rather than a texture.
    # 墨不到处笔不到 still holds, which is why the strokes stop well short
    # of the contour rather than running into it: an edge is a line or it
    # is a fade, and a rock that has both at the same place has neither.
    for _ in range(int(4 + 3 * old)):
        t0 = rng.uniform(0.06, 0.58)
        t1 = min(0.88, t0 + rng.uniform(0.16, 0.34))
        b = rng.betavariate(2.2, 1.6) * 0.46 + 0.50      # 0.50 .. 0.96
        u0 = b if lx > 0.0 else 1.0 - b
        u1 = clamp01(u0 + rng.uniform(-0.05, 0.05))
        line = []
        for q in range(9):
            f = q / 8.0
            tt = t0 + (t1 - t0) * f
            i = min(n, int(tt * n))
            xl, xr = edge[i]
            uu = clamp01(u0 + (u1 - u0) * f + rng.gauss(0, 0.005))
            # inside the contour by a margin, never on it
            uu = min(0.93, max(0.07, uu))
            line.append((xl + (xr - xl) * uu, pts[i][1]))
        for _ in range(2):
            line = [line[0]] + [((line[q - 1][0] + 2.0 * line[q][0]
                                  + line[q + 1][0]) * 0.25,
                                 line[q][1])
                                for q in range(1, len(line) - 1)] + [line[-1]]
        # ...and *this* is where 墨不到处笔不到 has to be asked rather than
        # assumed. `uu` inside [0.07, 0.93] keeps the stroke inside the
        # **contour**; on the lit face the contour and the ink are nowhere
        # near each other, and the strokes that fell in the gap were the
        # pale scratches under the stump's shoulder.
        tm = (t0 + t1) * 0.5
        if (inked(u0, t0) + inked(u1, t1) + inked((u0 + u1) * 0.5, tm)) < 0.36:
            continue
        drag(cv, line, lw * rng.uniform(0.60, 1.00), lw * rng.uniform(0.35, 0.65),
             dilute(gl, 0.62), dilute(gh, 0.74),
             alpha=rng.uniform(0.16, 0.34), d0=0.26, d1=0.08,
             taper=0.9, dry=0.62 + 0.30 * dry, dry_k=rng.uniform(3.0, 5.5),
             dry_ph=rng.uniform(0, TAU))

    # 擦 — a dozen flecks of the same dry brush across the lit face, which
    # is what keeps it from reading as three long scratches on blank paper.
    for _ in range(int(9 + 7 * old)):
        tt = rng.uniform(0.05, 0.86)
        i = min(n, int(tt * n))
        xl, xr = edge[i]
        b = rng.betavariate(2.0, 1.5) * 0.52 + 0.42
        uu = min(0.92, max(0.08, b if lx > 0.0 else 1.0 - b))
        px0 = xl + (xr - xl) * uu
        py0 = pts[i][1]
        if py0 < top_y(uu) + lw * 2.0 or inked(uu, tt) < 0.14:
            continue
        L = span * rng.uniform(0.020, 0.058)
        aa = math.pi * 0.5 + rng.uniform(-0.42, 0.42)
        drag(cv, [(px0, py0),
                  (px0 + math.cos(aa) * L, py0 + math.sin(aa) * L)],
             lw * rng.uniform(0.50, 0.95), lw * 0.35,
             dilute(gl, 0.50), dilute(gh, 0.66),
             alpha=rng.uniform(0.12, 0.26), d0=0.22, d1=0.06,
             dry=0.70, dry_k=rng.uniform(3.5, 6.0), dry_ph=rng.uniform(0, TAU))

    # ---- 苔 — last, and only on the crest.
    #
    # `moss` was being called on this mass from the composer, with the
    # stump's own half-widths — eighty pixels of them. It offsets each
    # cluster by up to 0.7 of that from the axis and scatters the dots by
    # half of it again, which on a branch is a sprinkle along the bark and
    # on a rock a hundred pixels wide is a handful of grit thrown at the
    # paper *above* the break: seven black specks floating round the bird's
    # feet with nothing under them. 点苔 goes where the rock turns, so it
    # is placed here, against the same `top_y` the crest line was drawn
    # from and the same `inked` grid everything else is gated by.
    for _ in range(int(2 + 4 * old)):
        uu = rng.uniform(0.10, 0.90)
        if inked(uu, 0.93) < 0.10:
            continue
        bx = sxl + (sxr - sxl) * uu
        by = top_y(uu) + lw * rng.uniform(1.0, 4.0)
        for _ in range(rng.randint(2, 5)):
            disc(cv, bx + rng.gauss(0, span * 0.012),
                 by + rng.gauss(0, span * 0.013),
                 rng.uniform(0.7, 1.8), rgb(st["wood"][1]),
                 rng.uniform(0.34, 0.74) * clamp01(0.5 + 0.5 * k), soft=0.4)

    seat_u = 0.5 + rng.uniform(0.08, 0.30) * rng.choice((1.0, -1.0))
    seat = (sxl + (sxr - sxl) * seat_u, top_y(seat_u) + lw * 2.2)
    return pts, hws, seat


def _leaf(cv, rng, x, y, ang, L, st, air, k, alpha=0.88):
    """点叶 — one leaf is one press and one rib. Nothing is outlined."""
    lo = dilute(rgb(st["leaf"][0]), clamp01(0.74 * k))
    hi = dilute(rgb(st["leaf"][1]), clamp01(k))
    W = L * rng.uniform(0.40, 0.56)
    fl = rng.choice((1.0, -1.0))
    press(cv, x + math.cos(ang) * L * 0.46, y + math.sin(ang) * L * 0.46,
          L, W, ang, lo, hi, load=0.80, grad=0.50, flank=fl, rim=0.12,
          tip=0.64, tail=0.46, splay=0.18, bulge=0.12,
          streak=0.12 + 0.34 * air["dry"], bare=0.10 + 0.20 * air["dry"],
          ragged=0.06, alpha=alpha, phase=rng.uniform(0, TAU))
    if L > 11.0:
        drag(cv, [(x, y), (x + math.cos(ang) * L * 0.82,
                           y + math.sin(ang) * L * 0.82)],
             max(0.45, W * 0.10), 0.3, dilute(hi, 0.42), hi,
             alpha=0.34, d0=0.70, d1=0.25)


def _spray(cv, rng, pts, hws, st, air, k, S, dens):
    """Leaves along a twig — alternate sides, drooping, and sparse.

    `dens` folds the scene's and the air's leaf dials together, which is
    how 雪霁 gets bare wood and 春晴 gets a spray without either being a
    different code path.
    """
    n = len(pts) - 1
    if n < 4 or dens < 0.02:
        return
    step = max(3, int(n / max(1.0, 3.0 + 7.0 * dens)))
    sgn = rng.choice((1.0, -1.0))
    for i in range(max(2, n // 5), n, step):
        if rng.random() > min(1.0, dens):
            continue
        x, y = pts[i]
        j = min(n, i + 1)
        ta = math.atan2(pts[j][1] - pts[i - 1][1], pts[j][0] - pts[i - 1][0])
        a = ta + sgn * rng.uniform(0.55, 1.05)
        # 叶随枝垂 — and then pulled a third of the way toward gravity, so a
        # sheet of leaves agrees with itself instead of bristling.
        a += _turn(a, math.pi * 0.5) * rng.uniform(0.22, 0.44)
        L = S * rng.uniform(0.026, 0.044) * (0.70 + 0.45 * dens)
        px = x + math.cos(a) * hws[i] * 0.8
        py = y + math.sin(a) * hws[i] * 0.8
        _leaf(cv, rng, px, py, a, L, st, air, k,
              alpha=rng.uniform(0.70, 0.94))
        if rng.random() < 0.42:                   # 二叶一组
            a2 = a + sgn * rng.uniform(-0.70, -0.30)
            _leaf(cv, rng, px, py, a2, L * rng.uniform(0.70, 0.92),
                  st, air, k, alpha=rng.uniform(0.58, 0.84))
        sgn = -sgn


def _perch(rng, paths, sc, W, H, S, want_hw, avoid=(), pad=0.0, foot=None):
    """Which node of this tangle can be stood on.

    Scored, not searched for: near-horizontal, inside the sheet, thick
    enough to hold and inside the scene's band. Returning the best of a
    few hundred candidates is both simpler than constraining the wood to
    produce one and better, because the wood stays free to be wood.
    """
    e = sc["entry"]
    lo, hi = sc["perch"]
    if e == "foot":
        # 八大's stump: the bird stands on the break, not on a limb, and
        # `_stump` already chose where on it — off centre, because a bird
        # planted in the middle of its rock is a weathervane.
        pts, hws = paths[0]
        if foot is None:
            foot = (pts[-1][0], pts[-1][1])
        # `_stand` lifts the ankle off the *axis* by `hw`, which is correct
        # for a branch — the axis is its centreline — and badly wrong for a
        # rock, because `_stump` hands back a point on the break itself. At
        # a half-width of a hundred pixels the bird floated a hand above its
        # own stump. So the width reported here is the **grip**, not the
        # mass, and the seat is pushed back down by exactly it.
        hw = max(2.0, want_hw)
        return foot[0], foot[1] + hw, 0.0, hw
    best = None
    for relax in (0, 1, 2):
        for pts, hws in paths:
            n = len(pts) - 1
            for i in range(2, n - 1):
                x, y = pts[i]
                # 立足要实 — two filters, both learned the same way. A twig
                # whose half-width has tapered under a pixel is *drawn*, but
                # it is not visible, so a bird seated on it stands on blank
                # paper with its claws closed round nothing. And a seat near
                # the edge is a seat with the bird's own body hanging off the
                # sheet, because the body is a sheet-width's tenth long and
                # extends backwards from the feet, not upwards.
                if hws[i] < (max(2.2, want_hw * 0.30), 1.6, 1.0)[relax]:
                    continue
                # Room for the bird, not for the foot: it extends a body
                # backwards from the claws and leg + body + head upwards.
                # Measured against `Lb`, that is about one pad sideways and
                # one and a half up -- the 1.9 this used to ask for was a
                # guess, and on a sheet whose only level wood sits high
                # (俯枝's bough comes down out of a top corner) it rejected
                # the one seat there was by nine pixels.
                sq = (1.0, 1.0, 0.55)[relax]
                if not (max(W * 0.14, pad) * sq < x < W - max(W * 0.14, pad) * sq
                        and max(H * 0.10, pad * 1.50) * sq < y < H * 0.92):
                    continue
                if not relax and not (H * (lo - 0.06) < y < H * (hi + 0.08)):
                    continue
                if any(math.hypot(x - vx, y - vy) < vr for vx, vy, vr in avoid):
                    continue
                ta = math.atan2(pts[i + 1][1] - pts[i - 1][1],
                                pts[i + 1][0] - pts[i - 1][0])
                level = abs(math.sin(ta))         # 0 horizontal, 1 vertical
                # 爪要抓枝, and nothing grips a vertical stick. Scoring this
                # softly was not enough: on a sheet whose only wood runs
                # steeply — 俯枝's bough comes down out of the top corner —
                # the best of a bad set still wins, and the bird comes out
                # skewered on an upright twig with the tip showing below its
                # own foot. A floor, and a looser one on the second pass.
                if level > (0.45, 0.72, 1.01)[relax]:
                    continue
                thick = min(1.0, hws[i] / want_hw)
                band = 1.0 - abs(y / H - (lo + hi) * 0.5) / max(0.08, hi - lo)
                s = (1.0 - level) * 2.2 + thick * 1.4 + band * 0.9 \
                    + rng.uniform(0.0, 0.5)
                if best is None or s > best[0]:
                    best = (s, x, y, ta, max(1.5, hws[i]))
        if best is not None:
            break
    if best is None:
        # 无枝可栖 — and a 翎毛 sheet without the 翎毛 on it is not a painting
        # of anything. Three passes that all come back empty mean the wood of
        # this particular seed runs steeply and thinly everywhere, so take the
        # thickest node that is merely *on the paper* and perch there.
        for pts, hws in paths:
            for i in range(2, len(pts) - 2):
                x, y = pts[i]
                if not (0 < x < W and H * 0.08 < y < H * 0.88):
                    continue
                if best is None or hws[i] > best[0]:
                    ta = math.atan2(pts[i + 1][1] - pts[i - 1][1],
                                    pts[i + 1][0] - pts[i - 1][0])
                    best = (hws[i], x, y, ta, max(1.5, hws[i]))
        if best is None:
            return None
    x, y, ta, hw = best[1], best[2], best[3], best[4]
    # A bird on a steep stem grips it *across*, not along it -- the claws close
    # round the stem crosswise and the body still stands upright. So the seat's
    # angle is flattened rather than obeyed: obeying it tips the whole creature
    # on its side and the painting reads as a bird falling off a twig.
    if abs(math.sin(ta)) > 0.50:
        flat = math.copysign(0.46, math.sin(ta))      # keep which way it dips
        ta = flat if math.cos(ta) >= 0.0 else math.pi - flat
    return x, y, ta, hw


def _stand(sp, Lb, px, py, ta, hw, **kw):
    """Put the bird where its feet land on the wood.

    Backwards, which is the only way it works: the perch is given, the
    ankle has to be on the wood's surface, and the body goes wherever
    that puts it. Placing the body first and then reaching for the branch
    is what produces a bird standing an inch above its own perch — and
    it is invisible at full size and glaring in a 册页.
    """
    nx, ny = math.sin(ta), -math.cos(ta)
    if ny > 0.0:
        nx, ny = -nx, -ny                         # the upper surface
    probe = Pose(sp, 0.0, 0.0, Lb, **kw)
    rx, ry = probe.L2S(*probe.leg_l)
    ax, ay = px + nx * hw, py + ny * hw           # the ankle, on the bark
    cx = ax + nx * probe.leg_len - rx
    cy = ay + ny * probe.leg_len - ry
    return Pose(sp, cx, cy, Lb, **kw), (ax, ay, ta, hw)


def _survived(pts, want, W, H, level=True):
    """Keep the best of a few gestures — 势要贯, with a perch in it.

    Deliberately *not* 墨梅's `_reach`, which rewards a diagonal and
    penalises wood lying flat across the sheet (忌横). That is right for a
    plum branch and wrong here: a bird needs somewhere level to stand, and
    a bough that never once runs level gives `_perch` nothing to choose
    between. So the second term is the length of the longest near-level
    run, capped — enough wood to hold a sparrow and no reward for more.
    """
    if len(pts) < 4:
        return -1e9
    x0, y0 = pts[0]
    far = max(math.hypot(x - x0, y - y0) for x, y in pts) / min(W, H)
    s = len(pts) / max(1, want) + 0.30 * far
    if not level:
        return s
    run = best = 0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        if abs(dy) < abs(dx) * 0.60:
            run += 1
            best = max(best, run)
        else:
            run = 0
    # Capped hard and capped low, which is the whole of the lesson. Scored
    # generously it selects, every time, for the one candidate that is level
    # from edge to edge — and a bough lying flat across the paper is 忌横,
    # the fault 墨梅's `_reach` exists to avoid. Eight steps of level wood is
    # a perch; more than that is a plank, and gets no further credit.
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = min(max(xs) - min(xs), max(ys) - min(ys)) / min(W, H)
    return s + 0.40 * min(1.0, best / 8.0) + 0.70 * min(span, 0.20)


# --------------------------------------------------------------------------
# 题款 — the writing.
#
# Same machinery as the other nine, and the same one rule that took 墨梅 a
# round to learn: a title that is a *claim about the weather* is held back and
# handed out by the weather, because the inscription is inside the painting
# and a reader sees the caption and the picture at once. 雪禽 on a sheet with
# no snow is not a liberty, it is a contradiction. The scene gets the same
# treatment — 双栖 and 相呼 belong to a sheet with two birds on it.
# --------------------------------------------------------------------------

TITLE_A = ["幽禽", "栖禽", "小鸟", "枝头", "清音", "鸟语", "闲趣", "一枝栖",
           "高枝", "禽鸣", "翎毛", "山禽"]
TITLE_AIR = {
    "春晴": ["春禽", "东风", "春晓", "柳莺"],
    "清秋": ["秋声", "霜枝", "秋禽", "晚晴"],
    "疏雨": ["听雨", "雨馀", "疏雨", "烟禽"],
    "雪霁": ["雪禽", "寒雀", "雪晴", "冻雀"],
}
TITLE_SCENE = {
    "栖枝": ["栖枝", "一枝栖", "高枝"],
    "双栖": ["双栖", "相呼", "比翼", "二禽"],
    "孤禽": ["孤禽", "独立", "寂寥", "冷眼"],
    "俯枝": ["俯枝", "临风", "枝头"],
}
TITLE_B = ["图", "小景", "小品", "写意", "册页", "图卷", "清趣"]
ACTION = ["写生", "戏作", "写意", "拟古", "画禽", "南窗戏墨", "并记",
          "拟八大笔意", "灯下作", "雪窗消寒"]
SEAL_WORDS = ["清赏", "墨缘", "写心", "一枝", "幽禽", "闲趣", "天真", "独立",
              "鸟语", "栖迟", "寄情", "老圃", "安之"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into bird_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [w for v in TITLE_AIR.values() for w in v]
             + [w for v in TITLE_SCENE.values() for w in v]
             + [v["cn"] for v in SCENES.values()]
             + [v["cn"] for v in AIR.values()]
             + [v["cn"] for v in STYLES.values()]
             + [v["cn"] for v in BIRDS.values()]
             + list(SCENES) + list(AIR) + list(STYLES) + list(BIRDS)
             + [TIANGAN, DIZHI, "日月年写于图春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a blob baked for this vocabulary; fall back to the siblings'
    (the family shares most characters), then to no 题款."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("bird_font", "mei_font", "crab_font", "grape_font",
                "orchid_font", "bamboo_font", "gongbi_font", "shanshui_font"):
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


def inscribe(cv, rng, occ, S, title, focus=None):
    """Measure the block, then let the 留白 decide where it goes."""
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
        bx, by = occ.place(bw, bh, margin, corner_bias=1.5, focus=focus)
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

def compose(cv, rng, scene_key, air_key, style_key, species=None, mood=None,
            stat=None):
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

    # ---- the photograph of the paper. Everything after this line can be
    # ---- lifted back out of the sheet; nothing before it can.
    res = Reserve(cv)

    # ---- 枝 — the perch -------------------------------------------------
    old = sc["old"]
    level = sc["entry"] != "foot"
    stump_seat = None
    if sc["entry"] == "foot":
        # 孤禽 does not grow a branch, it stands on a rock. Everything below
        # this block -- twigs, leaves, the bird -- reads the same (pts, hws).
        x0 = W * (rng.uniform(0.30, 0.44) if side > 0
                  else rng.uniform(0.56, 0.70))
        mpts, mhws, stump_seat = _stump(cv, rng, res, st, air, k, x0, W, H, S,
                                        old=old)
        mjoints, grow = (), -math.pi * 0.5
        w0, w1 = mhws[0], mhws[-1]
        occ.add_path(mpts, w0 * 1.15, w1 * 1.15)
        # Neither `blot` nor `flyback` here. Both are sized off `hws`, and
        # at a hundred pixels of half-width they stop being 斑驳 and 飞白
        # and print as soft pale clouds floating over the rock. The stump
        # makes its own 留白, inside the wash, where the dry brush skipped.
        load = 0.0
    else:
        best = None
        for _ in range(5):
            x, y, a0, L, grow, pull = _entry(rng, sc, W, H, S, side)
            want = [0]
            pts, joints = wood_path(rng, x, y, a0, L,
                                    segs=rng.randint(*sc["segs"]), kink=0.34,
                                    bow=0.16, grow=grow, pull=pull, cone=1.00,
                                    bounds=(W, H), want=want)
            s = _survived(pts, want[0], W, H, level=level)
            if best is None or s > best[0]:
                best = (s, pts, joints, grow)
        _, mpts, mjoints, grow = best
        w0 = S * rng.uniform(*sc["w0"])
        w1 = max(1.0, w0 * rng.uniform(0.30, 0.48))
        load = rng.choice((1.0, -1.0)) * rng.uniform(0.55, 0.95)
        mhws = bough(cv, rng, mpts, w0, w1, st, air, alpha=0.96, old=old,
                     bone=0.45 + 0.45 * old, k=k, side=load)
        occ.add_path(mpts, w0 * 1.3, w1 * 1.3)
        # 斑驳 first, bark second: `blot` lifts paper and would wipe out any
        # line already drawn inside the mass. (墨梅's order, and its reason.)
        blot(cv, rng, res, mpts, mhws, air, k=k, old=old, side=load,
             n=int(3 + 5 * old))
        flyback(cv, rng, res, mpts, mhws, air, k=k, side=load,
                n=int((2 + 4 * old + 3 * air["dry"]) * max(1.0, S / 820.0)))
        cun(cv, rng, mpts, mhws, air,
            n=int((20 + 40 * old) * max(1.0, S / 820.0)), k=k, old=old)
        if old > 0.4:
            moss(cv, rng, mpts, mhws, k=k, n=int(2 + 4 * old))

    paths = [(mpts, mhws)]
    twigs = []
    nt = rng.randint(*sc["twigs"])
    n_m = len(mpts) - 1
    sites = [i for i in list(mjoints)
             + [int(n_m * f) for f in (0.30, 0.48, 0.66, 0.84)]
             if 3 <= i <= n_m - 3]
    rng.shuffle(sites)
    sgn = rng.choice((1.0, -1.0))
    for i in sites[:nt]:
        tx, ty = mpts[i]
        ta = math.atan2(mpts[i + 1][1] - mpts[i - 1][1],
                        mpts[i + 1][0] - mpts[i - 1][0])
        a0 = ta + sgn * rng.uniform(0.42, 0.92)
        sgn = -sgn
        wp = w0 + (w1 - w0) * (i / n_m)
        tw0 = max(1.2, wp * rng.uniform(0.48, 0.74))
        want = [0]
        tpts, _tj = wood_path(rng, tx, ty, a0, S * rng.uniform(0.15, 0.32),
                              segs=rng.randint(2, 3), kink=0.38, bow=0.20,
                              grow=grow, pull=0.35, cone=1.05,
                              bounds=(W, H), want=want)
        if len(tpts) < 6:
            continue
        tw1 = max(0.7, tw0 * rng.uniform(0.24, 0.42))
        thws = bough(cv, rng, tpts, tw0, tw1, st, air, alpha=0.93,
                     old=old * 0.4, k=k, side=load)
        occ.add_path(tpts, tw0 * 1.3, tw1 * 1.3)
        cun(cv, rng, tpts, thws, air, n=int(6 + 10 * old), k=k, old=old * 0.5)
        paths.append((tpts, thws))
        twigs.append((tpts, thws))

    # ---- 叶 — enough to say what season it is, and no more --------------
    dens = sc["leaves"] * air["leaves"]
    for tpts, thws in twigs:
        _spray(cv, rng, tpts, thws, st, air, k, S, dens)
    if dens > 0.30:
        _spray(cv, rng, mpts, mhws, st, air, k, S, dens * 0.45)

    # ---- 禽 --------------------------------------------------------------
    if species is None:
        species = ("八哥" if (scene_key == "孤禽" and rng.random() < 0.60)
                   else sorted(BIRDS)[rng.randrange(len(BIRDS))])
    sp = BIRDS[species]
    Lb0 = S * rng.uniform(*sp["body"])

    seats, avoid = [], []
    for _ in range(sc["birds"]):
        # 1.2 bodies of margin, not one: the bird extends backwards from its
        # claws by half a body *and then a tail*, which on a 白头翁 is another
        # 0.88 of one. A pad of 0.95 put the tail over the edge of the sheet.
        seat = _perch(rng, paths, sc, W, H, S, Lb0 * 0.10, avoid=avoid,
                      pad=Lb0 * 1.20, foot=stump_seat)
        if seat is None:
            break
        seats.append(seat)
        # 双栖 is two birds, and two bodies a body-and-a-third apart overlap:
        # the near one eats the far one and the pair reads as one lumpy mass.
        avoid.append((seat[0], seat[1], Lb0 * 1.85))
    if not seats:
        # The subject is the bird. If every branch this seed grew fell outside
        # the sheet there is still a painting to make -- 空枝 with a bird in
        # the air over it beats a 翎毛 sheet with no 翎毛 on it -- so put the
        # feet where the 章法 wanted them and let the claws close on nothing.
        lo_, hi_ = sc["perch"]
        seats.append((W * rng.uniform(0.38, 0.62),
                      H * rng.uniform(lo_, hi_), 0.0, Lb0 * 0.055))

    # 呼应 — who looks at whom. With two birds the answer is each other and
    # it is not negotiable: a pair facing the same way is two copies of one
    # bird, which is the 册页 fault 虾蟹 §5 spent its length on. With one,
    # the bill points into whichever half of the sheet has more paper in
    # it, because that is what the emptiness is *for*.
    if len(seats) == 2:
        faces = [1 if seats[0][0] > seats[1][0] else -1,
                 1 if seats[1][0] > seats[0][0] else -1]
    elif seats:
        faces = [1 if seats[0][0] > W * 0.5 else -1]
        if rng.random() < 0.16:
            faces[0] = -faces[0]
    else:
        faces = []

    # 态 — one attitude per bird, and with two birds never the same one.
    # 呼应 is a relationship and two birds doing the same thing have none:
    # the pair reads as one bird printed twice however carefully the second
    # one was turned to face the first.
    names = sorted(MOODS)
    wts = [MOODS[nm]["w"] for nm in names]
    moods = []
    for _ in seats:
        if mood is not None and not moods:
            moods.append(mood)       # --mood pins the *first* bird only, so
            continue                 # 双栖's 呼应 still gets two attitudes
        pool = [(nm, w) for nm, w in zip(names, wts) if nm not in moods] \
            or list(zip(names, wts))
        r = rng.uniform(0.0, sum(w for _, w in pool))
        for nm, w in pool:
            r -= w
            if r <= 0.0:
                break
        moods.append(nm)

    poses = []
    for b, (px, py, ta, hw) in enumerate(seats):
        Lb = Lb0 * (1.0 if b == 0 else rng.uniform(0.84, 0.96))
        md = MOODS[moods[b]]
        po, pr = _stand(
            sp, Lb, px, py, ta, hw, face=faces[b],
            lean=rng.uniform(*sc["lean"]) + air["lean"]
                 + rng.uniform(*md["lean"]),
            hunch=clamp01(rng.uniform(*sc["hunch"]) + air["hunch"]
                          + rng.uniform(*md["hunch"])),
            # 嘴长者不深俯 — the nod is scaled by the bill, because the
            # bill is what the nod points. Half a radian of 俯啄 on a
            # sparrow is a bird looking at the branch; the same half radian
            # on a kingfisher, whose bill is longer than its own skull,
            # swings a black dagger down past its own feet. One factor,
            # read off the table that already says how long the bill is.
            turn=rng.uniform(*md["turn"]),
            dip=rng.uniform(*md["dip"]) * clamp01(1.30 - 0.70 * sp["bill"]),
            fluff=clamp01(air["fluff"] * md["fluff"]
                          * rng.uniform(0.86, 1.10)),
            tail_drop=rng.uniform(*md["tail"]))
        glare = sc["glare"]
        if glare < 0.05 and species == "八哥" and rng.random() < 0.30:
            glare = rng.uniform(0.60, 0.90)
        bird(cv, rng, res, po, st, air, k, perch=pr, glare=glare)
        occ.add(po.cx, po.cy, po.reach * 0.85)
        poses.append(po)

    # ---- 雪 --------------------------------------------------------------
    if air.get("snow"):
        keep = [(p.cx, p.cy, p.reach * 0.80) for p in poses]
        snow_on(cv, rng, res, mpts, mhws, -1.0, keep, k=k)
        for tpts, thws in twigs:
            if max(thws) < 1.8:
                continue
            snow_on(cv, rng, res, tpts, thws, -1.0, keep, k=k)
        snowfall(cv, rng, S, int(S * 0.38))

    if stat is not None:
        stat.update(birds=len(poses), species=species, twigs=len(twigs),
                    side=side, seats=seats, moods=moods, Lb=Lb0,
                    poses=[(p.cx, p.cy, p.Lb) for p in poses])

    r = rng.random()
    head = (air["cn"] if r < 0.10 else
            TITLE_SCENE[scene_key][rng.randrange(len(TITLE_SCENE[scene_key]))]
            if r < 0.42 else
            TITLE_AIR[air_key][rng.randrange(len(TITLE_AIR[air_key]))]
            if r < 0.70 else TITLE_A[rng.randrange(len(TITLE_A))])
    title = head + TITLE_B[rng.randrange(len(TITLE_B))]
    # The 款 is weighed against the *bird*, not against the branch: a bough
    # crosses the whole sheet and has no centre worth balancing, and it is
    # the creature the eye goes to.
    focus = (sum(p.cx for p in poses) / len(poses),
             sum(p.cy for p in poses) / len(poses)) if poses else None
    inscribe(cv, rng, occ, S, title, focus=focus)
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
          style=None, species=None, mood=None, quiet=False, stat=None):
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
    if species is not None and species not in BIRDS:
        raise SystemExit("unknown species %r; try --list" % species)
    if mood is not None and mood not in MOODS:
        raise SystemExit("unknown mood %r; try --list" % mood)

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    stat = {} if stat is None else stat
    title, ground = compose(cv, rng, sk, ak, yk, species=species, mood=mood,
                            stat=stat)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-24s %s  %s·%s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk,
                 stat.get("species", "?"), AIR[ak]["cn"], STYLES[yk]["cn"],
                 ground, seed, size[0], size[1], time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese ink-bird paintings (翎毛).")
    ap.add_argument("-o", "--out", default="bird.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="栖枝|双栖|孤禽|俯枝")
    ap.add_argument("--species", help="麻雀|八哥|白头翁|翠鸟")
    ap.add_argument("--mood", help="平栖|回首|仰首|俯啄|缩颈|侧顾")
    ap.add_argument("--air", help="春晴|清秋|疏雨|雪霁")
    ap.add_argument("--style", help="水墨|淡彩|赭墨|花青")
    ap.add_argument("--size", default="900x1300", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for key, v in SCENES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("禽 (species):")
        for key, v in BIRDS.items():
            print("  %-6s %s" % (key, v["en"]))
        print("态 (attitude):")
        for key, v in MOODS.items():
            print("  %-6s %s" % (key, v["en"]))
        print("风致 (air):")
        for key, v in AIR.items():
            print("  %-6s %-5s %s" % (key, v["cn"], v["en"]))
        print("设色 (colour):")
        for key, v in STYLES.items():
            print("  %-6s %s" % (key, v["en"]))
        print("画诀: 一笔点头, 两笔成背 — 头如卵, 背如弓, 嘴如钳, 眼如豆, "
              "腹宜虚, 翅分三停, 尾如剪, 爪要抓枝, 眼有神")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("翎毛 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style, a.species,
              a.mood)
    return 0


if __name__ == "__main__":
    sys.exit(main())
