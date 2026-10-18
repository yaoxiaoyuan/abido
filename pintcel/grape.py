#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grape.py — 随机生成中国水墨葡萄图 / Random Chinese ink-grape paintings (墨葡萄).

葡萄 came late to Chinese painting and arrived as a 写意 subject only —
there is no 工笔 grape tradition worth the name, because the whole appeal of
the thing is that a berry is *one touch* of a loaded brush and either it is
round or it is a blot. 徐渭 set the terms in the 墨葡萄图 and his own
inscription says what the picture is for: 「笔底明珠无处卖，闲抛闲掷野藤中」
— pearls under the brush, with nobody to sell them to, tossed away into the
wild vine. So the subject is two things at once, a still life of fruit and a
tangle of line, and a sheet that gets the fruit right and the line wrong is
not a 葡萄图 at all.

This generator is built as the 画诀 for it, the way bamboo.py is built as
叶诀 and orchid.py as 一笔三折:

    果如明珠   a berry is a pearl — round, but no two round the same way,
               and never twice the same size.
    忌算盘珠   the great prohibition: a bunch is not a row of abacus beads.
               No lattice, no shared radius, no visible rank and file.
    留白为光   the highlight is not white paint. It is paper the brush was
               steered around, so it carries the grain of the sheet.
    前实后虚   the near berries are dark and crisp, the far ones pale and
               soft-edged and allowed to merge. A bunch has a front.
    一串成形   下尖上宽 — broad at the shoulder, tapering to one berry at
               the point. A bunch is a cone, never a ball.
    藤如草书   the vine is written, not drawn: 中锋 wandering, curvature
               that changes sign, thick and thin in the same stroke, and
               dry where the brush ran out.
    卷须点睛   the tendrils are what make it alive. They are the last marks
               on the sheet and the only ones that spiral.
    墨叶衬珠   dark leaves set the fruit off. The leaf goes down first and
               the pearls sit on it.
    叶如掌     five lobes, a deep basal cleft where the stalk comes in, a
               serrated edge — and 忌对称, never twice the same leaf.

The structural idea carried over from gongbi.py is that **a shape is a
field, not a polygon**, and this module takes that further than its siblings
because both of its solids are radial. A berry and a leaf are the same
primitive — `touch`, one brush-touch whose outline is a polar radius
function and whose ink is a function of (radius, angle) — differing only in
which two functions they hand it. Everything linear (vine, stalk, petiole,
vein, tendril) is orchid.py's `sweep`, which is the only line tool in the
family that does not bead at its joints, and a tendril is nothing but
joints.

Usage
-----
    python3 grape.py                              # one random painting
    python3 grape.py -n 6                         # six of them
    python3 grape.py --scene 架下 --air 秋风
    python3 grape.py --style 紫玉 --seed 7
    python3 grape.py --size 900x1300 -o scroll.png
    python3 grape.py --list                       # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Noise, Occupancy, BrushFont, clamp01, disc,
                    jitter, mix, rgb, paint_column, paint_seal,
                    TIANGAN, DIZHI)
# 一家之墨 — the ink table, the 兑水 rule, the full-sheet veil and the 留白
# moon are bamboo.py's. 葡萄 is painted with the same stone and the same
# water as 竹 and 兰, and two modules of this family disagreeing about what
# 淡墨 is would show at once on any sheet that carried both.
from bamboo import I, dilute, veil, moon
# The one borrowed *renderer*: a brush dragged along an arbitrary polyline,
# accumulated by maximum and composited once. `brush_path` in gongbi.py
# composites each of its segments separately and so lays ink down twice at
# every joint; on a tendril, which is a hundred joints in two turns of a
# spiral, that is a string of beads instead of a line.
from orchid import sweep

TAU = math.tau


# --------------------------------------------------------------------------
# 设色 — the four schools.
#
# 墨葡萄 is the orthodox form and the others are departures from it, so they
# all keep ink leaves and an ink vine: it is only the fruit that takes
# colour, and even then it takes it thinly. A grape painted at full
# saturation stops being 写意 and becomes a fruit bowl.
#
# Each pair is the two stops of a wash's ramp — the pale flank and the
# loaded core. Everything in this module multiplies into the paper rather
# than covering it, so the pale stop is a *dilution*, not a grey.
# --------------------------------------------------------------------------

STYLES = {
    "墨葡萄": dict(cn="墨葡萄", en="plain ink, after 徐渭",
                   berry=("#9C958A", "#211E1A"), leaf=("#8B8475", "#2E2A26"),
                   vine=("#7C7466", "#332E28"), vein="#201C19"),
    "紫玉": dict(cn="紫玉", en="violet fruit, ink leaves",
                 berry=("#B5A4BE", "#3A2141"), leaf=("#8C8A78", "#33382C"),
                 vine=("#857A66", "#3A322A"), vein="#241A2A"),
    "水晶": dict(cn="水晶", en="green — 青提",
                 berry=("#C4CBA6", "#5E6C3C"), leaf=("#9AA980", "#3A5134"),
                 vine=("#8C8468", "#413A28"), vein="#2A3A22"),
    "玛瑙": dict(cn="玛瑙", en="agate red",
                 berry=("#C2A69B", "#50221E"), leaf=("#8A8572", "#322F26"),
                 vine=("#857B68", "#3A3128"), vein="#2C1A14"),
}

# 章法 — the four framings.
#
# `Lf` is the length of the main vine as a fraction of the sheet's short
# side; `entry` says where it comes in and in roughly what direction, which
# is the one thing that decides whether the sheet reads as 折枝 (a cut
# branch, hung from the top edge, the orthodox grape composition) or as a
# plant standing in a scene (which grapes never do — nobody paints the
# root).
SCENES = {
    "垂藤": dict(cn="垂藤", en="one vine hanging from the top edge",
                 entry="top", Lf=(0.78, 1.00), nodes=(4, 6), branch=(1, 2),
                 bunches=(1, 2), leaves=(4, 6), curls=(3, 5), trellis=0.0),
    "双串": dict(cn="双串", en="two bunches, 一聚一散",
                 entry="top", Lf=(0.72, 0.94), nodes=(5, 7), branch=(2, 2),
                 bunches=(2, 2), leaves=(4, 7), curls=(3, 5), trellis=0.0),
    "架下": dict(cn="架下", en="under the trellis",
                 entry="pole", Lf=(0.68, 0.92), nodes=(4, 6), branch=(1, 2),
                 bunches=(1, 2), leaves=(3, 6), curls=(3, 6), trellis=1.0),
    "野藤": dict(cn="野藤", en="wild vine, mostly tendril — 闲抛闲掷",
                 entry="top", Lf=(0.88, 1.08), nodes=(5, 7), branch=(2, 3),
                 bunches=(1, 1), leaves=(2, 4), curls=(6, 9), trellis=0.0),
}

# 风致 — one air for the whole sheet, as bamboo.py has one weather.
# `wet` scales the softness of every edge, `dry` the 飞白 in the long
# strokes, `sweep` the lean everything that grows shares, `k` the overall
# strength of the ink.
AIR = {
    "静": dict(cn="秋实", en="still air",
               sweep=0.00, wet=1.00, dry=0.26, k=1.00),
    "秋风": dict(cn="风藤", en="autumn wind",
                 sweep=0.34, wet=0.84, dry=0.44, k=1.00),
    "带露": dict(cn="露珠", en="dew still on the fruit",
                 sweep=0.04, wet=1.90, dry=0.04, k=0.95, dew=1.0),
    "月下": dict(cn="月藤", en="by moonlight",
                 sweep=0.10, wet=1.30, dry=0.14, k=0.86,
                 veil=("#6E7A86", 0.10), moon=1.0),
}


# --------------------------------------------------------------------------
# 点 — one touch of the brush.
#
# Both solids in this painting are radial: a berry is a disc with an
# off-centre highlight, a leaf is a five-lobed star with a cleft. So both
# are one function — give it the outline as a radius against angle, and the
# ink as a shade against (radius, angle), and it rasterises the field the
# way gongbi.py's `Blade` does, with the edge falling out anti-aliased
# instead of supersampled.
#
# Two details earn their keep:
#
#   * the callables are *tabulated* before the pixel loop, na angles by nr
#     radii, so the cost of the ink field is fixed per touch instead of per
#     pixel. The quantisation is invisible: the shade is indexed into a
#     25-entry colour ramp anyway.
#     Measured at 900x1300: a leaf covers ~20 600 px and tabulates 7 560
#     calls (2.7x saved), the 破墨 pass ~5 200 px against 792 (6.6x) — and a
#     berry covers ~890 px against 3 712, which is a 4x *loss*. na and nr are
#     fixed per call site and the berry's are set for the biggest berry the
#     module can draw; scaling them with the radius is the honest fix and has
#     not been done, because all 95 touches on a sheet together cost 1.5 s
#     against 5.6 s of paper underneath them.
#   * the tone function returns a *pair* — shade and coverage. Coverage is
#     what makes 留白 possible. The highlight on a berry is not pale ink, it
#     is a region the touch does not cover, so the paper's own tone and
#     grain come through it; painted as a light grey it would read as a
#     sticker, and on 淡茶 paper it would read as a hole.
# --------------------------------------------------------------------------

def touch(cv, cx, cy, rad, tone, lo, hi, alpha=1.0, feather=1.2,
          squash=1.0, rot=0.0, na=224, nr=20, cover=0.0):
    """Composite one brush-touch: polar outline `rad`, ink field `tone`.

    `rad(th) -> radius` and `tone(u, th) -> (shade, cov)` are both in the
    touch's own frame, where angle 0 is +x and `u` is the distance out as a
    fraction of the outline's radius at that angle. `squash` flattens that
    frame across the y axis (a berry seen from slightly off, a leaf seen
    nearly edge-on) and `rot` turns it.

    `cover` is how much of the sheet's own tone this touch *replaces*
    instead of multiplying into, and it is the one way out of a trap that
    is otherwise inescapable: under a multiply composite ink can only
    darken, so nothing can ever hide anything. A bunch of grapes painted
    that way has no front and no back — every berry's pooled rim prints
    straight through every berry in front of it, the vine prints through
    the fruit, and what reaches the paper is a lattice of circles instead
    of a solid. 没骨 practice is the answer and it is also the licence: a
    berry is a *loaded body of colour* laid onto a sheet whose ink has
    already sunk, and it sits on top. At `cover=0` the behaviour is the old
    pure multiply, which is what the washes still want — a leaf is thin and
    the paper belongs underneath it.

    The algebra is free. Multiplying gives `o·c/255`, covering gives `c`,
    and the mix of the two is `c · (o·(1-cover)/255 + cover)` — one extra
    multiply-add per channel, and it can never exceed `c`, so a covering
    touch still cannot bleach the sheet past its own colour.
    """
    rt = [rad(i * TAU / na) for i in range(na)]
    rmax = max(rt)
    if rmax <= 0.4:
        return
    shd = [0.0] * (na * (nr + 1))
    cvr = [1.0] * (na * (nr + 1))
    for ia in range(na):
        th = ia * TAU / na
        base = ia * (nr + 1)
        for ir in range(nr + 1):
            s, c = tone(ir / nr, th)
            shd[base + ir] = s
            cvr[base + ir] = c

    pad = rmax + feather + 1.0
    x0, y0, x1, y1 = cv.clip(cx - pad, cy - pad, cx + pad, cy + pad)
    if x0 >= x1 or y0 >= y1:
        return
    ca, sa = math.cos(rot), math.sin(rot)
    inv_sq = 1.0 / max(0.05, squash)
    k_ang = na / TAU
    inv_f = 1.0 / feather
    acc = {}
    cw = cv.w
    for py in range(y0, y1):
        dy = py + 0.5 - cy
        row = py * cw
        for px in range(x0, x1):
            dx = px + 0.5 - cx
            lx = dx * ca + dy * sa
            ly = (-dx * sa + dy * ca) * inv_sq
            d = math.sqrt(lx * lx + ly * ly)
            if d > pad:
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
            if r < 0.4 or d > r + feather:
                continue
            cov = (r - d) * inv_f + 0.5
            if cov <= 0.0:
                continue
            if cov > 1.0:
                cov = 1.0
            u = d / r
            ir = int(u * nr)
            if ir > nr:
                ir = nr
            j = i0 * (nr + 1) + ir
            cov *= cvr[j]
            if cov <= 0.004:
                continue
            sh = shd[j]
            k = row + px
            v = acc.get(k)
            if v is None:
                acc[k] = [cov, sh]
            else:
                if cov > v[0]:
                    v[0] = cov
                if sh > v[1]:
                    v[1] = sh

    # 一次落纸 — one composite for the whole touch, so the ground's tone and
    # grain stay visible through the ink and a touch can only change the
    # sheet once however finely it was sampled.
    buf = cv.buf
    ramp = [mix(lo, hi, j / 24.0) for j in range(25)]
    mul = (1.0 - cover) * 0.00392
    for k, v in acc.items():
        a = alpha * v[0]
        if a <= 0.004:
            continue
        if a > 1.0:
            a = 1.0
        sh = v[1]
        if sh < 0.0:
            sh = 0.0
        elif sh > 1.0:
            sh = 1.0
        c = ramp[int(sh * 24.0)]
        i = k * 3
        o = buf[i]
        buf[i] = int(o + (c[0] * (o * mul + cover) - o) * a)
        o = buf[i + 1]
        buf[i + 1] = int(o + (c[1] * (o * mul + cover) - o) * a)
        o = buf[i + 2]
        buf[i + 2] = int(o + (c[2] * (o * mul + cover) - o) * a)


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
    """The point `d` pixels to one side of `pts[i]`, along its normal."""
    j = min(len(pts) - 1, i + 1)
    k = max(0, i - 1)
    dx, dy = pts[j][0] - pts[k][0], pts[j][1] - pts[k][1]
    n = math.hypot(dx, dy) or 1.0
    return pts[i][0] - dy / n * d, pts[i][1] + dx / n * d


# --------------------------------------------------------------------------
# 果 — the berry.
#
# One touch, and the whole subject rests on it. What makes an ink sphere
# read as a sphere and not as a dot:
#
#   * 留白 — a crescent of bare paper up toward the light, at about four
#     tenths of the radius out, never dead centre (that is a doughnut) and
#     never on the rim (that is a bite).
#   * a dark rim — a wet 没骨 touch dries with its pigment pulled to the
#     edge (水渍痕). The rim is also what separates one berry from the next
#     when they overlap, which is the whole reason a bunch is legible.
#   * the terminator — shade rising away from the light across the body,
#     which is the only part of this that is ordinary shading.
#   * 果脐 — the navel, a small dark press opposite the stalk. Cheap, and it
#     is what stops the berry from being a bead.
#
# 忌算盘珠 is enforced here as well as in the packing: every berry gets its
# own wobble on the outline and its own size, because a dozen identical
# circles read as a machine however they are arranged.
# --------------------------------------------------------------------------

def berry(cv, rng, x, y, r, st, air, back=0.0, alpha=1.0, nav=0.0,
          navdir=(0.0, 1.0), k=1.0, ink=1.0, cover=0.0):
    """One pearl. `back` in [0,1] pushes it behind the front of the bunch:
    paler, softer-edged, less highlight — 前实后虚.

    `ink` is how much ink this particular berry got. It comes from a field
    over the bunch rather than from this berry's own die roll, because
    墨色 varies by the *patch*: a painter recharges the brush, lays four or
    five berries from that load and recharges again, so the dark ones come
    in groups. Rolled per berry it reads as salt and pepper.
    """
    lt, dp = rgb(st["berry"][0]), rgb(st["berry"][1])
    b = clamp01(back)
    hx, hy = gb.LIGHT
    # a sphere seen from slightly above and off to one side
    squash = rng.uniform(0.90, 1.00)
    rot = rng.uniform(-0.5, 0.5)
    # The outline wants three harmonics and no fourth. An n=4 lobe at any
    # visible amplitude makes a rounded square, and a sheet of rounded
    # squares reads as dice however well the ink inside them behaves. So:
    # harmonic 1 shifts the centre (an egg), 2 ovals it, 3 takes the
    # symmetry away. Amplitudes stay small — a berry is round. It is only
    # *not quite* round.
    a1, a2, a3 = (rng.uniform(0.018, 0.048), rng.uniform(0.025, 0.060),
                  rng.uniform(0.012, 0.034))
    p1, p2, p3 = (rng.uniform(0, TAU) for _ in range(3))

    def rad(th):
        return r * (1.0 + a1 * math.cos(th + p1) + a2 * math.sin(2.0 * th + p2)
                    + a3 * math.sin(3.0 * th + p3))

    # The highlight sits toward the light and fades with depth: a berry at
    # the back of the bunch is in the bunch's own shadow and has none.
    #
    # It has to be *small*. The first version spread it over a third of the
    # radius at 0.86 strength and every berry came out an airbrushed egg —
    # which is what a photograph of a grape looks like and not at all what a
    # painted one does. A brush steered around a highlight leaves a nick of
    # paper a few millimetres across; the eye reads it as wet, and it only
    # reads as wet while it is small enough to be a *glint*.
    hl = (0.86 - 0.66 * b) * k
    if air.get("dew"):
        hl = min(0.98, hl * 1.08)
    # one light for the sheet, but the glint lands a little differently on
    # every berry, because every berry faces a little differently
    # It also sits well out toward the rim, not in the middle of the field.
    # A pale lobe with dark all round it is a specular highlight on glass;
    # 留白 is paper the stroke went *round*, so it opens onto the edge of
    # the touch on the side the brush came from.
    #
    # 0.54 of the radius was not out far enough, and the berries read as
    # bubbles because of it. With the pooled band now sitting *on* the
    # contour, a glint at 0.54 has dark on every side of it: a closed ring
    # with a pale blob inside is a lit sphere of glass, which is the one
    # thing it must not be. At 0.72 the lobe reaches the band and breaks it,
    # so the pale is paper the brush went round rather than light reflected
    # off a berry — and the gap it opens is the same gap the rim pools away
    # from, because both are the moment the hand lifted.
    #
    # And the clock position is the berry's own. A berry is a sphere on a
    # short stalk: which way its shoulder turns is its own business, and
    # with the glint pinned to the sheet's one light every berry opened its
    # paper on the same side and a bunch of thirty read as a tray of ball
    # bearings before it read as fruit. One angle per berry turns the glint
    # and — below — the gap in the pooled rim together, because they are
    # two halves of one event and must not drift apart; it is the event
    # that is not nailed to the light.
    hg = rng.uniform(-0.95, 0.95)
    ch, sg = math.cos(hg), math.sin(hg)
    gx0, gy0 = hx * ch - hy * sg, hx * sg + hy * ch
    hcx = gx0 * 0.72 + rng.uniform(-0.07, 0.07)
    hcy = gy0 * 0.72 + rng.uniform(-0.07, 0.07)
    # ...and it is a *glint*, so it is elongated across the light, not round:
    # a round dot on a round berry is a bubble. Two half-widths, the short
    # one along the light direction.
    sa = 2.0 * (0.100 + 0.026 * rng.random()) ** 2
    sb = 2.0 * (0.142 + 0.038 * rng.random()) ** 2
    # 墨不匀 — the ink inside one touch is never one value. Three sinusoids
    # cost nothing and are the difference between a painted berry and a
    # moulded one.
    mk1, mk2 = rng.uniform(2.0, 3.6), rng.uniform(2.4, 4.2)
    mp1, mp2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
    mt = (0.11 + 0.07 * rng.random()) * (1.0 - 0.6 * b)
    nvx, nvy = navdir
    # 水渍痕 — a wet 没骨 touch dries with its pigment pulled to the edge, so
    # the rim is the darkest part of the berry and the deep stop of the ramp
    # belongs there. It is also what separates one berry from the next where
    # they overlap, which is the whole reason a packed bunch is legible.
    base = (0.40 + 0.14 * rng.random()) * k * ink - 0.34 * b
    # 后虚 applies to the rim before anything else. Left at full strength on
    # the far layer, the pooled edge survived the paling of everything else
    # and the back of a bunch came out as a raft of pale rings — fish roe.
    # A berry deep in the bunch is in shadow: it has no rim because it has
    # no light to pool against, and no edge of its own worth drawing.
    rim = (0.62 + 0.16 * rng.random()) * k * ink * (1.0 - 0.58 * b)
    # ...and it does not pool evenly all the way round. The brush entered
    # somewhere and left somewhere, the sheet was not level, the berry next
    # door drank from this one: on one arc the rim is nearly black, on
    # another there is no rim at all and the two berries run together. That
    # unevenness is the single strongest difference between a painted berry
    # and a shaded sphere — a complete dark ring reads as a drawn outline,
    # and a drawn outline on a round thing reads as a bead.
    # Where the rim goes missing is not free either: it goes missing on the
    # lit side, because that is where the brush lifted and where the paper
    # was left. One event, two halves — a rim that fades exactly where the
    # 留白 opens is the whole look of a 没骨 berry, and the two drawn
    # independently just read as a glass bead with a scuff on it.
    la = math.atan2(gy0, gx0)
    pk1, pk2 = 1.0, rng.uniform(2.0, 3.0)
    # The gap follows the glint, and the glint follows this berry's own
    # hang — so the leash here only has to be the slop between the two
    # halves of one event, not the whole scatter. A wide leash here as
    # well would pull the pooled gap off the 留白 and leave a bead with a
    # scuff on it.
    pp1 = math.pi * 0.5 - la + rng.uniform(-0.45, 0.45)
    pp2 = rng.uniform(0, TAU)
    pd = 0.52 + 0.44 * rng.random()        # how much of the ring goes missing
    # The band of pigment sits *at* the contour. It was set in at 0.84 of
    # the radius with a width of 0.18, which leaves a pale margin outside
    # the dark band, and a dark ring with light on both sides of it is a
    # washer: the eye reads a second, smaller, drawn circle inside the
    # berry instead of an edge. 水渍痕 is pigment carried to where the water
    # stopped, so the crest is on the outline and the band is narrow enough
    # to be an edge rather than a zone.
    bc = 0.95 + 0.05 * rng.random()
    bw = 1.0 / (0.10 + 0.06 * rng.random()) ** 2
    # 一笔之中有浓淡 — and the interior is where it has to happen, because
    # once the pooled rim sits on the contour instead of inside it the rim
    # is an *edge*, and an edge round a flat field is an outline. A berry
    # drawn that way is a circle with a line round it however good the line
    # is. What a real one has is a load axis: the brush came down on one
    # side, so that side is 浓 and the other is nearly water, and the axis
    # has nothing to do with where the light is — it is where the hand was.
    ld = rng.uniform(0, TAU)
    ldx, ldy = math.cos(ld), math.sin(ld)
    load = (0.26 + 0.20 * rng.random()) * k * (1.0 - 0.55 * b)
    # the light's own contribution is the small part of it
    lit = (0.10 + 0.05 * rng.random()) * k * (1.0 - 0.5 * b)

    def tone(u, th):
        ux, uy = u * math.cos(th), u * math.sin(th)
        gx, gy = ux - hcx, uy - hcy
        ga = gx * gx0 + gy * gy0
        gb_ = -gx * gy0 + gy * gx0
        g = math.exp(-(ga * ga / sa + gb_ * gb_ / sb))
        pool = 1.0 - pd * (0.5 + 0.5 * math.sin(pk1 * th + pp1)) \
            * (0.62 + 0.38 * (0.5 + 0.5 * math.sin(pk2 * th + pp2)))
        du = u - bc
        sh = (base + rim * pool * (math.exp(-du * du * bw) + 0.14 * u ** 8)
              + load * (ux * ldx + uy * ldy)
              - lit * (ux * hx + uy * hy)
              + mt * math.sin(ux * mk1 + mp1) * math.sin(uy * mk2 + mp2))
        if nav:
            dx2, dy2 = ux - nvx * 0.52, uy - nvy * 0.52
            sh += nav * math.exp(-(dx2 * dx2 + dy2 * dy2) / 0.10)
        # The glint is cut off rather than faded out. A gaussian's tail left
        # a third of the berry faintly pale and the fruit read as a bean; a
        # brush steered round a nick of paper leaves an edge.
        q = (g - 0.14) * 1.163
        return sh, 1.0 if q <= 0.0 else 1.0 - hl * q ** 1.35

    # na and nr were fixed at 128 and 28, set for the biggest berry the
    # module can draw, and the docstring on `touch` has been admitting for
    # two versions that this is a 4x *loss* on a small one — 3 712 tabulated
    # calls to cover 890 pixels. Now that a bunch is a 3-D pack and carries
    # half again as many berries, the honest fix pays for the extra fruit:
    # the outline wants about a sample per pixel of its own circumference
    # and the ink field about one per pixel of radius, and that is this.
    na = max(56, min(144, int(r * 5.2)))
    nr = max(10, min(28, int(r * 1.05)))
    # 后虚 is a matter of *value*, not of transparency. Fading the far
    # berries out by 42% of their alpha made them ghosts: the leaf behind
    # the bunch printed straight through the fruit at the tip, and a pale
    # ring you can see another berry inside is a soap bubble. A painter
    # does not thin the paint to put a berry at the back, they thin the
    # *ink* — the touch is just as opaque and carries less black. So the
    # alpha barely moves and the paling is all in `base`, `rim` and the
    # feather.
    touch(cv, x, y, rad, tone, dilute(lt, 0.86), dp,
          alpha=alpha * (1.0 - 0.14 * b),
          feather=(0.9 + 2.6 * b) * air["wet"] ** 0.4,
          squash=squash, rot=rot, na=na, nr=nr, cover=cover)


# --------------------------------------------------------------------------
# 一串 — the bunch.
#
# 下尖上宽: a cone, broad at the shoulder where the stalk enters and
# tapering to a single berry at the point.
#
# 一串是一个体，不是一片 — and this is the whole of it. The first two
# versions packed the fruit as discs on the *paper*: a point in a flat cone,
# a minimum separation a little under two radii so that neighbours overlap,
# and a `front` number invented afterwards to decide which of them got
# painted last. Everything that was wrong with the result followed from
# that one decision. Berries overlapped because they had been allowed to
# interpenetrate, which is a thing grapes cannot do, so the overlaps landed
# anywhere and read as a Venn diagram. `front` was a separate die roll, so
# the berry that was painted on top was not the berry that was nearest to
# anything — a bunch cannot have a front if its depth is not where its
# shape is. And the two-pass near/far split made the hierarchy a step
# function with two values when what it needs is a gradient.
#
# So the pack is in three dimensions. A point in a *cone of revolution*,
# minimum separation just under the sum of the radii (grapes press each
# other a little and do not pass through each other at all), and then the
# projection does all the rest of the work for nothing:
#
#   depth      is the z the berry was packed at — no second invention
#   overlap    is a consequence of depth, so it is always a near berry
#              lying over a far one and never two berries cutting
#   size       grows a little toward the viewer (perspective)
#   value      pales away from the viewer (前实后虚, now a gradient)
#   softness   likewise, so the back of the bunch merges into one mass
#   order      is just a sort on z
#
# The remaining piece is that depth is only legible if the near berry
# actually *hides* the far one, which pure multiply cannot do. That is what
# `cover` is for; see `touch`.
# --------------------------------------------------------------------------

def bunch(cv, rng, x, y, L, W, st, air, lean=0.0, k=1.0, root=None):
    """A bunch whose shoulder is at (x, y), hanging `L` downward.

    Returns the list of (x, y, r) it painted, for the occupancy map.
    """
    # A berry is about a fifth of the width of a leaf and a bunch is four or
    # five of them across — it was a third and three across to begin with,
    # and at that size the fruit was plums and the bunch was a handful of
    # them dropped on the paper with daylight between.
    r0 = W * rng.uniform(0.105, 0.135)
    # the spine of the bunch: hanging, but never plumb
    bow = rng.uniform(-0.22, 0.22) + lean * 0.5
    tip = (x + L * (bow + lean * 0.30), y + L)

    def spine(t):
        s = 1.0 - t
        # quadratic: the bunch starts straight down and swings late
        mx = x + L * bow * 0.18
        my = y + L * 0.52
        return (s * s * x + 2 * s * t * mx + t * t * tip[0],
                s * s * y + 2 * s * t * my + t * t * tip[1])

    def halfw(t):
        # broad shoulder, a slight swell below it, then out to one berry
        return W * 0.5 * max(0.10, (1.0 - t ** 1.9) ** 0.72) * \
            (1.0 + 0.10 * math.sin(t * 4.0))

    # how deep the bunch is against how wide, and how hard the fruit
    # presses: at jam=1 the berries merely kiss, and a real bunch is packed
    # tight enough to deform
    dep_k = rng.uniform(0.78, 1.06)
    jam = rng.uniform(0.84, 0.93)

    # 累累 — how many. Taken from the *volume* of the cone now rather than
    # from its area, at a packing fraction a little below random-close, so a
    # bunch is full. The old area figure came out at about 1.2 berry-areas
    # per cone-area, which is a cone with daylight in it: the sheets showed
    # a ring of fruit with a hole in the middle, and the second pass existed
    # to fill the hole.
    vol = 0.0
    for i in range(24):
        t = (i + 0.5) / 24.0
        vol += math.pi * halfw(t) ** 2 * dep_k * (L / 24.0)
    want = int(vol * 0.54 / (4.18879 * r0 ** 3) * rng.uniform(0.88, 1.12))
    want = max(12, min(74, want))

    def t_at(cy):
        """Where on the axis a given height is — the spine is monotonic in
        y, so a dozen halvings is exact enough and survives the spine being
        rewritten."""
        lo, hi = 0.0, 1.0
        for _ in range(14):
            mid = (lo + hi) * 0.5
            if spine(mid)[1] < cy:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    # 粒粒相连 — the bunch is *grown*, not scattered.
    #
    # Dart-throwing was the fault behind the loose fruit. It only ever
    # forbade berries from coming too close, and said nothing about their
    # coming close enough: a berry thrown at the rim of the cone with its
    # neighbours thrown elsewhere stayed exactly where it landed, and the
    # sheet showed a pearl hanging in clear air a radius clear of the
    # fruit. Pruning the stragglers afterwards was tried and is the wrong
    # shape of fix — in three dimensions a dart-thrown pack at this density
    # is mostly stragglers, and one bunch in five lost three quarters of
    # its fruit to the pruner.
    #
    # Growing it cures the disease instead. Put one berry on the axis at
    # the shoulder; then every further berry is placed *touching* one that
    # is already there, in a random direction with a downward preference,
    # at a centre distance of `jam` times the sum of the two radii — so it
    # presses into its neighbour a little, as fruit does. Rejected if it
    # collides with a third berry or if its centre leaves the cone.
    # Connectivity is then structural: there is no arrangement of these
    # berries in which one of them is loose.
    seed_t = 0.03 + 0.04 * rng.random()
    sx, sy = spine(seed_t)
    pack = [(sx, sy, 0.0, r0 * rng.uniform(0.92, 1.08), seed_t)]
    y_top = y - r0 * 0.8
    guard = want * 44
    while len(pack) < want and guard > 0:
        guard -= 1
        ax, ay, az, ar, at = pack[rng.randrange(len(pack))]
        r = r0 * rng.uniform(0.84, 1.16) * (1.0 - 0.13 * at)
        ux, uy, uz = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        n = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        # 下垂 — grapes hang. An isotropic direction grows a ball, and a
        # ball is the one thing 一串成形 forbids; the cone bound below
        # would carve a cone out of it in the end, but only by throwing
        # most of the work away.
        ux, uy, uz = ux / n, uy / n + 0.42, uz / n
        n = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        d = (ar + r) * jam / n
        bx, by, bz = ax + ux * d, ay + uy * d, az + uz * d
        if by < y_top:
            continue
        t = t_at(by)
        cx2, cy2 = spine(t)
        rho = halfw(t)
        if (bx - cx2) ** 2 + (bz / dep_k) ** 2 > rho * rho:
            continue
        ok = True
        for p in pack:
            md = (r + p[3]) * jam
            if ((bx - p[0]) ** 2 + (by - p[1]) ** 2
                    + (bz - p[2]) ** 2 < md * md - 0.5):
                ok = False
                break
        if ok:
            pack.append((bx, by, bz, r, t))

    lt, dp = rgb(st["vine"][0]), rgb(st["vine"][1])
    # 果梗 — the stalk into the shoulder and the little rachis inside. Drawn
    # first, so the fruit covers where they join; what is left showing is
    # the few millimetres between berries that a real bunch shows.
    # 果蒂 — the stalk, from the node it hangs off to the shoulder of the
    # bunch. `root` is that node and it is not optional in practice: the
    # first version started the stalk a fixed distance above the shoulder
    # and ran it to nowhere, which drew a straight dark stick into open
    # paper beside the fruit on every sheet. A bunch of grapes is the one
    # thing in this painting that cannot be 折枝 — it has to be joined.
    if root is not None:
        sw = max(0.9, r0 * 0.13)
        rx, ry = root
        sx, sy = spine(0.08)
        # a hanging stalk bows under the weight it carries
        mx = (rx + sx) * 0.5 + rng.uniform(-0.10, 0.10) * L
        my = (ry + sy) * 0.5 + abs(ry - sy) * 0.10 + L * 0.03
        drag(cv, _arc([(rx, ry), (mx, my), (sx, sy)], 14),
             sw * 1.7, sw, dilute(lt, 0.9), dp,
             0.88 * k, d0=0.55, d1=0.9, core=0.34)
    # 投影 — out of the pack onto the paper. `zn` runs -1 at the back of the
    # bunch to +1 at the front and is the only depth number from here on:
    # size, value, softness, the navel, the glint and the painting order
    # are all read off it, so they cannot disagree with each other the way
    # a separately-rolled `front` could.
    gz = max(1e-6, max(abs(p[2]) for p in pack))
    # 墨色以块分 — the ink field. One patch of the bunch is dark and another
    # is thin, because a brush is recharged every four or five berries;
    # rolled per berry this was salt and pepper, and a bunch with the dark
    # fruit scattered through it at random has no weight anywhere.
    fld = Noise(rng, 3, 4)
    span = max(1.0, L)
    shot = []
    for bx, by, bz, r, t in pack:
        zn = bz / gz
        shot.append((bx, by - bz * 0.17, r * (1.0 + 0.13 * zn), zn, t, bz))
    shot.sort(key=lambda p: p[3])          # 后者先画 — back to front

    out = []
    for px, py, pr, zn, t, bz in shot:
        # 前实后虚 as a gradient rather than two layers. The exponent keeps
        # the front half of the bunch nearly fully inked and spends the
        # paling on the back half, which is where the eye reads depth.
        b = clamp01((0.5 - 0.5 * zn) ** 1.35 * 0.88)
        # 墨分五色 wants the *whole* range, not a tenth of it either side of
        # the middle. At ±20% every berry on the sheet came out the same
        # mid tone and the bunch had no weight anywhere; a real 墨葡萄 runs
        # from near-black fruit to fruit that is barely a stain.
        ink = 0.72 + 0.76 * fld.at(clamp01((px - x) / span + 0.5),
                                   clamp01((py - y) / span))
        # 果梗 — the stub of rachis this berry hangs by, drawn immediately
        # before its own berry and therefore hidden by every berry in front
        # of it, which is what a rachis inside a bunch actually is: a few
        # millimetres of it show between fruit and the rest is buried.
        # (Before `cover` existed this could not be done at all — a stalk
        # that passed under a berry printed straight through it, so stalks
        # had to be cut short and kept off the fruit by hand.)
        if zn > -0.45 and rng.random() < 0.34:
            sx, sy = spine(max(0.0, t - 0.12))
            dx, dy = sx - px, sy - py
            d = math.hypot(dx, dy) or 1.0
            # short. A stub that reaches a couple of radii crosses the fruit
            # behind it, and a straight dark bar lying over a pale berry
            # reads as a scratch on the paper rather than as a stalk in a
            # gap. What shows between real berries is millimetres.
            ln = min(d, pr * rng.uniform(0.55, 1.25))
            ax, ay = px + dx / d * ln, py + dy / d * ln
            mx = (ax + px) * 0.5 - dy / d * pr * 0.18
            my = (ay + py) * 0.5 + dx / d * pr * 0.18
            drag(cv, _arc([(ax, ay), (mx, my), (px, py - pr * 0.55)], 8),
                 max(0.55, r0 * 0.09), max(0.45, r0 * 0.06),
                 dilute(lt, 0.8), dp, 0.52 * k * (0.55 + 0.45 * zn),
                 d0=0.5, d1=0.85)
        # 果脐 — the navel, on the fruit that hangs toward the eye and down
        nav = 0.0
        if zn > 0.25 and t > 0.26 and rng.random() < 0.55:
            nav = rng.uniform(0.18, 0.34)
        berry(cv, rng, px, py, pr, st, air, back=b,
              alpha=0.97, nav=nav, navdir=(bow * 0.4, 1.0), k=k, ink=ink,
              # 没骨 — the fruit covers. Not quite wholly: a berry at the
              # back of the bunch is in its shadow and has given up most of
              # its body, so it stays the thin wash it always was.
              cover=0.76 - 0.18 * b)
        out.append((px, py, pr))
    return out


def _arc(ctrl, n):
    """A quadratic or cubic through `ctrl`, sampled `n` ways — the small
    curve helper the stalks and veins want."""
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


# --------------------------------------------------------------------------
# 叶 — the leaf.
#
# A grape leaf is cordate-palmate: five lobes, sinuses between them, a
# serrated edge, and a deep cleft at the base where the petiole comes in.
# As a polar field with the petiole at the origin that is four factors
# multiplied together, and the cleft comes for free — the radius simply goes
# small in the direction the stalk comes from.
#
#   body   cos(φ/2)² — long away from the stalk, nearly nothing behind it
#   sinus  cos(5φ) — five lobes, peaks at the apices, notches between
#   skew   one slow term so no two leaves are the same and none is
#          symmetric about its own midrib (忌对称)
#   teeth  a fine ripple, four or five pixels deep — 锯齿
#
# The wash is deliberately mottled and the veins go on top of it in 浓墨,
# which is the actual order of work: lay the leaf wet, let it sink, then
# write the ribs into it while it is still damp (破墨).
# --------------------------------------------------------------------------

def _leaf_rad(rng, L, dpt, tooth, nt):
    ph1, ph2, ph3 = (rng.uniform(0, TAU) for _ in range(3))
    a1, a2 = rng.uniform(0.05, 0.12), rng.uniform(0.03, 0.07)
    sk = rng.uniform(-0.20, 0.20)

    def rad(th):
        p = th if th <= math.pi else th - TAU
        c = math.cos(p * 0.5)
        # 0.42 to begin with, which made a disc with a dent in it: the leaf
        # has to be *long* toward its apex and short across its shoulders or
        # nothing about it says which way it is pointing.
        body = 0.13 + 0.87 * (c * c) ** 0.60
        sinus = 1.0 - dpt * ((1.0 - math.cos(5.0 * p)) * 0.5) ** 0.62
        skew = (1.0 + sk * math.sin(p) + a1 * math.cos(p + ph1)
                + a2 * math.cos(2.0 * p + ph2))
        teeth = tooth * ((1.0 + math.sin(nt * p + ph3)) * 0.5) ** 0.7
        return L * (body * sinus * skew + teeth)
    return rad


def leaf(cv, rng, x, y, ang, L, st, air, back=0.0, k=1.0, curl=None):
    """One leaf, its petiole joining at (x, y) and its apex pointing `ang`.

    Returns (cx, cy, radius) for the occupancy map — the centre of mass,
    not the attachment, because that is what the inscription has to dodge.
    """
    lt, dp = rgb(st["leaf"][0]), rgb(st["leaf"][1])
    b = clamp01(back)
    dpt = rng.uniform(0.26, 0.42)
    tooth = rng.uniform(0.030, 0.052)
    rad = _leaf_rad(rng, L, dpt, tooth, rng.randint(26, 40))
    # 翻叶 — a leaf is a plate and we are not above it. Squash it across an
    # axis that is not its own midrib, so the foreshortening is of a thing
    # in space rather than of a drawing.
    squash = curl if curl is not None else rng.uniform(0.52, 0.98)
    mot = Noise(rng, 7, 7)
    hx, hy = gb.LIGHT
    # 墨分五色 — darker and much more varied than the first version, which
    # laid every leaf at one value and got grey card. A 写意 leaf is one
    # wet touch of a brush loaded unevenly: part of it is 浓墨, part is
    # barely more than water, and the join between them is the subject.
    base = (0.30 + 0.22 * rng.random()) * k - 0.24 * b
    rim = (0.46 + 0.18 * rng.random()) * k
    wob = (0.40 + 0.22 * rng.random())
    dry = clamp01(air["dry"] * rng.uniform(0.6, 1.4) - 0.10) * (1.0 - 0.4 * b)
    # 一笔之中有浓淡 — the single biggest thing a 写意 leaf has that a filled
    # shape does not: one side of it is loaded and the other is nearly
    # water, on an axis that has nothing to do with the midrib. Without this
    # ramp every leaf came out one value, and a leaf at one value is a piece
    # of grey card with a serrated edge however good the edge is.
    gd = rng.uniform(0, TAU)
    gax, gay = math.cos(gd), math.sin(gd)
    grad = (0.34 + 0.26 * rng.random()) * k * (1.0 - 0.5 * b)

    def tone(u, th):
        ux, uy = u * math.cos(th), u * math.sin(th)
        # The mottle is sampled in the leaf's own *plane*, not in (u, th).
        # Polar noise looks like polar noise: concentric rings out from the
        # stalk and a burst of spokes at the centre where every angle meets.
        m = mot.at(clamp01(ux * 0.47 + 0.5), clamp01(uy * 0.47 + 0.5))
        sh = (base + rim * u ** 3.0 + (m - 0.5) * wob
              + grad * (ux * gax + uy * gay) * 0.5
              - 0.10 * k * u * (math.cos(th) * hx + math.sin(th) * hy))
        cov = 1.0
        if dry:
            # 飞白 — a leaf laid with a brush that has given most of its
            # water up skips, and it skips in the body, not on the rim
            # where the ink has pooled.
            sk = m + 0.35 * mot.at(clamp01(ux * 1.3 + 0.5),
                                   clamp01(uy * 1.1 + 0.5)) - 0.17
            if sk > 0.56:
                cov = max(0.18, 1.0 - dry * (sk - 0.56) * 3.4
                          * (1.0 - u ** 4))
        return sh, cov

    touch(cv, x, y, rad, tone, dilute(lt, 0.92 - 0.25 * b), dp,
          alpha=(0.94 - 0.34 * b) * k, feather=0.95 * air["wet"] ** 0.4,
          squash=squash, rot=ang, na=360, nr=20,
          # A leaf is a thin wash and the paper belongs under it, so this is
          # a quarter of what the fruit takes — just enough that a near leaf
          # is in front of the vine it hangs from and of the trellis bar
          # behind it, instead of having both printed across it. 远叶 keep
          # none of it: they are behind those things anyway.
          cover=0.34 * (1.0 - b) ** 2)

    # 破墨 — a second, wetter, darker pass over part of the leaf. One of the
    # two reasons an ink leaf looks wet; the other is the rim.
    if b < 0.5 and rng.random() < 0.62:
        q = rng.uniform(0.34, 0.52)
        ph = rng.uniform(0, TAU)
        qx = x + math.cos(ang + ph * 0.0 + rng.uniform(-1.1, 1.1)) * L * 0.42
        qy = y + math.sin(ang + rng.uniform(-1.1, 1.1)) * L * 0.42 * squash
        r2 = _leaf_rad(rng, L * q, rng.uniform(0.16, 0.30), tooth * 0.6, 17)
        touch(cv, qx, qy, r2,
              lambda u, th: (0.52 + 0.30 * u ** 3, 1.0),
              dilute(lt, 0.8), dp, alpha=0.30 * k,
              feather=2.6 * air["wet"] ** 0.5, squash=squash,
              rot=ang + rng.uniform(-0.8, 0.8), na=72, nr=10)

    # 叶脉 — five main ribs out to the lobe apices, then a few seconds off
    # them. Written into the damp wash in 浓墨, so they are the darkest
    # thing on the leaf and the only hard line in it.
    ca, sa = math.cos(ang), math.sin(ang)

    def w2(lx, ly):                            # leaf frame -> paper
        # 脉不出叶 — pull the point back inside the blade first. A rib runs
        # to a lobe apex and a secondary runs off it at an angle, and the
        # second of those overshot the edge often enough to put black
        # whiskers on the paper around half the leaves on the first sheet.
        # The outline is a radius function, so the test is one call to it.
        d = math.hypot(lx, ly)
        if d > 1e-6:
            m = rad(math.atan2(ly, lx) % TAU) * 0.90
            if d > m:
                lx *= m / d
                ly *= m / d
        ly *= squash
        return (x + lx * ca - ly * sa, y + lx * sa + ly * ca)

    vi = rgb(st["vein"])
    # Paler and thinner than the first version, which drew all five ribs at
    # one weight in near-black and turned every leaf into a drafting
    # diagram. 写意 veins are a few quick strokes into damp paper: the
    # midrib is the only one that is sure of itself, and the outer pair are
    # often just suggested or left out.
    # 叶脉 are a hint, not a diagram. Any more than this and the ribs plus
    # their 支脉 read as an armature inside the leaf rather than as folds
    # of it — the wash is the leaf and the veins are only where it creases.
    va = (0.44 - 0.28 * b) * k * rng.uniform(0.78, 1.12)
    vw = max(0.50, L * 0.011)
    for j, p in enumerate((0.0, TAU / 5.0, -TAU / 5.0,
                           2.0 * TAU / 5.0, -2.0 * TAU / 5.0)):
        if j >= 3 and rng.random() < 0.34:
            continue
        # 主脉 — short of the rim, and bowed. A rib taken all the way out
        # at a constant heading is a ruled line, and five ruled lines out
        # of one point is a drafting diagram of a leaf rather than a leaf:
        # the thing that reads as a vein is a line that bends with the
        # blade it supports and dies before the edge.
        rr = rad(p % TAU) * rng.uniform(0.56, 0.78)
        bend = rng.uniform(0.22, 0.62) * rng.choice((1.0, -1.0))
        pts = [w2(math.cos(p) * rr * f - bend * rr * f * (1.0 - f) * 0.7
                  * math.sin(p),
                  math.sin(p) * rr * f + bend * rr * f * (1.0 - f)
                  * math.cos(p))
               for f in (i / 13.0 for i in range(14))]
        prom = 1.0 if j == 0 else rng.uniform(0.56, 0.90)
        drag(cv, pts, vw * prom, vw * 0.18, dilute(vi, 0.40), vi,
             va * (1.0 if j == 0 else rng.uniform(0.6, 0.95)),
             d0=0.55, d1=0.98, taper=0.8, feather=0.9, core=0.25)
        if b > 0.45:
            continue
        for _ in range(rng.randint(0, 2)):     # 支脉
            f = rng.uniform(0.26, 0.74)
            side = rng.choice((-1.0, 1.0))
            a2 = p + side * rng.uniform(0.42, 0.80)
            l2 = rr * rng.uniform(0.18, 0.34)
            o = (math.cos(p) * rr * f, math.sin(p) * rr * f)
            pts = [w2(o[0] + math.cos(a2) * l2 * g,
                      o[1] + math.sin(a2) * l2 * g)
                   for g in (i / 6.0 for i in range(7))]
            drag(cv, pts, vw * 0.44, vw * 0.14, dilute(vi, 0.32), vi,
                 va * 0.56, d0=0.5, d1=0.92, feather=0.9, core=0.25)

    cx = x + math.cos(ang) * L * 0.34
    cy = y + math.sin(ang) * L * 0.34 * (0.5 + 0.5 * squash)
    return cx, cy, L * 0.80


# --------------------------------------------------------------------------
# 藤 — the vine, and 卷须 — the tendril.
#
# 藤如草书. The path is got by integrating a curvature that *changes sign*,
# which is the whole difference between a vine and a hose: a single bezier
# bends one way and reads as a pipe however thick and thin it is made.
# Width then swells at the nodes and the ink runs 浓淡相间 along the length,
# because one load of ink does.
# --------------------------------------------------------------------------

def vine_path(rng, x, y, a0, L, n=96, turn=2.2, waves=2.4, drift=0.0,
              bounds=None, slack=0.08, cone=1.15):
    """A written line: arc length `L`, total turning about `turn` radians,
    the curvature reversing `waves` times. Truncated if it leaves the
    sheet, because a vine that walks off the paper took the picture with
    it.

    `cone` bounds the heading to that many radians either side of `a0`.
    Free wandering is what makes the line look written, and it is also what
    let the first version turn a hanging vine through ninety degrees and
    walk it out of the right-hand edge — taking the composition's whole
    vertical with it. A 草书 stroke wanders inside a gesture; it does not
    change its mind about the gesture.
    """
    pts = [(x, y)]
    ang = a0
    ph = rng.uniform(0, TAU)
    step = L / n
    for i in range(1, n + 1):
        t = i / n
        ang += (turn * (math.sin(t * waves * TAU + ph)
                        + 0.35 * math.sin(t * waves * 2.3 * TAU + ph * 1.7))
                + drift) / n
        ang += rng.uniform(-0.012, 0.012)      # 颤笔
        if cone is not None:
            if ang > a0 + cone:
                ang = a0 + cone
            elif ang < a0 - cone:
                ang = a0 - cone
        x += math.cos(ang) * step
        y += math.sin(ang) * step
        if bounds is not None:
            W, H = bounds
            if not (-slack * W < x < (1 + slack) * W
                    and -slack * H < y < (1 + slack) * H):
                break
        pts.append((x, y))
    return pts


def vine(cv, rng, pts, w0, w1, st, air, alpha=0.95, knots=3.0, sinew=True,
         wide=3.4):
    """老藤 — woody, knotted, dry at the far end, with a fibre riding
    inside it."""
    n = len(pts) - 1
    if n < 1:
        return
    lt, dp = rgb(st["vine"][0]), rgb(st["vine"][1])
    ph = rng.uniform(0, TAU)
    hws, dens = [], []
    # 提按 — the hand rises and presses as it goes. The first version
    # modulated the width by 22% and the result was an even tube with a
    # ripple in it: what says 老藤 is a stroke that nearly doubles and
    # nearly disappears inside its own length, and whose value changes with
    # its width, because pressing a brush down puts more ink out as well as
    # more hair.
    ph2 = rng.uniform(0, TAU)
    kn2 = knots * rng.uniform(1.7, 2.6)
    for i in range(n + 1):
        t = i / n
        p = (0.36 * math.sin(t * knots * TAU + ph)
             + 0.17 * math.sin(t * kn2 * TAU + ph2)
             + rng.uniform(-0.055, 0.055))
        hws.append(max(0.45, (w0 + (w1 - w0) * t) * (1.0 + p)))
        dens.append(clamp01(0.36 + 0.44 * t + 0.20 * math.sin(t * 5.3 + ph)
                            + 0.26 * p))
    sweep(cv, pts, hws, dens, dilute(lt, 0.86), dp, alpha,
          feather=0.9 * air["wet"] ** 0.3, core=0.44,
          side=rng.uniform(-0.20, 0.20), dry=air["dry"] * 1.15,
          dry_k=2.1, dry_ph=rng.uniform(0, TAU))
    # 筋 — the fibre. A thin dark line *inside* the stroke, not along its
    # edge: 老藤 is ropy and this is the cheapest way to say so.
    # ...but only in a stroke wide enough to hold one. On a branch it does
    # not ride *inside* the stroke, it rides alongside it, and a thin dark
    # line alongside a thin pale line is not a ropy vine — it is the branch
    # drawn twice.
    if sinew and w0 > wide:
        d = rng.choice((-1.0, 1.0))
        off = [_perp(pts, i, d * hws[i] * rng.uniform(0.22, 0.40))
               for i in range(0, n + 1, 2)]
        drag(cv, off, w0 * 0.20, w1 * 0.20, I("zhong"), I("jiao"),
             0.30 * air["k"], d0=0.4, d1=0.85, feather=0.8, core=0.15)


def tendril(cv, rng, x, y, a0, L, st, air, hand=1.0, k=1.0):
    """卷须 — a run, then a spiral of one and a half to three turns.

    The spiral is logarithmic, which is why it looks alive: an Archimedean
    coil has even gaps and reads as a watch spring. It is also the last
    thing drawn on the sheet, in the thinnest and driest line on it, because
    that is when a painter adds them and that is what they weigh.
    """
    lt = rgb(st["vine"][0])
    run = L * rng.uniform(0.30, 0.62)
    pts = []
    nr_ = 16
    bend = rng.uniform(-0.9, 0.9)
    for i in range(nr_ + 1):
        t = i / nr_
        a = a0 + bend * t * t
        pts.append((x + math.cos(a) * run * t, y + math.sin(a) * run * t))
    # the coil hangs off the end of the run, offset to the winding side
    ex, ey = pts[-1]
    ea = a0 + bend
    turns = rng.uniform(1.2, 3.1)
    r0 = L * rng.uniform(0.11, 0.24)
    decay = math.log(rng.uniform(3.2, 10.0)) / (turns * TAU)
    ccx = ex - math.sin(ea) * hand * r0
    ccy = ey + math.cos(ea) * hand * r0
    p0 = math.atan2(ey - ccy, ex - ccx)
    nc = max(26, int(turns * 24))
    # A clean logarithmic spiral about a fixed centre is a watch spring, and
    # the first sheet had five of them on it looking stamped out of a
    # catalogue. A real 卷须 coils around something, so its centre *travels*
    # along the axis as it winds, and the turns are not quite round.
    ph = rng.uniform(0, TAU)
    wob = rng.uniform(0.05, 0.14)
    axis = rng.uniform(0.10, 0.34) * r0
    cax, cay = math.cos(ea), math.sin(ea)
    for i in range(1, nc + 1):
        f = i / nc
        phi = turns * TAU * f
        r = r0 * math.exp(-decay * phi) * (1.0 + wob * math.sin(phi * 1.7 + ph))
        a = p0 + hand * phi
        pts.append((ccx + cax * axis * f + math.cos(a) * r,
                    ccy + cay * axis * f + math.sin(a) * r))
    w = max(0.78, L * 0.019)
    drag(cv, pts, w, w * 0.30, dilute(lt, 0.7), rgb(st["vein"]),
         (0.80 + 0.14 * rng.random()) * k, d0=0.46, d1=0.98, taper=0.75,
         feather=0.8, core=0.22, dry=air["dry"] * 0.8,
         dry_ph=rng.uniform(0, TAU))


def trellis(cv, rng, W, H, st, air, side):
    """架 — the frame the vine is grown on: two or three poles, kept pale
    and kept straight, so they read as behind everything and as made by a
    person. They are the only straight lines this painter draws."""
    # Pale, and broken. At full 淡墨 and unbroken these were two grey bars
    # of even weight running the width of the sheet — scaffolding, not a
    # trellis, and the only thing on the paper with no brush in it. 架 is
    # 远景: it is there to be looked past.
    ink = dilute(I("dan"), 0.56 * air["k"])
    ink2 = dilute(I("zhong"), 0.52 * air["k"])
    y0 = H * rng.uniform(0.14, 0.24)
    sag = H * rng.uniform(0.004, 0.016)
    tilt = rng.uniform(-0.016, 0.016)
    w = max(1.3, min(W, H) * 0.0040)

    def barpt(u):                              # u in [-0.04, 1.04]
        x = -W * 0.04 + W * 1.08 * u
        return (x, y0 + sag * math.sin(math.pi * u) + tilt * (x - W * 0.5)
                + rng.uniform(-0.7, 0.7))

    # 笔断意连 — the bar is laid in two or three runs with paper between
    # them, the way one dry stroke actually crosses a wide sheet. The gaps
    # are where a leaf or a bunch will pass in front of it.
    cuts = sorted(rng.uniform(0.18, 0.82) for _ in range(rng.randint(1, 2)))
    runs, u = [], 0.0
    for c in cuts + [1.04]:
        if c - u > 0.12:
            runs.append((u, c))
        u = c + rng.uniform(0.03, 0.09)
    for (ua, ub) in runs:
        n = max(3, int((ub - ua) * 18))
        seg = [barpt(ua + (ub - ua) * i / n) for i in range(n + 1)]
        drag(cv, seg, w * rng.uniform(0.9, 1.1), w * 0.88, ink, ink2,
             rng.uniform(0.34, 0.46), d0=0.40, d1=0.56, feather=1.2,
             core=0.5, dry=0.72, dry_k=3.2, dry_ph=rng.uniform(0, TAU))
    # one or two uprights, off to the side the inscription will not want
    for _ in range(rng.randint(1, 2)):
        px = W * (rng.uniform(0.62, 0.92) if side else rng.uniform(0.08, 0.38))
        top = y0 - H * rng.uniform(0.04, 0.10)
        bot = y0 + H * rng.uniform(0.14, 0.34)
        lean = rng.uniform(-0.03, 0.03)
        col = [(px + rng.uniform(-0.9, 0.9) + (y - y0) * lean, y)
               for y in (top + (bot - top) * i / 12.0 for i in range(13))]
        drag(cv, col, w * 0.88, w * 0.62, ink, ink2, rng.uniform(0.26, 0.36),
             d0=0.36, d1=0.52, feather=1.2, core=0.5, dry=0.80, dry_k=3.0,
             dry_ph=rng.uniform(0, TAU))
        # 绑扎 — the tie where upright meets bar
        disc(cv, px, y0, w * 1.4, I("dan"), 0.16, squash=0.6, soft=0.7)


def dew(cv, rng, x, y, r, air):
    """露 — a drop. Under a multiply composite ink can only darken, so the
    bright side of a drop has to be *lifted* instead: a pale disc at low
    alpha, with a dark meniscus under it. Rare and small; a sheet of them
    looks like rain damage."""
    disc(cv, x, y, r, (252, 250, 243), 0.30, squash=0.92, soft=0.7)
    disc(cv, x + r * 0.18, y + r * 0.38, r * 0.62, I("dan"), 0.22,
         squash=0.55, soft=0.8)


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

TITLE_A = ["葡萄", "墨葡萄", "明珠", "珠光", "野藤", "秋实", "紫玉", "累累",
           "藤阴", "架上", "秋藤", "垂珠", "草龙", "珠圆", "三友"]
TITLE_B = ["图", "小景", "清趣", "写意", "图卷", "册页", "小品"]
ACTION = ["写生", "戏作", "写意", "拟古", "画藤", "灯下戏墨", "南窗写葡萄",
          "闲抛闲掷", "并记"]
SEAL_WORDS = ["明珠", "草龙", "墨缘", "写心", "秋实", "野藤", "珠圆", "一串",
              "闲抛", "笔底", "藤香", "老圃", "安之"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into grape_font.py."""
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
    for mod in ("grape_font", "orchid_font", "bamboo_font", "gongbi_font",
                "shanshui_font"):
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

    Same machinery as the other five painters. A grape sheet needs it in a
    particular way: the subject hangs from the top edge and leaves the
    bottom of the paper empty, so the column of writing is usually what
    stops the painting from sliding off the top.
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
    for fy in (0.30, 0.48, 0.66):
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

def _nodes(rng, pts, n, lo=0.06, hi=0.95):
    """n points along a path, spaced unevenly (疏密), with the tangent at
    each — a vine puts out its leaf, its bunch and its tendril from the same
    node, so everything on the sheet is hung off these."""
    out = []
    m = len(pts) - 1
    if m < 2:
        return out
    for i in range(n):
        t = lo + (hi - lo) * (i + rng.uniform(0.15, 0.85)) / n
        j = max(1, min(m, int(t * m)))
        a = math.atan2(pts[j][1] - pts[j - 1][1], pts[j][0] - pts[j - 1][0])
        out.append((pts[j][0], pts[j][1], a, t))
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

    # 虚实 — one half of the sheet carries the vine, the other the writing,
    # and the wind blows into the empty half. Blowing the other way it lays
    # a subject that is already against its border over the border, and the
    # sheet keeps the emptiness and loses the picture.
    side = rng.choice((0, 1))
    wind = 1.0 if side else -1.0
    lean = air["sweep"] * wind * rng.uniform(0.85, 1.15)

    if air.get("veil"):
        veil(cv, air["veil"][0], air["veil"][1])
    if air.get("moon"):
        mr = S * rng.uniform(0.09, 0.14)
        mx = W * (rng.uniform(0.14, 0.32) if side else rng.uniform(0.68, 0.86))
        my = H * rng.uniform(0.12, 0.26)
        moon(cv, rng, mx, my, mr, dilute(I("dan"), 0.75))
        occ.add(mx, my, mr * 1.1)

    if sc.get("trellis"):
        trellis(cv, rng, W, H, st, air, side)

    # 主藤 — in at the top, because 葡萄 is a 折枝 subject: the picture is a
    # cut length of vine hung over the top edge of the paper, and anything
    # that looks rooted in the bottom of the sheet is a different plant.
    # A hanging vine is measured down the scroll, not across it. Taking the
    # length from the short side alone put every subject in the top half of
    # a tall sheet and left the bottom two-fifths dead — the vine simply did
    # not have the material to reach down there, and the fruit hangs off the
    # vine's lower nodes, so the fruit could not either. On a square sheet
    # this expression is S, which is what it always was.
    L = (H * 0.62 + S * 0.38) * rng.uniform(*sc["Lf"])
    pole = sc["entry"] == "pole"
    if pole:
        turn, cone, drift, deep = rng.uniform(1.4, 2.4), 1.15, 0.85, 0.52
    else:
        turn, cone, drift, deep = rng.uniform(1.4, 2.6), 0.82, lean * 0.4, 0.62

    # 势要贯 — the gesture has to cross the paper. A vine that wanders out
    # of a border in its first third leaves a sheet with a twig in one
    # corner and nothing else on it, which happened to two sheets in six
    # before this loop existed. Draw the line a few times and keep the
    # best one; the cone above makes that cheap, this makes it certain.
    #
    # "Deepest" was the whole of the score for two versions, and depth
    # alone buys the wrong line. A vine that enters at four-fifths of the
    # width and drops straight down an inch from the right border scores
    # perfectly on depth and is the worst composition the module can make:
    # the subject is pinned to the frame, the leaves are cut off by it,
    # and three-fifths of the sheet is not 留白 but a hole. 留白 is空
    # that the gesture *shapes*; a margin the gesture never visits is just
    # unpainted paper. So the score is depth, plus credit for the width
    # the line actually traverses, less a penalty for the share of its
    # length spent in the outer eighth — and the entry is re-rolled every
    # throw, because a border-hugger is usually decided at the entry and
    # no amount of re-drawing from the same point will cure it.
    def _score(p):
        n = float(len(p))
        dep = p[-1][1] / H
        hug = sum(1 for q in p if q[0] < W * 0.12 or q[0] > W * 0.88) / n
        span = (max(q[0] for q in p) - min(q[0] for q in p)) / W
        # 偏而不倚 — a 折枝 line belongs about a third in, never dead centre
        # and never jammed against the frame, so the penalty is on the
        # *excess* beyond a comfortable offset and is zero inside it.
        off = max(0.0, abs(sum(q[0] for q in p) / n / W - 0.5) - 0.22)
        return dep + 0.75 * min(span, 0.40) - 0.85 * hug - 1.6 * off

    # All seven are drawn and the best is kept. Breaking out at the first
    # throw that cleared a threshold was the same mistake as scoring on
    # depth alone, one level up: nearly every throw clears a threshold on
    # depth, so the loop was taking the first line and the other six
    # existed only on paper. Seven integrations of a path cost nothing
    # next to one berry.
    best, bs = None, None
    for _ in range(7):
        if pole:
            a0 = (0.10 if side else math.pi - 0.10) + rng.uniform(-0.25, 0.25)
            ex = W * (0.06 if side else 0.94)
            ey = H * rng.uniform(0.15, 0.24)
        else:
            a0 = math.pi * 0.5 + rng.uniform(-0.42, 0.42) - lean * 0.35
            ex = W * (rng.uniform(0.52, 0.78) if side
                      else rng.uniform(0.22, 0.48))
            ey = -H * rng.uniform(0.01, 0.05)
        p = vine_path(rng, ex, ey, a0, L, turn=turn,
                      waves=rng.uniform(1.8, 3.0), drift=drift,
                      bounds=(W, H), cone=cone)
        # a line that fell out of the paper in its first third is not a
        # candidate at all, however the rest of it scores
        s = _score(p) - (0.0 if len(p) > 84 or p[-1][1] > H * deep else 0.60)
        if bs is None or s > bs:
            best, bs = p, s
    spine = best
    w0 = S * rng.uniform(0.0090, 0.0135)

    # the branches come off nodes of the main vine and carry their own nodes
    nodes = _nodes(rng, spine, rng.randint(*sc["nodes"]))
    branches = []
    for _ in range(rng.randint(*sc["branch"])):
        if not nodes:
            break
        bx, by, ba, bt = nodes[rng.randrange(len(nodes))]
        # Fork *into* the sheet. Picking the side by coin toss sent half the
        # branches straight at the nearest border, and a branch that leaves
        # the paper at a shallow angle draws a long flat line across the
        # whole width on its way out — which is the one shape a vine must
        # never make.
        d = 1.0 if bx < W * 0.5 else -1.0
        if rng.random() < 0.25:
            d = -d
        a1 = ba + d * rng.uniform(0.55, 1.25) + lean * 0.3
        # and fork *downward*: the fruit hangs off these.
        a1 = math.atan2(math.sin(a1) + rng.uniform(0.0, 0.45), math.cos(a1))
        bl = L * rng.uniform(0.22, 0.44)
        # and it has to *bend*. At turn≈1.5 over a third of the sheet a
        # branch is a ruler line with a vine's width, and the eye finds a
        # straight line in a painting of a vine faster than it finds
        # anything else on the sheet.
        bp = vine_path(rng, bx, by, a1, bl, n=56,
                       turn=rng.uniform(1.8, 3.2),
                       waves=rng.uniform(1.4, 2.4), drift=lean * 0.6,
                       bounds=(W, H), cone=1.05)
        branches.append((bp, w0 * (0.44 + 0.26 * bt)))
        nodes += _nodes(rng, bp, rng.randint(1, 2), lo=0.3, hi=0.97)

    # 先定果位 — where the fruit will hang is decided *before* a single leaf
    # goes down, even though the fruit is painted last.
    #
    # The order on the sheet and the order of the decisions are not the same
    # order, and conflating them cost two rounds of bad sheets: painting the
    # leaves first and letting the bunch fall where it liked put a big dark
    # leaf squarely behind the fruit about one sheet in four, and a bunch
    # read against a leaf of its own value is a bunch nobody can see. 墨叶衬
    # 珠 means the leaf is the *foil*, which it can only be from beside or
    # behind-and-above, never from directly under.
    nb = rng.randint(*sc["bunches"])
    plan = []
    cand = sorted(nodes, key=lambda p: -p[1])     # the lower nodes first
    big = rng.uniform(0.30, 0.40)
    for i in range(nb):
        pick = None
        for nd in cand:
            nx, ny = nd[0], nd[1]
            if ny < H * 0.06 or ny > H * 0.80:
                continue
            # 一聚一散 is a statement about distance as well as size: two
            # bunches a third of a bunch apart are one lumpy bunch.
            if any((nx - px2) ** 2 + (ny - py2) ** 2 < (pr2 * 2.1) ** 2
                   for px2, py2, pr2 in
                   [(p[0], p[1] + p[2] * 0.4, p[2] * 0.5) for p in plan]):
                continue
            # — and about height in particular. Two bunches level with each
            # other read as a pair of earrings whatever the gap between
            # them; 一聚一散 needs one of them to be the one that hangs.
            if any(abs(ny - p[1]) < H * 0.11 for p in plan):
                continue
            pick = nd
            break
        if pick is None:
            break
        nx, ny = pick[0], pick[1]
        cand = [c for c in cand if c is not pick]
        # 一聚一散 — if there are two, one is the subject and the other is
        # its echo, and they are never the same size.
        f = big if i == 0 else big * rng.uniform(0.52, 0.74)
        # the bunch has to finish on the paper: its tip berry reaches a
        # little past `bl` and the fruit is the one thing never cropped
        bl = min(S * f, (H - ny) * 0.74)
        if bl < S * 0.12:
            continue
        bw = bl * rng.uniform(0.52, 0.70)
        bx = nx + rng.uniform(-0.1, 0.1) * bw
        # 不出纸 — a bunch half off the border is a bunch nobody painted.
        # The vine may run off the paper; the fruit may not.
        # half the width plus a berry: the shoulder berries sit at
        # ±halfw and have a radius of their own beyond that
        bx = min(max(bx, W * 0.035 + bw * 0.80), W * 0.965 - bw * 0.80)
        by = ny + bl * rng.uniform(0.10, 0.20)
        plan.append((bx, by, bl, bw, nx, ny))

    # 远叶 — the leaves that go behind the vine. Painted first and painted
    # pale; they are what gives the tangle a depth it otherwise has to fake.
    nl = rng.randint(*sc["leaves"])
    rng.shuffle(nodes)
    # 叶就果 — the leaves gather where the fruit is. Picking their nodes at
    # random spread them evenly down the vine, which on a tall sheet put
    # every leaf in the top third and left the bunch hanging alone at the
    # end of a long bare line: a lollipop. The foil has to be next to the
    # thing it sets off. The jitter is what keeps a leaf or two out in the
    # open, which is also wanted — but as the exception.
    if plan:
        def near_fruit(nd):
            d = min(math.hypot(nd[0] - p[0], nd[1] - p[1] - p[2] * 0.3)
                    for p in plan)
            return d * rng.uniform(0.55, 1.9)
        nodes.sort(key=near_fruit)
    leafy = nodes[:nl]
    # the pale 远叶 are the ones *furthest* from the fruit, since the sort
    # above put the nearest first and it is the dark near leaf that has to
    # be the one beside the bunch
    nback = max(1, int(nl * rng.uniform(0.25, 0.42)))
    nfront = max(0, len(leafy) - nback)
    marks = []
    # the footprint the fruit has booked: a leaf centre inside one of these
    # is refused the same way a leaf on top of another leaf is
    booked = [(p[0], p[1] + p[2] * 0.42, max(p[2] * 0.46, p[3] * 0.62))
              for p in plan]
    for i, (nx, ny, na_, nt) in enumerate(leafy):
        back = 0.0 if i < nfront else 1.0
        if back:
            lL = S * rng.uniform(0.12, 0.19) * (1.0 - 0.18 * nt)
            a = na_ + rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 1.5) \
                + lean * 0.4
            px, py = (nx + math.cos(a) * lL * 0.18,
                      ny + math.sin(a) * lL * 0.18)
            drag(cv, _arc([(nx, ny), ((nx + px) * 0.5, (ny + py) * 0.5),
                           (px, py)], 6),
                 max(0.8, lL * 0.022), max(0.6, lL * 0.015),
                 dilute(rgb(st["vine"][0]), 0.7), rgb(st["vine"][1]),
                 0.5 * k, d0=0.4, d1=0.7)
            marks.append(leaf(cv, rng, px, py, a, lL, st, air,
                              back=rng.uniform(0.55, 0.85), k=k))

    for p, w in branches:
        vine(cv, rng, p, w, w * 0.42, st, air, alpha=0.93 * k,
             knots=rng.uniform(2.0, 4.0), sinew=False)
        occ.add_path(p, w * 1.4, w * 0.6, step=6)
    vine(cv, rng, spine, w0, w0 * 0.40, st, air, alpha=0.96 * k,
         knots=rng.uniform(2.0, 4.0))
    occ.add_path(spine, w0 * 1.5, w0 * 0.6, step=6)

    # 近叶 — and these go over it, with their petioles starting *on* the
    # vine, which is the only way a leaf ever looks attached.
    for i, (nx, ny, na_, nt) in enumerate(leafy):
        if i >= nfront:
            continue
        # 叶大于果 — in a grape painting the leaf is the big shape and the
        # bunch is the small dense one it is shown against. Reversed, the
        # sheet reads as a fruit stall.
        lL = S * rng.uniform(0.16, 0.25) * (1.0 - 0.16 * nt)
        # alternate sides down the vine, lean with the air, droop with
        # gravity: a leaf hangs, it does not point.
        a = na_ + (1.0 if i % 2 else -1.0) * rng.uniform(0.75, 1.45) \
            + lean * 0.5
        a = math.atan2(math.sin(a) + rng.uniform(0.10, 0.45),
                       math.cos(a))
        pl = lL * rng.uniform(0.18, 0.34)
        px, py = nx + math.cos(a) * pl, ny + math.sin(a) * pl
        if not (-lL * 0.4 < px < W + lL * 0.4 and -lL * 0.4 < py < H + lL * 0.4):
            continue
        # 疏密 — refuse a leaf that would land on top of another one. Three
        # tries, then give the space up; a 写意 sheet is allowed to be bare
        # and is not allowed to be a thicket.
        for _ in range(6):
            if all((px - mx) ** 2 + (py - my) ** 2 > (mr * 0.62) ** 2
                   for mx, my, mr in marks) \
                    and all((px - bx2) ** 2 + (py - by2) ** 2
                            > (br2 * 0.82 + lL * 0.12) ** 2
                            for bx2, by2, br2 in booked):
                break
            # Turn, and also *reach*: a refused leaf is usually refused
            # because the node it grows from is crowded, and a longer
            # petiole takes the blade out of the crowd without detaching it
            # from the vine. Six tries, turning and reaching — three tries
            # that could only turn threw most of the leaves away, and once
            # the leaves were told to gather near the fruit the two rules
            # fought each other and the sheets came out nearly bare.
            a += rng.choice((-1.0, 1.0)) * rng.uniform(0.25, 0.6)
            pl = min(pl * rng.uniform(1.06, 1.22), lL * 0.46)
            px, py = nx + math.cos(a) * pl, ny + math.sin(a) * pl
        else:
            continue
        # 叶柄 stops at the basal cleft, not at the centre of the blade.
        # Run to the centre and it prints straight across the leaf — the
        # wash above it is 淡 and ink multiplies, so a petiole under a leaf
        # is a petiole *through* it, a ruled stick over the one shape on
        # the sheet that must not have a ruled line on it. The blade's own
        # radius at the cleft is about a tenth of its length.
        ex2 = px - math.cos(a) * lL * 0.10
        ey2 = py - math.sin(a) * lL * 0.10
        drag(cv, _arc([(nx, ny),
                       ((nx + ex2) * 0.5 + rng.uniform(-pl, pl) * 0.3,
                        (ny + ey2) * 0.5 + rng.uniform(-pl, pl) * 0.3),
                       (ex2, ey2)], 8),
             max(0.9, lL * 0.022), max(0.6, lL * 0.014),
             dilute(rgb(st["vine"][0]), 0.85), rgb(st["vine"][1]),
             0.80 * k, d0=0.5, d1=0.9, core=0.3)
        marks.append(leaf(cv, rng, px, py, a, lL, st, air,
                          back=rng.uniform(0.0, 0.22), k=k))

    for mx, my, mr in marks:
        occ.add(mx, my, mr * 0.78)

    # 串 — the fruit, last of the solids and on top of the leaves, because
    # 墨叶衬珠: the leaf is the ground the pearls are shown against.
    for bx, by, bl, bw, nx, ny in plan:
        got = bunch(cv, rng, bx, by, bl, bw, st, air, lean=lean * 0.5, k=k,
                    root=(nx, ny))
        for gx, gy, gr in got:
            occ.add(gx, gy, gr * 1.1)
        if air.get("dew") and got:
            # `bunch` returns its fruit back to front, and a drop sits on
            # the berry nearest the eye — on one at the back it would be a
            # bright speck under three other berries.
            for _ in range(rng.randint(1, 3)):
                gx, gy, gr = got[rng.randrange((len(got) * 2) // 3,
                                               len(got))]
                hx2, hy2 = gb.LIGHT
                dew(cv, rng, gx + hx2 * gr * 0.4, gy + hy2 * gr * 0.4,
                    gr * rng.uniform(0.16, 0.24), air)
        if stat is not None:
            stat.setdefault("berries", []).append(len(got))

    # 卷须点睛 — the tendrils go on last and they go on over everything.
    #
    # ...which is exactly why the direction cannot be a free die roll. A
    # tendril is the one mark on the sheet that is pure line, and drawn
    # across a bunch it stops being a tendril: two turns of a tight spiral
    # stamped on the front berry read as a watermark, not as a plant. A
    # tendril reaches into air — that is its whole job, it is looking for
    # something to hold — so it is thrown several times and the throw that
    # lands in the emptiest place is the one that gets painted.
    for _ in range(rng.randint(*sc["curls"])):
        if not nodes:
            break
        best = None
        for _try in range(7):
            nx, ny, na_, nt = nodes[rng.randrange(len(nodes))]
            a = na_ + rng.choice((-1.0, 1.0)) * rng.uniform(0.35, 1.15) \
                + lean * 0.8
            tL = S * rng.uniform(0.07, 0.14)
            # where the curl itself will sit: out along the throw, not at
            # the node, because the node is on the vine and the vine is
            # painted matter by definition.
            ex, ey = nx + math.cos(a) * tL, ny + math.sin(a) * tL
            pen = occ.overlap(ex - tL * 0.5, ey - tL * 0.5,
                              ex + tL * 0.5, ey + tL * 0.5)
            if ex < 0 or ey < 0 or ex > W or ey > H:
                pen += 1e6
            if best is None or pen < best[0]:
                best = (pen, nx, ny, a, tL)
            if pen == 0.0:
                break
        _, nx, ny, a, tL = best
        tendril(cv, rng, nx, ny, a, tL, st, air,
                hand=rng.choice((-1.0, 1.0)), k=k)

    # 风 in this painter is carried entirely by the sweep of the vine and
    # the lean of the bunches. A torn-loose leaf dropped on bare paper was
    # tried here and read as a dead insect: 落叶 is 竹 and 芦 territory,
    # where the stroke that draws the leaf is the same stroke either way.
    # A grape leaf is a wash with a shape, and a wash with nothing holding
    # it has no reason to be in the air.

    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))]) \
        if rng.random() < 0.74 \
        else (air["cn"] + TITLE_B[rng.randrange(len(TITLE_B))])
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
        description="Generate random Chinese ink-grape paintings (墨葡萄).")
    ap.add_argument("-o", "--out", default="grape.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="垂藤|双串|架下|野藤")
    ap.add_argument("--air", help="静|秋风|带露|月下")
    ap.add_argument("--style", help="墨葡萄|紫玉|水晶|玛瑙")
    ap.add_argument("--size", default="900x1300", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for k, v in SCENES.items():
            print("  %-6s %s" % (k, v["en"]))
        print("风致 (air):")
        for k, v in AIR.items():
            print("  %-5s %-6s %s" % (k, v["cn"], v["en"]))
        print("设色 (colour):")
        for k, v in STYLES.items():
            print("  %-8s %s" % (k, v["en"]))
        print("果 (the berry): 果如明珠, 留白为光, 前实后虚 — "
              "忌排如算盘珠")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("墨葡萄 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
