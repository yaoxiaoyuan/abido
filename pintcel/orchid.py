#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchid.py — 随机生成中国墨兰图 / Random Chinese orchid paintings (墨兰).

兰 is the second of the 四君子 and the least forgiving of them, because the
whole picture is a dozen strokes and every one is naked: there is no 勾线 to
correct a bad edge and no wash to bury a bad curve. The tradition is
correspondingly explicit about what a leaf *is*, and this generator is built
as those rules — the way bamboo.py is built as 叶诀:

    一笔三折   one stroke, three turns. The blade leaves the root, arcs over,
               and the tip runs away; the hand never stops in the middle.
    螳螂肚     "mantis belly" — thin at the root, swelling once through the
               middle, then drawn out to a thread (鼠尾, the rat's tail).
    折叶       where a blade twists it is seen edge-on: the width pinches to
               a third and the ink darkens, because the brush turned there.
    凤眼       two blades crossing leave an eye-shaped piece of paper. That
               piece of paper is the reason for the crossing.
    忌         no two blades at one angle, no two of a length, no three
               crossing in one place, nothing crossing square (井字).
    花在叶间   the flowers go in the gaps between the leaves, never on top
               of them — so the gaps are measured before a spike is placed.
    点心如品   the heart is three dots, written like the character 品.

The structural idea carried over from gongbi.py is that **a shape is a field,
not a polygon**; from shanshui.py, that **the primitive should be the thing
the painter actually practises**. For 兰 that thing is one long swept stroke
of varying width, and gongbi's `Blade` cannot be it: a Blade bends as
`u**2` in one direction only, and a 兰叶 reverses. So this module owns one
renderer of its own — `sweep`, a brush dragged along an arbitrary spine —
and every mark on the sheet, leaf, petal, lip, stalk, bract and root, is one
call to it.

Everything else (PNG writer, paper ground, value noise, baked calligraphy
outlines, seals, the occupancy map) is shared with gongbi.py, and the ink
table, the 兑水 rule and the 折带 rock are shared with bamboo.py — the two
are the same family of 写意 subject and should not disagree about what 淡墨
is.

Usage
-----
    python3 orchid.py                             # one random painting
    python3 orchid.py -n 6                        # six of them
    python3 orchid.py --scene 兰石 --air 临风
    python3 orchid.py --style 青兰 --seed 7
    python3 orchid.py --size 900x1300 -o scroll.png
    python3 orchid.py --list                      # the repertoire
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, Noise, Occupancy, BrushFont, clamp01,
                    disc, fill_contours, jitter, mix, rgb, brush_path,
                    brush_seg, paint_column, paint_seal, TIANGAN, DIZHI)
# 一家之墨 — the ink table, the 兑水 rule, the ink-plane tones, the full-sheet
# veil and the 留白 moon are bamboo.py's, not copies of them. 兰 and 竹 are
# painted with the same stone and the same water, and two modules of this
# family disagreeing about what 淡墨 is would show at once in any sheet that
# carried both. The 折带 rock is *not* borrowed: see `stone`.
from bamboo import I, dilute, plane_tone, veil, moon


# --------------------------------------------------------------------------
# 设色 — the four schools. 兰 is an ink subject first: even the coloured
# versions keep the leaves near ink and the flowers barely tinted, because a
# saturated orchid stops being 写意 and becomes a botanical plate.
# --------------------------------------------------------------------------

STYLES = {
    "墨兰": dict(cn="墨兰", en="plain ink",
                 leaf=("#7E7668", "#2C2825"), flower=("#A9A396", "#574F45"),
                 heart="#1C1917", rock=("#B6B0A0", "#3A352E")),
    "素心": dict(cn="素心", en="ink leaves, gamboge heart",
                 leaf=("#847C6E", "#332F2A"), flower=("#C9C4AC", "#8B8870"),
                 heart="#9A6C12", rock=("#B8B2A2", "#3C3730")),
    "青兰": dict(cn="青兰", en="malachite and sap green",
                 leaf=("#93A878", "#33543C"), flower=("#C6CFB2", "#7B8F66"),
                 heart="#8E6A12", rock=("#B2B6A4", "#3C443A")),
    "紫蕊": dict(cn="紫蕊", en="ink leaves, violet flowers",
                 leaf=("#8A8878", "#2F332B"), flower=("#C7B6C6", "#7A5C84"),
                 heart="#45182F", rock=("#B6AEA6", "#3A3430")),
}

# 章法 — the four framings. `Lf` is the length of the longest blade as a
# fraction of the sheet's short side, and it is the single number that
# decides whether a sheet reads as 兰: below about a third the clump is a
# tuft of grass sitting on the floor, above about three quarters every blade
# runs off the paper and the arc — the whole subject — is lost.
SCENES = {
    "幽兰": dict(cn="幽兰", en="one clump on bare paper",
                 clumps=1, nl=(8, 11), spikes=(1, 2), rock=0.0, bank=0.55,
                 Lf=(0.78, 0.98)),
    "双丛": dict(cn="双丛", en="two clumps, 一聚一散",
                 clumps=2, nl=(6, 9), spikes=(1, 2), rock=0.0, bank=0.85,
                 Lf=(0.60, 0.78)),
    "兰石": dict(cn="兰石", en="orchid and rock",
                 clumps=1, nl=(8, 12), spikes=(1, 2), rock=1.0, bank=0.25,
                 Lf=(0.68, 0.88)),
    "崖兰": dict(cn="崖兰", en="hanging from a cliff",
                 clumps=1, nl=(7, 10), spikes=(1, 1), rock=0.0, ledge=1.0,
                 hang=True, Lf=(0.72, 0.92)),
}

# 风致 — one air for the whole sheet, the way bamboo.py has one weather.
# `wet` scales the softness of every edge (兰 is painted wetter than 竹 even
# in fair air), `dry` how much 飞白 the long blades show, `sweep` the shared
# lean of everything that grows.
AIR = {
    "静": dict(cn="静兰", en="still air",
               sweep=0.00, wet=1.00, dry=0.32, k=1.00),
    "临风": dict(cn="风兰", en="in the wind",
                 sweep=0.30, wet=0.88, dry=0.46, k=1.00, fly=(1, 2)),
    "带露": dict(cn="露兰", en="dew still on the leaves",
                 sweep=0.05, wet=2.10, dry=0.04, k=0.94, pool=1.0),
    "月下": dict(cn="月兰", en="by moonlight",
                 sweep=0.12, wet=1.35, dry=0.16, k=0.86,
                 veil=("#6E7A86", 0.10), moon=1.0),
}


# --------------------------------------------------------------------------
# 笔 — one brush, dragged along a spine.
#
# The mark is a field, like everything in gongbi.py: coverage and ink density
# are functions of (distance along, distance across) and the edge comes out
# anti-aliased without supersampling. What is different here is that the
# spine is an arbitrary polyline, because a 兰叶 turns through two radians
# and reverses at the tip, and because that is what lets one function draw
# every mark in the painting.
#
# The accumulation is by *maximum*, into a sparse map, and then composited
# once. That matters: `brush_path` composites each of its segments on its own
# and so lays ink down twice at every joint — on a stroke sampled every four
# pixels that is a bead every four pixels, which is exactly what the 折带
# shelves in bamboo.py had to stop doing. A stroke is one load of ink and can
# only darken the paper once, however finely it is sampled.
# --------------------------------------------------------------------------

def sweep(cv, pts, hws, dens, lo, hi, alpha=1.0, feather=0.9, core=0.30,
          side=0.0, dry=0.0, dry_k=2.4, dry_ph=0.0):
    """Drag the brush along `pts`, half-width `hws`, ink density `dens`.

    `lo`/`hi` are the two stops of the stroke's own ramp: density 0 is the
    pale flank, 1 the loaded core. `core` pulls the rim back toward 淡 (a
    round brush carries its load in the middle), `side` tilts the ink onto
    one flank for a 侧锋 stroke, and `dry` breaks the coverage into streaks
    toward the tip — 飞白, the paper showing through a brush that has run
    out of water.
    """
    n = len(pts)
    if n < 2:
        return
    acc = {}
    cw, ch = cv.w, cv.h
    inv_n = 1.0 / (n - 1)
    for i in range(n - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        h0, h1 = hws[i], hws[i + 1]
        d0, d1 = dens[i], dens[i + 1]
        dx, dy = x1 - x0, y1 - y0
        ll = dx * dx + dy * dy
        if ll < 1e-9:
            continue
        inv = 1.0 / ll
        seg = math.sqrt(ll)
        hm = (h0 if h0 > h1 else h1) + feather + 1.0
        ax0 = int((x0 if x0 < x1 else x1) - hm)
        ax1 = int((x0 if x0 > x1 else x1) + hm) + 1
        ay0 = int((y0 if y0 < y1 else y1) - hm)
        ay1 = int((y0 if y0 > y1 else y1) + hm) + 1
        if ax0 < 0:
            ax0 = 0
        if ay0 < 0:
            ay0 = 0
        if ax1 > cw:
            ax1 = cw
        if ay1 > ch:
            ay1 = ch
        for py in range(ay0, ay1):
            fy = py + 0.5 - y0
            row = py * cw
            for px in range(ax0, ax1):
                fx = px + 0.5 - x0
                t = (fx * dx + fy * dy) * inv
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                ex = fx - dx * t
                ey = fy - dy * t
                d = math.sqrt(ex * ex + ey * ey)
                hw = h0 + (h1 - h0) * t
                cov = (hw - d) / feather + 0.5
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                # signed position across the stroke, in units of its own
                # half-width: the 侧锋 shading and the 飞白 streaks both need
                # a side, not just a distance
                sl = (ex * dy - ey * dx) / (seg * (hw if hw > 0.35 else 0.35))
                if sl > 1.0:
                    sl = 1.0
                elif sl < -1.0:
                    sl = -1.0
                lat = sl if sl >= 0.0 else -sl
                sh = (d0 + (d1 - d0) * t) * (1.0 - core * lat * lat) \
                    + side * sl
                if dry:
                    tg = (i + t) * inv_n
                    q = dry * tg * tg
                    g = math.sin(sl * dry_k * math.pi + dry_ph)
                    cov *= 1.0 - q * g * g
                    if cov <= 0.0:
                        continue
                k = row + px
                v = acc.get(k)
                if v is None:
                    acc[k] = [cov, sh]
                else:
                    if cov > v[0]:
                        v[0] = cov
                    if sh > v[1]:
                        v[1] = sh
    # 一次落纸 — composite the whole stroke in one pass, multiplying into the
    # paper, so the ground's tone and grain stay visible through the ink.
    buf = cv.buf
    ramp = [mix(lo, hi, j / 24.0) for j in range(25)]
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
        buf[i] = int(o + (o * c[0] * 0.00392 - o) * a)
        o = buf[i + 1]
        buf[i + 1] = int(o + (o * c[1] * 0.00392 - o) * a)
        o = buf[i + 2]
        buf[i + 2] = int(o + (o * c[2] * 0.00392 - o) * a)


def stroke(cv, rng, x, y, a0, L, W, lo, hi, alpha=1.0, turn=0.0, rev=0.0,
           rev_at=0.68, belly=0.32, taper=1.5, root=0.34, folds=(),
           dens=1.0, fade=0.40, dry=0.0, feather=0.9, core=0.30, side=0.0,
           gap=None, step=None, wob=0.0035, pool=0.0, bow=1.22):
    """一笔 — build the spine and the two profiles, then lay the ink down.

    Returns the spine, which the composition needs: the 疏密 of a clump, the
    placing of the flowers in the gaps and the 凤眼 test are all measured off
    these points, never guessed from the parameters that made them.

    The turning is `turn * t**bow`, i.e. the curvature *increases* along the
    stroke, and `bow` is the most important number in the file. With a
    constant curvature (bow = 1) every blade is a circular arc, and the
    first render of this module showed exactly what a clump of circular arcs
    does: the blades peel over within the first third of their length and the
    whole plant lies down on the ground like a pressed specimen, every tip at
    the same height, no reach in it at all. 兰叶 go *up* first. At bow ≈ 1.7
    the first half of the blade is nearly straight — it is holding its own
    weight — and the turn happens out at the end where the lever is long.
    That one exponent is the difference between an orchid and a fountain.
    """
    st = step or max(2.2, L * 0.013)
    n = max(6, int(L / st) + 1)
    sgn = 1.0 if turn >= 0.0 else -1.0
    inv = 1.0 / (n - 1)
    seglen = L * inv
    pts, hws, dns = [], [], []
    cx, cy, drift, ripple = x, y, 0.0, 0.0
    for i in range(n):
        t = i * inv
        # --- 螳螂肚 then 鼠尾 ------------------------------------------
        if t <= belly and belly > 1e-6:
            f = root + (1.0 - root) * (t / belly) ** 0.62
        else:
            z = (t - belly) / max(1e-6, 1.0 - belly)
            f = (1.0 - z) ** taper
            if f < 0.0:
                f = 0.0
        # 折叶 — where the blade twists it is seen edge-on and nearly
        # vanishes. Narrow *and* darker: the pinch alone reads as a flaw in
        # the stroke, the pinch with ink in it reads as a turn.
        kd = 1.0
        for ft, fa, fw in folds:
            g = math.exp(-((t - ft) / fw) ** 2)
            f *= 1.0 - fa * g
            kd += 0.22 * g
        hws.append(W * f)
        # --- 墨色 along the stroke ------------------------------------
        # Heaviest where the brush landed, running out toward the tip; plus
        # the 起笔 press at the very root and the ink the brush leaves in a
        # pool where it stops moving (带露 air only).
        d = dens * kd * (1.0 - fade * t ** 1.15)
        if t < 0.06:
            d *= 1.0 + 0.30 * (1.0 - t / 0.06)
        if pool and t > 0.90:
            d *= 1.0 + pool * (t - 0.90) * 4.0
        # 墨不匀 — a slow random walk on top of it. The clean profile alone
        # gives a blade that is evenly dark all the way down, and a wide
        # evenly dark blade reads as rubber, not as a loaded brush being
        # dragged over paper.
        ripple += rng.gauss(0.0, 0.055)
        if ripple > 0.16:
            ripple = 0.16
        elif ripple < -0.16:
            ripple = -0.16
        dns.append(d * (1.0 + ripple))
        pts.append((cx, cy))
        a = a0 + turn * (t ** bow) + drift
        if rev and t > rev_at:
            a -= sgn * rev * ((t - rev_at) / (1.0 - rev_at)) ** 1.6
        cx += math.cos(a) * seglen
        cy += math.sin(a) * seglen
        if wob:
            drift += rng.gauss(0.0, wob)

    # 断叶 — a blade crossed by something in front of it, or simply folded
    # away from the light, is drawn in two pieces with paper between them.
    # Both new ends have to be closed to a point: a flat-ended piece of leaf
    # is a cut, and a cut is the one thing that cannot be explained as ink.
    if gap:
        i0 = max(2, int(gap[0] * (n - 1)))
        i1 = min(n - 3, int(gap[1] * (n - 1)))
        runs = [(0, i0), (i1, n - 1)] if i1 > i0 + 1 else [(0, n - 1)]
    else:
        runs = [(0, n - 1)]
    for lo_i, hi_i in runs:
        if hi_i - lo_i < 2:
            continue
        p = pts[lo_i:hi_i + 1]
        h = hws[lo_i:hi_i + 1]
        d = dns[lo_i:hi_i + 1]
        if lo_i > 0:
            h = list(h)
            h[0] *= 0.18
            h[1] *= 0.62
        if hi_i < n - 1:
            h = list(h)
            h[-1] *= 0.18
            h[-2] *= 0.62
        sweep(cv, p, h, d, lo, hi, alpha, feather=feather, core=core,
              side=side, dry=dry, dry_k=rng.uniform(1.8, 3.2),
              dry_ph=rng.uniform(0.0, 6.28))
    return pts


# --------------------------------------------------------------------------
# 布叶 — the clump. This is the whole painting; the flowers are its accent.
# --------------------------------------------------------------------------

def leaf_plan(rng, n, bias=1.0, hang=False):
    """The skeleton of one 丛 — one entry per blade, before any ink.

    Three decisions, in this order, and the order is the composition:

    *分左右* — the blades are split between the two sides unequally. An even
    split is a fountain; 兰 is read as a crowd leaning one way with two or
    three blades answering back.

    *长短参差* — the lengths come from a descending palette, not from a
    uniform draw, so there is exactly one 主叶 and nothing that ties with it.
    Two blades of one length on one side is the 忌 the old books call
    蜻蜓 (a dragonfly's wings).

    *取势* — each blade is given the direction its *tip* is to point, and the
    turn is whatever gets it there from its launch angle. The obvious
    alternative — set the turn from the length, since a long blade bends
    further — is what the first draft did, and it fails in a way worth
    recording: the turn then being a function of the launch angle and the
    length, and both being drawn in order, every blade on a side came out as
    a scaled copy of its neighbour. The sheets were perfect croissants. Two
    nested arcs are 平行, which the manuals forbid before anything else, and
    no amount of jitter on a parallel construction makes it not parallel.
    Drawing the tip direction instead makes 挺 (a blade that stays stiff and
    keeps rising) and 垂 (one that falls past the horizontal) independent of
    how long the blade is, which is both true of real clumps and the thing
    that produces crossings: two blades from one root whose tips point
    different ways *must* cross, and the paper they enclose is the 凤眼.
    """
    maj = 1.0 if bias >= 0.0 else -1.0
    n_maj = max(2, int(round(n * rng.uniform(0.62, 0.76))))
    n_min = max(1, n - n_maj)
    out = []
    # 主客 — the two sides are not two halves. The lesser side gets fewer
    # blades, shorter ones, and a narrower fan: it answers the main side, it
    # does not balance it. Equal sides make the clump a shuttlecock, and no
    # 兰 in any album is symmetrical about its own root.
    for s, cnt, pal, fan in ((maj, n_maj, 1.00, 1.00),
                             (-maj, n_min, rng.uniform(0.70, 0.88), 0.74)):
        # 出叶宜紧 — the blades leave the root in a *tight* sheaf, within a
        # third of a radian of each other, and it is the turning that opens
        # the clump out as it rises. Splaying them at the root instead (the
        # fan ran to a full radian in the first drafts) is what makes a
        # bouquet: a wide launch puts every blade on its own radius from the
        # start, so none of them ever meets another and the clump reads as a
        # splayed shuttlecock with no crossings anywhere in it.
        if hang:
            p_lo, p_hi = 0.95, 0.95 + 0.70 * fan
        else:
            p_lo, p_hi = 0.04, 0.04 + 0.52 * fan
        # launch angles, stratified across the fan so no two can coincide
        ps = sorted(p_lo + (p_hi - p_lo) * (i + rng.uniform(0.2, 0.8)) / cnt
                    for i in range(cnt))
        lens = []
        f = pal
        for i in range(cnt):
            lens.append(f * rng.uniform(0.94, 1.0))
            # The decay has to be gentle. At 0.7 per blade — which is what
            # "clearly different lengths" suggests — a clump of seven comes
            # out as two long sweeps and a tuft of stubs round the root,
            # because the fifth blade is a quarter of the first. 长短参差
            # means no two alike, not a geometric series.
            f *= rng.uniform(0.80, 0.91)
        # ...with one blade deliberately kept short: the 护根叶 that covers
        # the base. Having it come out of the palette, rather than being the
        # tail of it, is what keeps the rest of the clump substantial.
        if cnt > 3:
            lens[-1] = rng.uniform(0.17, 0.28) * pal
        # The 主叶 is given a steep-ish launch from the inside of the fan,
        # never the outermost one: the outermost blade of a clump sweeps away
        # from the subject, and the longest blade sweeping away takes the
        # whole composition off the paper with it.
        idx = list(range(cnt))
        rng.shuffle(idx)
        home = cnt // 3
        idx.remove(home)
        idx.insert(0, home)
        aims = []
        for j, i in enumerate(idx):
            p, lf = ps[i], lens[j]
            # 取势 — where this blade's tip is to point, measured in world
            # angles on the right-hand side of the root (the left side is
            # mirrored when it is drawn). Negative is still rising.
            #
            # Each blade takes one of four characters, and a clump wants a
            # mixture: 挺 still climbing at the tip, 平 levelling out, 垂
            # falling away, 回 curling back under itself. Long blades can
            # afford any of them; short ones have not got the length to fall,
            # so they mostly climb — which is also why a clump reads as
            # having a *middle*.
            if hang:
                bands, wts = ((1.25, 1.75), (1.60, 2.20), (2.00, 2.60)), \
                    (0.34, 0.40, 0.26)
            else:
                bands = ((-0.95, -0.15), (-0.12, 0.45), (0.50, 1.30),
                         (1.35, 2.05))
                wts = ((0.42, 0.34, 0.20, 0.04) if lf < 0.45 else
                       (0.26, 0.26, 0.36, 0.12))
            r = rng.random()
            acc_w = 0.0
            blo, bhi = bands[-1]
            for b, wt in zip(bands, wts):
                acc_w += wt
                if r <= acc_w:
                    blo, bhi = b
                    break
            aim = rng.uniform(blo, bhi)
            # 忌平行 — and this is the whole of it: two blades whose tips
            # point the same way within a third of a radian are parallel
            # wherever they happen to sit, so the aim is redrawn until it
            # clears every aim already taken on this side.
            for _try in range(12):
                if all(abs(aim - a) > 0.34 for a in aims):
                    break
                aim = rng.uniform(blo, bhi)
            aims.append(aim)
            T = aim + math.pi * 0.5 - p
            # 浓淡 — which plane of ink this blade is painted in. The
            # spread matters more than the values: with everything between
            # 0.5 and 1.0 a clump comes out one even mid-grey mass, and the
            # 墨分五色 that makes a 写意 sheet breathe is gone. Two blades
            # carry 浓墨 and the rest fall well back.
            k = 1.0 if lf > 0.80 else (0.84 if lf > 0.62 else
                                       rng.uniform(0.32, 0.60))
            # 长叶挺, 短叶垂 — a long blade carries its turn out at the far
            # end (it is stiff enough to stand most of its length); a short
            # one peels over almost at once. Same total turn, different
            # distribution of it, and the pair of them is what gives a clump
            # its silhouette.
            out.append(dict(side=s, pitch=p, lf=lf, turn=T, k=k,
                            bow=rng.uniform(1.45, 1.85) if lf > 0.55
                            else rng.uniform(1.10, 1.45),
                            rev=(rng.uniform(0.22, 0.60)
                                 if (lf > 0.5 and rng.random() < 0.34)
                                 else 0.0),
                            broken=False))
    # 断叶 — at most one in a clump, and never the 主叶: the blade the eye
    # follows first cannot be the one with a piece missing.
    mids = [d for d in out if 0.35 < d["lf"] < 0.78]
    if mids and rng.random() < 0.38:
        mids[rng.randrange(len(mids))]["broken"] = True
    return out


def _cross(a, b, skip=0.16):
    """Where two spines cross, and at what angle — the 凤眼 instrument.

    The root region of both blades is skipped: every blade in a clump starts
    at the same point, so without that they all "cross" at the root and the
    measurement says nothing.
    """
    out = []
    ia = max(1, int(len(a) * skip))
    ib = max(1, int(len(b) * skip))
    for i in range(ia, len(a) - 1, 2):
        ax, ay = a[i]
        bx, by = a[i + 1]
        r1, r2 = bx - ax, by - ay
        for j in range(ib, len(b) - 1, 2):
            cx, cy = b[j]
            dx_, dy_ = b[j + 1][0] - cx, b[j + 1][1] - cy
            den = r1 * dy_ - r2 * dx_
            if abs(den) < 1e-9:
                continue
            t = ((cx - ax) * dy_ - (cy - ay) * dx_) / den
            u = ((cx - ax) * r2 - (cy - ay) * r1) / den
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                ang = abs(math.atan2(r1 * dy_ - r2 * dx_,
                                     r1 * dx_ + r2 * dy_))
                if ang > math.pi * 0.5:
                    ang = math.pi - ang
                out.append((ax + r1 * t, ay + r2 * t, ang))
    return out


def draw_leaf(cv, rng, x, y, spec, L, st, air, lean=0.0, alpha=0.95):
    """One 兰叶, from one entry of the plan."""
    ll = L * spec["lf"]
    k = spec["k"] * air["k"]
    lt, dp = plane_tone(st, "leaf", k)
    s = spec["side"]
    # 兰叶如凤眼 — the blade is one part in thirty of its own length at the
    # widest. Wider than that and it is an iris or a day-lily; much narrower
    # and the 螳螂肚 disappears and the clump reads as drawn with a pen.
    # 兰叶如凤眼 — one part in forty of its own length at the widest, give or
    # take a fifth. Two things were learnt by measuring this against the
    # sheets: without the random factor every long blade in a clump comes out
    # the same weight and the group reads as extruded, and at one part in
    # thirty — which is what the proportion looks like in an album leaf — two
    # dark blades overlapping stop being two blades and become a slab.
    W = (ll * rng.uniform(0.0098, 0.0136) * rng.uniform(0.84, 1.20)
         * (1.04 if k > 0.9 else 0.94))
    folds = []
    for _ in range(rng.randint(0, 2) if ll > L * 0.42 else 0):
        folds.append((rng.uniform(0.30, 0.86), rng.uniform(0.45, 0.72),
                      rng.uniform(0.028, 0.050)))
    gap = None
    if spec["broken"]:
        g0 = rng.uniform(0.34, 0.54)
        gap = (g0, g0 + rng.uniform(0.05, 0.10))
    return stroke(cv, rng, x, y,
                  -math.pi * 0.5 + s * spec["pitch"] + lean,
                  ll, W, lt, dp, alpha=alpha,
                  turn=s * spec["turn"], rev=spec["rev"],
                  belly=rng.uniform(0.26, 0.40), taper=rng.uniform(1.3, 1.8),
                  root=rng.uniform(0.26, 0.42), folds=folds,
                  dens=0.94, fade=rng.uniform(0.30, 0.52),
                  dry=air["dry"] * (1.0 if k > 0.75 else 0.3) *
                  rng.uniform(0.6, 1.3),
                  feather=0.85 * air["wet"], core=0.34,
                  side=rng.uniform(-0.10, 0.10), gap=gap,
                  pool=air.get("pool", 0.0) * 0.5, bow=spec["bow"])


def roots(cv, rng, x, y, s, st, air, k=1.0):
    """兰根 — what the blades come out of.

    A dozen blades meeting at one mathematical point is the giveaway of a
    generated clump: real ones come out of a sheath, so the bases are hidden
    under two or three short fat strokes and a couple of dark dots. It is
    four marks and it is the difference between a clump and a firework.
    """
    lt, dp = plane_tone(st, "leaf", min(1.0, k * 1.1))
    for i in range(rng.randint(2, 4)):
        a = -math.pi * 0.5 + rng.uniform(-0.75, 0.75)
        ll = s * rng.uniform(0.045, 0.105)
        stroke(cv, rng, x + rng.uniform(-s * 0.012, s * 0.012),
               y + rng.uniform(-s * 0.006, s * 0.010), a, ll,
               ll * rng.uniform(0.040, 0.065), lt, dp,
               alpha=0.80, turn=rng.uniform(-0.5, 0.5),
               belly=0.42, taper=1.1, root=0.55,
               dens=1.0, fade=0.22, feather=0.9 * air["wet"], core=0.26)
    ink = dilute(I("jiao"), 0.9)
    for _ in range(rng.randint(1, 3)):
        disc(cv, x + rng.uniform(-s * 0.02, s * 0.02),
             y + rng.uniform(-s * 0.012, s * 0.016),
             s * rng.uniform(0.004, 0.008), ink,
             rng.uniform(0.28, 0.55), squash=rng.uniform(0.6, 1.0))


# --------------------------------------------------------------------------
# 花 — 五瓣一舌, and the heart written as 品.
#
# Botanically a 兰花 is three sepals and two petals with the lip hanging
# below; the painter's formula counts the same six marks and calls them
# 五瓣 (五出) plus 舌. The convention that matters is that every one of them
# is drawn *from the centre outward*, so they all meet exactly where the
# pedicel arrives — the flower cannot come unglued from its own stalk.
# --------------------------------------------------------------------------

PETALS = (                      # (angle off the facing axis, length, width)
    (0.00, 1.00, 0.27),         # 萼片 — the one that stands up
    (2.05, 0.92, 0.25),         # 萼片 — the two that fall away
    (-2.05, 0.92, 0.25),
    (0.82, 0.58, 0.34),         # 花瓣 — the short pair that embraces the heart
    (-0.82, 0.58, 0.34),
)
# Those widths are three times the first draft's. A 兰 petal is narrow —
# about one part in four of its length — but at one part in ten, which is
# what 兰叶 proportions suggest, five of them radiating from a point do not
# read as a flower at all: they read as another little tuft of leaves, and
# the first sheets came out with sprigs of bamboo growing in the middle of
# the clump. The flower is the one place in this painting where the brush
# presses rather than pulls.


def bloom(cv, rng, cx, cy, face, s, st, air, squash=1.0, k=1.0, lip=True):
    """One open 兰花.

    `squash` is the only perspective in this painting: it compresses the
    flower across its own axis, so one draw gives a flower seen square on and
    the next the same flower seen nearly edge-on. 兰 are painted in threes
    and fours on a stalk and no two of them ever face the same way — a stalk
    of identical blossoms reads as printed, whatever the blossom is like.
    """
    lt, dp = plane_tone(st, "flower", k)
    pl = []
    for off, lf, wf in PETALS:
        # the squash, in the flower's own frame: along the axis untouched,
        # across it compressed
        ca, sa = math.cos(off), math.sin(off) * squash
        a = face + math.atan2(sa, ca)
        ll = s * lf * math.hypot(ca, sa) * rng.uniform(0.92, 1.08)
        if ll < 2.0:
            continue
        pl.append(stroke(cv, rng, cx, cy, a, ll, ll * wf * 0.5 *
                         rng.uniform(0.86, 1.14), lt, dp, alpha=0.90,
                         turn=rng.uniform(-0.45, 0.45),
                         belly=rng.uniform(0.34, 0.48), taper=1.25,
                         root=rng.uniform(0.14, 0.26),
                         dens=0.90, fade=0.20,
                         feather=0.85 * air["wet"], core=0.30,
                         pool=air.get("pool", 0.0)))
    if lip:
        # 舌 — broader, blunter and paler than the petals, and it hangs: the
        # lip is the one part of the flower that always knows where down is.
        a = face + math.pi * (1.0 if math.sin(face) < 0.0 else -1.0) \
            * rng.uniform(0.80, 1.0)
        ll = s * rng.uniform(0.40, 0.56)
        llt, ldp = plane_tone(st, "flower", k * 0.82)
        stroke(cv, rng, cx, cy, a, ll, ll * rng.uniform(0.21, 0.30),
               llt, ldp, alpha=0.86, turn=rng.uniform(-0.3, 0.3),
               belly=0.52, taper=0.95, root=0.30, dens=0.78, fade=0.10,
               feather=1.0 * air["wet"], core=0.24)
    # 点心如品 — two or three dots of 焦墨 at the throat, written as the
    # character 品: one above, two below, never in a row. They are the
    # darkest ink in the painting and the whole flower hangs off them; a
    # bloom without its heart is a handful of petals lying on the paper.
    heart = jitter(rgb(st["heart"]), rng, 6)
    hx = cx + math.cos(face) * s * 0.07
    hy = cy + math.sin(face) * s * 0.07
    nx, ny = -math.sin(face), math.cos(face)
    for j in range(rng.randint(2, 3)):
        u = (j - 0.5) * s * 0.055 * squash
        v = s * (0.0 if j == 0 else 0.06) * rng.uniform(0.7, 1.3)
        px = hx + nx * u + math.cos(face) * -v
        py = hy + ny * u + math.sin(face) * -v
        ll = s * rng.uniform(0.10, 0.17)
        stroke(cv, rng, px, py,
               face + math.pi + rng.uniform(-0.5, 0.5), ll,
               ll * rng.uniform(0.17, 0.26), heart, heart, alpha=0.88,
               turn=rng.uniform(-0.6, 0.6), belly=0.45, taper=1.0,
               root=0.45, dens=1.0, fade=0.0, feather=0.8, core=0.12)
    return pl


def bud(cv, rng, cx, cy, face, s, st, air, k=1.0):
    """蕾 — a closed bud: two or three petals still wrapped round each other,
    drawn as one narrow spindle with a second stroke inside it."""
    lt, dp = plane_tone(st, "flower", k)
    for i in range(rng.randint(2, 3)):
        a = face + rng.uniform(-0.22, 0.22)
        ll = s * rng.uniform(0.40, 0.58) * (1.0 - 0.12 * i)
        stroke(cv, rng, cx, cy, a, ll, ll * rng.uniform(0.12, 0.17),
               lt, dp, alpha=0.76, turn=rng.uniform(-0.3, 0.3),
               belly=0.46, taper=1.15, root=0.22, dens=0.92, fade=0.14,
               feather=0.9 * air["wet"], core=0.28)


def spike(cv, rng, x, y, a0, L, st, air, n=3, lean=0.0, k=0.86):
    """花箭 — the flower stalk and everything on it.

    Two rules out of the manuals. 花不出叶: the stalk stays shorter than the
    blades, because the flowers belong *inside* the clump — a spike standing
    clear above the leaves is a hyacinth. And 一枝数花, 朝向不同: the blooms
    alternate up the stalk, each on its own short pedicel with a bract at the
    joint, and each faces its own way.
    """
    lt, dp = plane_tone(st, "flower", k)
    slt, sdp = plane_tone(st, "leaf", k * 0.72)
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    sp = stroke(cv, rng, x, y, a0, L, L * rng.uniform(0.006, 0.010),
                slt, sdp, alpha=0.85, turn=sgn * rng.uniform(0.15, 0.45),
                belly=0.04, taper=0.85, root=0.95, dens=0.80, fade=0.30,
                feather=0.9 * air["wet"], core=0.22)
    m = len(sp)
    bs = L * rng.uniform(0.25, 0.33)                 # how big a bloom is
    # Where on the stalk the flowers sit: the top half, alternating, and no
    # two of them within a fifth of the stalk — blooms that land on top of
    # each other merge into one unreadable mass of petals, which is the
    # same 疏密 problem the leaves have, one scale down.
    ts = []
    for _ in range(n):
        for _try in range(8):
            t = rng.uniform(0.34, 0.97)
            if all(abs(t - u) > 0.24 for u in ts):
                ts.append(t)
                break
    ts.sort()
    side = 1.0 if rng.random() < 0.5 else -1.0
    out = []
    for i, t in enumerate(ts):
        j = min(m - 2, max(1, int(t * (m - 1))))
        px, py = sp[j]
        ax, ay = sp[j + 1][0] - sp[j - 1][0], sp[j + 1][1] - sp[j - 1][1]
        aa = math.atan2(ay, ax)
        s2 = side * (1.0 if i % 2 == 0 else -1.0)
        pa = aa + s2 * rng.uniform(0.55, 1.15)       # the pedicel's own angle
        pl = L * rng.uniform(0.05, 0.11)
        # 苞片 — the little bract where the pedicel leaves the stalk. Small,
        # but it is what makes the joint a joint instead of a fork.
        stroke(cv, rng, px, py, pa + s2 * 0.35, pl * rng.uniform(0.5, 0.8),
               pl * 0.10, slt, sdp, alpha=0.7, turn=-s2 * 0.3, belly=0.3,
               taper=1.3, root=0.5, dens=0.7, fade=0.3, feather=0.9, core=0.2)
        ped = stroke(cv, rng, px, py, pa, pl, pl * rng.uniform(0.045, 0.075),
                     slt, sdp, alpha=0.82, turn=s2 * rng.uniform(0.1, 0.5),
                     belly=0.1, taper=0.8, root=0.9, dens=0.78, fade=0.2,
                     feather=0.9 * air["wet"], core=0.2)
        ex, ey = ped[-1]
        # The bloom's centre is the pedicel's last point, pushed a little
        # further along it. Every petal starts at that centre, so the stalk
        # meets the flower where the flower's own strokes meet — attachment
        # is structural here, not a coincidence of two radii.
        fa = math.atan2(ey - py, ex - px)
        cxx = ex + math.cos(fa) * bs * 0.10
        cyy = ey + math.sin(fa) * bs * 0.10
        bz = bs * (1.0 - 0.18 * t)        # 上小下大 — the top ones younger
        if t > 0.88 and rng.random() < 0.72:
            bud(cv, rng, cxx, cyy, fa + rng.uniform(-0.3, 0.3), bz, st, air,
                k=k)
        else:
            # 低头 — an orchid nods: the flower faces outward and a little
            # down, never straight up at the ceiling.
            face = fa + rng.uniform(-0.45, 0.45) + lean * 0.4
            bloom(cv, rng, cxx, cyy, face, bz, st, air,
                  squash=rng.uniform(0.34, 1.0), k=k)
        out.append((cxx, cyy, bz))
    return sp, out


# --------------------------------------------------------------------------
# 坡石 — the ground. 兰 is never painted standing on nothing: 坡 for a slope,
# 石 for the boulder a 兰石 sheet needs, 崖 for the cliff a 崖兰 hangs off.
#
# bamboo.py's `rock` was used for the boulder first, and it is the wrong
# stone. It is a 折带 slab — level shelves, straight flanks, hard corners —
# which is right under 竹 and reads as masonry under 兰: the subject is a
# plant with nothing straight in it anywhere. 兰 is painted on a 坡石 or a
# 湖石, rounded and flat-topped, so `stone` below is this module's own.
# --------------------------------------------------------------------------

def bank(cv, rng, x, y, w, st, k=0.5, moss=1.0):
    """坡 — a low slope of 淡墨, flat enough to read as ground.

    Drawn as a filled profile rather than as strokes: a 坡 is a wash with one
    drawn edge, and the drawn edge is only its top.

    The wash is a *lens* — top profile and bottom profile meeting at both
    ends — and not a slab running off the bottom border, which is what it
    was first. A slab at eight percent alpha is still a slab: its two
    vertical sides came out as a pale rectangle sitting under the plant,
    clearly a box, visible in every sheet once it had been seen. A wash on
    paper has no sides. It thins out.
    """
    lt, dp = plane_tone(st, "rock", k)
    n = 26
    nz = Noise(rng, 5, 2)
    amp = w * rng.uniform(0.020, 0.048)
    tilt = rng.uniform(-0.09, 0.09)
    top = []
    for i in range(n + 1):
        t = i / n
        yy = (y + (t - 0.5) * w * tilt
              + (nz.at(t, 0.3) - 0.5) * amp * 2.0
              - amp * math.sin(math.pi * t) * 0.6)
        top.append((x - w * 0.5 + w * t, yy))
    depth = w * rng.uniform(0.09, 0.17)
    under = []
    for i in range(n, -1, -1):
        t = i / n
        bx, by = top[i]
        under.append((bx, by + depth * math.sin(math.pi * t) ** 0.65
                      + depth * 0.25 * nz.at(t, 0.8)))
    # two or three passes, each a little narrower, so the wash is deeper
    # where the ground is nearest the root — 淡墨积 rather than one flat tint
    for q in (1.0, 0.72, 0.46):
        poly = [(x + (bx - x) * q, by) for bx, by in top]
        poly += [(x + (bx - x) * q, by) for bx, by in under]
        fill_contours(cv, [poly], mix(lt, dp, 0.30), 0.042)
    # 勾 — the top edge, in *one or two* short pulls near the plant and
    # nowhere else.
    #
    # Drawn the obvious way — walk the whole profile in broken runs, the way
    # a rock's crest is drawn — this produced a pair of pale lines running
    # right across the sheet on either side of the clump, which is the same
    # defect gongbi.py had to have ruled out of it: a long faint straight
    # line reads as a construction line, not as ground. A 坡 under 兰 is a
    # wash with the ghost of an edge where the eye is already looking, and
    # nothing at all out at the margins.
    mid = n // 2
    for _ in range(rng.randint(1, 2)):
        i = max(0, min(n - 5, mid + int(rng.gauss(0.0, n * 0.14))))
        m = rng.randint(3, 5)
        seg = top[i:i + m + 1]
        if len(seg) > 1:
            brush_path(cv, seg, max(0.7, w * 0.0026), max(0.5, w * 0.0014),
                       mix(dp, I("zhong"), 0.4), rng.uniform(0.16, 0.30))
    # 苔点 — moss, in two or three tight clusters, never sprinkled
    for _ in range(int(moss * 2) + 1):
        c = max(0, min(n, mid + int(rng.gauss(0.0, n * 0.16))))
        bx, by = top[c]
        for _ in range(rng.randint(2, 5)):
            disc(cv, bx + rng.gauss(0.0, w * 0.020),
                 by + rng.uniform(w * 0.004, w * 0.030),
                 max(0.7, w * 0.0028) * rng.uniform(0.7, 1.5),
                 mix(dp, I("nong"), 0.7), rng.uniform(0.25, 0.55), squash=0.7)
    return top


def stone(cv, rng, cx, yb, rw, rh, st, k=0.70, moss=1.0, alpha=0.78):
    """坡石 — the boulder of 兰石图, and the one mass in this family that is
    not bamboo.py's.

    bamboo.py's `rock` is a 折带 slab: its profile is built out of level
    shelves meeting straight flanks at hard corners, deliberately, because
    that angular silhouette is what 竹石 wants and because the alternative —
    a rounded shoulder — is, in its own words, "a boulder, which is the wrong
    rock for bamboo". Under an orchid it is the wrong rock the other way
    round: 兰石 sheets drawn with it came out with the plant standing on a
    grey stepped wall, and no amount of ink in the wash fixed a silhouette
    that reads as masonry. 兰 is painted beside a 湖石 or a 坡石 — a lump,
    flat-topped, bumped and notched, steep on one side and long on the other.

    So the profile here is a flat-topped lens (sin to a power below one)
    rather than a stack of shelves, broken at two scales of noise, with a
    smaller lump shouldering out of its foot for 石分三面. Returns one crest
    y per integer column from `cx - rw/2`, the same contract as `rock`, so
    the clump can be rooted on it.
    """
    w = int(rw)
    if w < 12 or rh < 6:
        return None
    x0 = cx - rw * 0.5
    lt, dp = plane_tone(st, "rock", k)
    big = Noise(rng, 4, 2)
    fin = Noise(rng, 11, 2)
    grn = Noise(rng, 21, 2)
    # 石分三面 — the mass, and a lower lump shouldering out of one side of it
    lumps = [(0.0, 1.0, 1.0, rng.uniform(0.34, 0.55))]
    if rng.random() < 0.85:
        cw_ = rng.uniform(0.28, 0.52)
        c0 = rng.uniform(0.0, 1.0 - cw_)
        lumps.append((c0, c0 + cw_, rng.uniform(0.30, 0.62),
                      rng.uniform(0.40, 0.70)))
    tb = rng.uniform(-0.40, 0.40)                  # 不可对称
    # 一凹 — and a cleft bitten out of the crest, which is the difference
    # between a stone and a loaf
    dt, dw, da = rng.uniform(0.22, 0.78), rng.uniform(0.05, 0.11), \
        rng.uniform(0.05, 0.18)
    top = []
    for i in range(w):
        t = i / (w - 1.0)
        best = 0.0
        for c0, c1, hh, p in lumps:
            if not c0 <= t <= c1:
                continue
            u = (t - c0) / max(1e-6, c1 - c0)
            v = math.sin(math.pi * u) ** p * hh
            if v > best:
                best = v
        best *= (1.0 + tb * (t - 0.5)
                 + (big.at(t, 0.5) - 0.5) * 0.30
                 + (fin.at(t, 0.3) - 0.5) * 0.14)
        best -= da * math.exp(-((t - dt) / dw) ** 2)
        top.append(yb - max(0.0, best) * rh)
    # 石不宜尖 — three taps of smoothing. Noise on a lens profile makes peaks,
    # and a peak with two straight flanks under it is a hill: the sheet comes
    # out as a miniature 山水 with an orchid standing on the summit.
    top = [top[0]] + [(top[i - 1] + 2.0 * top[i] + top[i + 1]) * 0.25
                      for i in range(1, w - 1)] + [top[-1]]
    bot = [yb + (fin.at(i / (w - 1.0), 0.8) - 0.5) * rh * 0.10
           for i in range(w)]
    lx = gb.LIGHT[0]
    buf, cwid = cv.buf, cv.w
    for i in range(w):
        px = int(x0) + i
        if px < 0 or px >= cwid:
            continue
        ty, by = top[i], bot[i]
        d = by - ty
        if d < 1.0:
            continue
        t = i / (w - 1.0)
        sl = (top[min(w - 1, i + 2)] - top[max(0, i - 2)]) / 4.0
        face = clamp01(0.5 - 0.5 * max(-1.0, min(1.0, sl)) * lx * 1.4)
        thin = clamp01(d / max(3.0, rh * 0.06))
        for py in range(max(0, int(ty)), min(cv.h, int(by) + 1)):
            v = clamp01((py + 0.5 - ty) / d)
            sh = (0.28 + 0.48 * (1.0 - v) ** 1.3
                  + 0.24 * math.exp(-(v / 0.10) ** 2)
                  + 0.10 * face
                  + (big.at(t, v) - 0.5) * 0.30
                  + (grn.at(t, v) - 0.5) * 0.22)
            c = mix(lt, dp, clamp01(sh))
            # 下虚 — the wash dies before the ground, so the stone sits in
            # the paper instead of on it
            a = alpha * thin * (0.95 - 0.86 * v ** 1.4)
            e = min(py + 0.5 - ty, by - py - 0.5)
            if e < 1.2:
                a *= max(0.0, e / 1.2)
            if a <= 0.01:
                continue
            idx = (py * cwid + px) * 3
            for ch in range(3):
                o = buf[idx + ch]
                buf[idx + ch] = int(o + (o * c[ch] * 0.00392 - o) * a)
    # 勾 — the crest, in two or three long broken pulls, drawn with `sweep`
    # and *not* exactly on the edge of the wash: an outline that agrees with
    # a fill to the pixel is what makes a mass look cut out of paper.
    off = Noise(rng, 6, 2)
    gl = mix(lt, dp, 0.70)
    gh = mix(dp, I("zhong"), 0.55)
    step = max(2, w // 30)
    cols = list(range(0, w, step))
    live = [j for j in cols if yb - top[j] > rh * 0.12]
    if not live:
        return top
    i = 0
    while i < len(live):
        m = rng.randint(5, 10)
        run = live[i:i + m + 1]
        if len(run) > 2:
            pts = [(x0 + j, top[j]
                    + (off.at(j / (w - 1.0), 0.5) - 0.5) * rh * 0.045)
                   for j in run]
            ns = len(pts)
            hw = max(1.0, rh * 0.022)
            hws = [hw * (0.40 + 0.60
                         * math.sin(math.pi * (q + 0.5) / ns) ** 0.45)
                   for q in range(ns)]
            dn = [rng.uniform(0.72, 1.0) for _ in range(ns)]
            sweep(cv, pts, hws, dn, gl, gh,
                  alpha=rng.uniform(0.46, 0.76), feather=0.9, core=0.24)
        i += m + rng.randint(1, 3)
    # 皴 — 卷云/披麻, short arcs falling away from the crest down the shaded
    # flank, in two or three clusters. Straight horizontal pulls (which is
    # what the 折带 rock uses, correctly) came out on a round form as a set
    # of scaffolding planks floating inside the mass.
    for _ in range(rng.randint(3, 5)):
        j0 = rng.randrange(w)
        aa0 = rng.uniform(-0.45, 0.45) + (0.0 if lx > 0 else math.pi)
        cur0 = rng.uniform(0.25, 0.80) * rng.choice((1.0, -1.0))
        for _ in range(rng.randint(2, 5)):
            j = max(0, min(w - 1, j0 + int(rng.gauss(0.0, w * 0.045))))
            dep = bot[j] - top[j]
            if dep < rh * 0.22:
                continue
            sx = x0 + j
            sy = top[j] + dep * rng.uniform(0.08, 0.34)
            ll = rh * rng.uniform(0.16, 0.42)
            aa = aa0 + rng.uniform(-0.16, 0.16)
            cur = cur0 * rng.uniform(0.8, 1.2)
            pts, hws, dn = [], [], []
            ns = 5
            hw = max(0.8, rh * 0.016)
            for q in range(ns):
                u = q / (ns - 1.0)
                ang = aa + cur * u * u
                pts.append((sx + math.cos(ang) * ll * u,
                            sy + math.sin(ang) * ll * u * 0.55 + ll * u * 0.5))
                hws.append(hw * (1.0 - 0.55 * u))
                dn.append(1.0 - 0.45 * u)
            if pts[-1][1] > bot[j] - 1.0:
                continue
            sweep(cv, pts, hws, dn, mix(lt, dp, 0.6), gh,
                  alpha=rng.uniform(0.14, 0.30), feather=1.0, core=0.22)
    # 点苔 — two or three tight clusters on the crest
    for _ in range(int(moss * 2) + 1):
        j0 = rng.randrange(w)
        if yb - top[j0] < rh * 0.25:
            continue
        for _ in range(rng.randint(2, 5)):
            j = max(0, min(w - 1, j0 + int(rng.gauss(0.0, w * 0.030))))
            disc(cv, x0 + j + rng.uniform(-1, 1),
                 top[j] + rng.uniform(rh * 0.02, rh * 0.14),
                 max(0.7, rh * 0.014) * rng.uniform(0.7, 1.4),
                 mix(dp, I("nong"), 0.85), rng.uniform(0.28, 0.60), squash=0.7)
    return top


def _profile(nodes, nz, n, y, h, wob=0.055, ph=0.37):
    """Sample a chain of straight runs into n+1 depths.

    `nodes` is [(t, v)] with t rising through [0, 1] and v the drop below `y`
    measured in units of `h`. Consecutive nodes are joined by *straight*
    pulls, and that is the whole point of the thing: 折带 — a rock edge is a
    sequence of straight runs meeting at corners, and the corner is the only
    mark that tells stone from cloth. The wobble is small and low-frequency
    so the runs stay recognisably straight; at the 0.10 the first cliff used,
    the corners are sanded off and the edge is a soft curve again.
    """
    out, m = [], len(nodes) - 1
    for i in range(n + 1):
        t = i / n
        q = 0
        while q < m - 1 and t > nodes[q + 1][0]:
            q += 1
        t0, v0 = nodes[q]
        t1, v1 = nodes[q + 1]
        u = (t - t0) / max(1e-6, t1 - t0)
        out.append(y + h * (v0 + (v1 - v0) * u)
                   + h * (nz.at(t, ph) - 0.5) * wob)
    return out


def _lip_nodes(rng):
    """The underside of the overhang: down to a prow, then up to the cut end.

    This is the decision that governs whether the mass reads as rock at all,
    and the first cliff had it backwards. Its lip rose monotonically from the
    side border to a thin tip, and the ink was a band hugging that one line —
    which on paper is a long soft diagonal with a smear above it. That is
    drapery; on a pale seed it is haze. Nothing about it was wrong *locally*.
    There was simply no form: an edge is not a solid, and a solid is what the
    plant has to be hanging off.

    A 悬崖 in a 崖兰 is read from its **prow**. The underside falls as the
    rock projects, reaches its lowest point about two-thirds of the way out,
    and the mass then climbs back to a cut face. That puts a downward corner
    in the middle of the sheet — which is what the plant hangs from, what the
    eye reads as weight, and what no amount of texture inside a band supplies.
    """
    # The prow hangs *late* and the climb back is short. Put it at half the
    # span with a long steep climb after it — the first arrangement — and the
    # silhouette is a wide symmetrical V with the plant at its point, which
    # on six sheets in a row read as a pair of wings. A cliff is not
    # symmetrical about anything: it is a mass wedged into one corner with
    # one end free, so most of the span belongs to the descent.
    tp = rng.uniform(0.46, 0.80)               # where the prow hangs
    d = rng.uniform(0.32, 0.74)                # how far below the border depth
    nodes = [(0.0, 0.0)]
    # 折带 — the descent is not a ramp but three or four runs with *uneven*
    # shares of the drop, so that one of them comes out nearly level (a
    # shelf) and another nearly vertical (a step). Even shares give a
    # polyline that is a straight line with a tremble in it, which is the
    # curve the first cliff had and the reason its lip read as cloth.
    k = rng.randint(3, 4)
    wq = [rng.uniform(0.5, 1.5) for _ in range(k)]
    vq = [rng.uniform(0.06, 1.0) ** 1.7 for _ in range(k)]
    sw, sv = sum(wq), sum(vq)
    tt = vv = 0.0
    for q in range(k - 1):
        tt += wq[q] / sw
        vv += vq[q] / sv
        nodes.append((tp * tt, d * vv))
    nodes.append((tp, d))
    # ...and the climb out of the prow is one short steep run, after which
    # the underside runs out nearly level to the cut face. Letting it climb
    # all the way back above the border depth — which is where it started —
    # makes the far side of the mass a long diagonal converging on the prow,
    # and two converging sides with a point between them is a kite, not a
    # rock. The prow has to be a *notch bitten out of a block*, so the lip
    # comes back up only part of what it fell and then stops rising.
    tipv = d * rng.uniform(0.40, 0.80)
    nodes.append((tp + (1.0 - tp) * rng.uniform(0.25, 0.55), tipv))
    nodes.append((1.0, tipv + d * rng.uniform(-0.12, 0.06)))
    out = [nodes[0]]
    for t, v in nodes[1:]:
        if t > out[-1][0] + 0.04:
            out.append((min(1.0, t), v))
    if out[-1][0] < 0.999:
        out.append((1.0, nodes[-1][1]))
    return out, tp


def _recede(e, mh, fade=0.40, gpow=2.6):
    """How much of the mass survives `e` pixels above its lip.

    虚 is a distance, not a fraction. Measuring the dissolve as a fraction of
    each column's own depth empties the mass exactly where it is shallow —
    the wedge behind the cut face — which is the one place that must have
    ink in it, because a drawn edge with nothing behind it is a line lying on
    the paper. On the sheets that came out as a dashed vertical stroke
    standing clear of the rock in the top corner. The scale is the depth of
    the mass where it leaves the sheet, for every column.
    """
    vz = e / mh
    return 1.0 - fade * min(1.0, max(0.0, vz)) ** gpow


def _fold_t(f, e):
    """Where a crease of 石分三面 crosses the span, `e` pixels above the lip.

    A crease held at a constant `t` is a ruled vertical line, and a value
    step along a ruled vertical line is not two faces of a rock, it is a
    crease in the *paper*. Slope plus one slow bow is enough to break it;
    what it must not have is a tremble, which reads as a torn edge.
    """
    return f[0] + f[1] * e + f[2] * math.sin(f[3] * e + f[4])


def _anchor(t, root=0.76):
    """实于其根 — heaviest where the mass runs off the side of the sheet.

    The opposite gradient, 虚 at the border and 实 at the free end, leaves
    the visible rock a band following the lip: the corner the mass is
    anchored in washes out, nothing says the stone continues past the frame,
    and the silhouette collapses into the two converging edges that the
    underside and the cut face make between them — which on paper is a kite.

    `root` is the floor, and it had to come up once the far end stopped
    being a taper and became a *cut face*. 虚 is a statement about something
    that trails off; a corner drawn on three sides with a hard vertical
    does not trail off, and at 0.62 the enclosed third of the mass measured
    within four levels of bare paper. An outline round nothing is a paper
    cut-out whatever the outline is doing.
    """
    return root + (1.0 - root) * (1.0 - min(1.0, max(0.0, t))) ** 1.1


def _cliff_wash(cv, xs, lip, top, lt, dp, mot, grn, n, folds=(), fval=(1.0,),
                alpha=0.82, body=0.44, band=0.56, bd=60.0, sink=0.50,
                fade=0.40, blot=0.55, gpow=2.6, root=0.76, mh=None):
    """染 — the body of the mass, between its top boundary and its lip.

    Two failures are worth keeping, because they are opposite and the way out
    is not between them.

    *The slab.* Filled from the lip to the top border at one tone, the mass
    measures 155–172 against paper at 200 over a region 300×600 pixels, cut
    dead straight by two borders. That is a dirty corner, not a rock, and a
    reported bug.

    *The band.* The repair was to stop the wash a fixed distance above the
    lip and let the paper alone past it. The corner stayed clean and the
    cliff disappeared: what is left of a 悬崖 whose only ink is a 120-pixel
    shadow under one line is a shadow under one line.

    The way out is not a thickness, it is a **distribution**. Ink is spent
    where a cliff spends it — a short, strong, opaque band at the overhang
    (`band`, over `bd` pixels) — and the rest of the mass carries a 淡墨 wash
    at a fifth of that alpha (`sink`), which against this paper is a six per
    cent darkening: visibly stone, nowhere near dirty. The wash then fades
    out upward (`fade`) so the top border cuts almost nothing, and is broken
    by its own grain (`blot`) so it is a wash and not a fill. A large pale
    area with crisp edges on it reads as rock; a middling grey one never
    does, at any size.

    `folds` are the creases of 石分三面, as (t at the lip, dt per pixel of
    height). They do not draw anything here — they switch `fval`, so that the
    facet on one side of a crease is a different value from the facet on the
    other. That switch is most of what makes the mass look faceted; the 勾
    drawn along the crease afterwards only confirms it.
    """
    buf, cw, chh = cv.buf, cv.w, cv.h
    ramp = [mix(lt, dp, q / 63.0) for q in range(64)]
    if mh is None:
        mh = max(1.0, lip[0] - top[0])
    dx = xs[n] - xs[0]
    nf = len(folds)
    # the creases, tabulated per pixel of height: `_fold_t` has a sine in it
    # and this is the one loop in the module that runs per pixel
    maxe = int(max(lip[i] - top[i] for i in range(n + 1))) + 2
    ftab = [[_fold_t(f, float(q)) for q in range(maxe)] for f in folds]
    xa, xb = int(min(xs[0], xs[n])), int(max(xs[0], xs[n]))
    for px in range(max(0, xa), min(cw, xb + 1)):
        t = (px - xs[0]) / dx
        if not 0.0 <= t <= 1.0:
            continue
        fi = t * n
        j = min(n - 1, int(fi))
        u = fi - j
        ly = lip[j] + (lip[j + 1] - lip[j]) * u
        ty = top[j] + (top[j + 1] - top[j]) * u
        dep = ly - ty
        if dep < 1.0 or ly <= 1.0:
            continue
        cut = ty > 1.0                        # a drawn top edge, or the border
        rf = _anchor(t, root)                  # hoisted: it has no y in it
        for py in range(min(chh - 1, int(ly)), max(0, int(ty)) - 1, -1):
            e = ly - py - 0.5
            if e < 0.0:
                continue
            v = e / dep
            if v > 1.0:
                break
            b = e / bd
            if b > 1.0:
                b = 1.0
            fv = fval[0]
            ei = int(e)
            for q in range(nf):
                if ftab[q][ei] < t:
                    fv = fval[q + 1]
            mn = mot.at(t, v)
            ib = 1.0 - b
            # 下重上轻, and steeply. The grain runs the other way: weak at the
            # lip, where a wet 勾 lays solid ink and the edge has to stay one
            # tone, and strong in the body, which is where a dry brush skips.
            gn = 0.30 + 0.70 * b
            sh = (body * fv + band * ib ** 1.9 * (0.85 + 0.15 * fv)
                  + (mn - 0.5) * 0.30 * gn)
            if b < 1.0:
                sh += (grn.at(t, b) - 0.5) * 0.22 * gn
            # Two separate things are being asked of the alpha and they have
            # to be *shaped* separately, not multiplied as two ramps of the
            # same kind. `sink` is how much weaker the body is than the band
            # — a step, over `bd` pixels. `_recede` is the mass dissolving as
            # it goes back into the sheet, and it must stay near 1 over most
            # of the depth and fall off a cliff at the end, or the two ramps
            # compound: at a linear fade the body came out at 0.066 alpha,
            # which measured two levels against paper, and the finished rock
            # was an empty outline with its 皴 floating inside it.
            # (`_recede` is inlined here and nowhere else: this is the one
            # loop in the module that runs per pixel.)
            vz = e / mh
            if vz > 1.0:
                vz = 1.0
            a = (alpha * (sink + (1.0 - sink) * ib ** 2.2) * rf
                 * (1.0 - fade * vz ** gpow)
                 * (1.0 - blot * b * (1.0 - mn)))
            # 面有向背 — a facet is not only a different value, it is a
            # different *weight* of ink. Changing the shade alone moves a
            # 淡墨 wash by a few levels and the mass stays one flat grey
            # cut-out; changing the alpha with it is what makes the eye
            # read two planes meeting at the crease rather than one plane
            # with a line on it.
            # ...but the *shade* carries most of it and the alpha only
            # seconds it. Swung as hard as the shade, the pale facet of a
            # pale mass came out at eight levels against paper over a third
            # of the rock, with a hard contour round it — two correct ideas
            # multiplying into a hole.
            a *= 0.70 + 0.30 * fv
            if e < 1.2:
                a *= e / 1.2
            if cut:
                d2 = py - ty
                if d2 < 1.2:
                    a *= max(0.0, d2 / 1.2)
            if a <= 0.012:
                continue
            if sh < 0.0:
                sh = 0.0
            elif sh > 1.0:
                sh = 1.0
            c = ramp[int(sh * 63.0)]
            idx = (py * cw + px) * 3
            for q in range(3):
                o = buf[idx + q]
                buf[idx + q] = int(o + (o * c[q] * 0.00392 - o) * a)


def _gou(cv, rng, pts, hw, gl, gh, keep=None, a0=0.60, a1=0.88, run=(8, 16)):
    """勾 — an edge of the mass, in long broken pulls.

    Drawn with this module's `sweep`, because `brush_path` composites every
    sample of its polyline separately and a line traced along a profile
    sampled every eight pixels then comes out as a bead every eight pixels:
    the first cliff had a zipper along its lip instead of an edge.

    `keep` is a per-sample weight, and it does two jobs with one array. Zero
    drops the sample, which is how the crease stops before it reaches bare
    paper; a fraction scales the pull's alpha, which is how the cut face
    dissolves as it climbs into the part of the mass that is itself
    dissolving. A 勾 that stays at full strength where its wash has faded out
    is a dark wire standing on nothing, and that is the single commonest way
    a drawn rock goes wrong here.
    """
    n = len(pts) - 1
    live = [q for q in range(n + 1) if keep is None or keep[q] > 0.03]
    i = 0
    while i < len(live):
        idx = live[i:i + rng.randint(*run) + 1]
        c = 1
        while c < len(idx) and idx[c] == idx[c - 1] + 1:
            c += 1
        idx = idx[:c]
        seg = [pts[q] for q in idx]
        ns = len(seg)
        if ns > 2:
            ka = 1.0 if keep is None else sum(keep[q] for q in idx) / ns
            hws = [hw * (0.40 + 0.60
                         * math.sin(math.pi * (q + 0.5) / ns) ** 0.45)
                   for q in range(ns)]
            dn = [rng.uniform(0.74, 1.0) for _ in range(ns)]
            sweep(cv, seg, hws, dn, gl, gh,
                  alpha=rng.uniform(a0, a1) * ka, feather=0.9, core=0.22)
        i += len(idx) + rng.randint(1, 3)


def ledge(cv, rng, side, y, h, st, k=0.62):
    """崖 — the cliff a 崖兰 hangs from.

    Twenty-odd rounds of repair went into the first version of this function
    and it still could not be recognised as a cliff. Every one of those
    rounds was a correct local fix — the wash was too dark, the corner was
    dirty, the 皴 stood on bare paper, the taper grew a pale horn — and the
    sum of them was a mass with less and less ink in it until what remained
    was a soft grey smear with one hard line under it. The complaint that
    ended it was the right one: *it is hard to tell that it is a cliff.*

    The mistake was in the brief, which said: a 悬崖 has one edge, so draw
    that edge and wash behind it. Three things follow from dropping it.

    **形 — the silhouette is a prow, and it is closed.** The underside falls
    as the rock projects, hangs lowest about two-thirds out, and climbs back
    to a face that is *cut* rather than faded (`_lip_nodes`, and the wall
    below). A shape that ends in a drawn corner is a solid; a shape that
    fades out is weather.

    **质 — ink by distribution, not by thickness.** A short opaque band at
    the overhang, a 淡墨 wash everywhere else, dissolving upward
    (`_cliff_wash`). This is what lets the mass be large — and it has to be
    large, because a rock the plant hangs from has to outweigh the plant.

    **理 — 石分三面.** One or two creases run from the lip up into the body,
    each a change of value with a 勾 along it, and the 皴 in each facet are
    pulled by a different hand. A face divided into facets is the plainest
    statement in the language that something is stone, and it costs two lines.

    The rest is what hangs on that frame: a nearer block near the corner
    (崖有两层), cracks off the lip, 苔点 in the crevices, and one or two small
    tufts of grass hanging from the near half of the lip — which are there to
    be *small*, since nothing says cliff like something on it that isn't.
    """
    W, H = cv.w, cv.h
    lt, dp = plane_tone(st, "rock", k)
    n = 64
    nz = Noise(rng, 7, 2)
    # fine cells: 8×6 over a mass 600 pixels deep gives blotches 100 pixels
    # across, and a 100-pixel blotch is not grain, it is a cloud — which was
    # most of what "dirty" meant in the original bug report.
    mot = Noise(rng, 13, 9)
    grn = Noise(rng, 19, 3)                    # 石骨 — the grain under the lip
    x0 = 0.0 if side < 0 else float(W)
    x1 = W * (rng.uniform(0.50, 0.70) if side < 0 else rng.uniform(0.30, 0.50))
    span = x1 - x0
    xs = [x0 + span * (i / n) for i in range(n + 1)]
    nodes, tp = _lip_nodes(rng)
    lip = _profile(nodes, nz, n, y, h)
    lip = [max(h * 0.24, min(H - h * 0.08, q)) for q in lip]

    def lip_at(t):
        f = min(1.0, max(0.0, t)) * n
        j = min(n - 1, int(f))
        return lip[j] + (lip[j + 1] - lip[j]) * (f - j)

    # 崖顶 — and this is the line that finally made the thing a rock.
    #
    # The mass used to run to the top border across its whole span, on the
    # argument that a 悬崖 comes in from above and the border is its edge.
    # What that buys is a shape with one drawn side: the underside. The other
    # three boundaries are the sheet's own edges and a fade, and a region
    # bounded by a fade is a cloud — which is what it looked like, at every
    # tone the wash was tried at, for twenty-odd rounds. 山水画 draws the rock
    # 勾 first and treats 染 as a whisper, and it is right to: **a rock is a
    # contour, not a fill.**
    #
    # So the slab is closed on three sides. It leaves the top border only
    # where it is anchored (t < `ct0`); past that the crest comes down inside
    # the sheet in 折带 runs, and the far end is a short vertical face between
    # crest and lip. Enter at a corner, descend, cut, return along the
    # overhang, hang at the prow — a silhouette an eye can walk round.
    ct0 = rng.uniform(0.14, 0.48)
    thk = h * rng.uniform(0.36, 0.95)
    cn = [(0.0, 0.0), (ct0, 0.0)]
    kc = rng.randint(2, 3)
    wq = [rng.uniform(0.6, 1.5) for _ in range(kc)]
    # 肩 — the shoulder takes most of the drop, so the crest leaves the
    # border at an angle. Shared evenly, the first run is nearly level and
    # lies a few pixels under the frame for half the sheet, which reads as a
    # ruled line parallel to the edge of the paper and not as stone at all.
    vq = sorted((rng.uniform(0.15, 1.0) ** 1.4 for _ in range(kc)),
                reverse=True)
    sw, sv = sum(wq), sum(vq)
    tt = vv = 0.0
    for q in range(kc - 1):
        tt += wq[q] / sw
        vv += vq[q] / sv
        cn.append((ct0 + (1.0 - ct0) * tt, vv))
    cn.append((1.0, 1.0))
    crest = _profile(cn, nz, n, 0.0, thk, wob=0.06, ph=0.53)

    # 收 — and the mass has to be *allowed* to end.
    #
    # The crest and the lip are drawn from two independent profiles, so on
    # any seed where the lip climbs at the far end the two run nearly
    # parallel and a few tens of pixels apart. Clamped to a floor thickness
    # — which is what `h * 0.16` below is for, since a mass ending in a point
    # is a flame and not a rock — that parallel run comes out as a straight
    # finger of constant width poking out of the side of the cliff and
    # stopped by a riser: a spout on a teapot. The clamp was not the error;
    # carrying the mass to `x1` regardless was. So find where the rock
    # genuinely runs out of thickness, cut the whole span there, and let the
    # far face be the corner where crest and lip converge.
    def _ipl(arr, t):
        f = min(1.0, max(0.0, t)) * n
        j = min(n - 1, int(f))
        return arr[j] + (arr[j + 1] - arr[j]) * (f - j)

    thin = h * 0.30
    ce = n
    lo = max(int(n * (tp + 0.07)), n // 2)
    while ce > lo and lip[ce] - crest[ce] < thin:
        ce -= 1
    if ce < n:
        ct = ce / n
        span *= ct
        x1 = x0 + span
        xs = [x0 + span * (i / n) for i in range(n + 1)]
        lip = [_ipl(lip, ct * i / n) for i in range(n + 1)]
        crest = [_ipl(crest, ct * i / n) for i in range(n + 1)]
        ct0, tp = min(0.90, ct0 / ct), min(0.94, tp / ct)
    # 崖有厚薄 — but never less than this, or the far end tapers to a point
    # and a mass that ends in a point is a flame.
    top = [max(0.0, min(crest[i], lip[i] - h * 0.16)) for i in range(n + 1)]

    def top_at(t):
        f = min(1.0, max(0.0, t)) * n
        j = min(n - 1, int(f))
        return top[j] + (top[j + 1] - top[j]) * (f - j)

    # 墨不到处笔不到 — a 勾 may be no stronger than the mass behind it. The
    # fault this fixes was three faults: the cut face standing in the pale top
    # corner as a dashed vertical line with nothing under it, creases running
    # on past the ink, and 皴 floating above the wash like rain. They had one
    # cause — every drawn mark was fading on a ramp of its own invention while
    # the wash faded on `_recede` × `_anchor` — and so they have one fix, which
    # is to ask the wash. `ink(t, y)` is what the wash has at that point, so
    # any mark multiplied by it dies exactly where its support does.
    mh = max(1.0, lip[0] - top[0])

    def ink(t, yy):
        return _recede(lip_at(t) - yy, mh) * _anchor(t)

    # 石分三面 — the creases, and the values they separate. Adjacent facets
    # are made to differ by a real interval: two facets within a tenth of
    # each other are one facet with a line drawn on it.
    nf = rng.randint(1, 2)
    folds = [(q, rng.uniform(-0.0014, 0.0014), rng.uniform(0.012, 0.032),
              rng.uniform(0.010, 0.026), rng.uniform(0.0, 6.283))
             for q in sorted(rng.uniform(0.16, 0.82) for _ in range(nf))]
    fval = [rng.uniform(0.70, 1.10)]
    for _ in range(nf):
        v = rng.uniform(0.62, 1.70)
        while abs(v - fval[-1]) < 0.42:
            v = rng.uniform(0.62, 1.70)
        fval.append(v)
    # 皴随石势 — the rake of one facet. Narrow: at ±0.8 radians a cluster
    # lies across the face instead of down it, and strokes lying across a
    # near-horizontal overhang are sticks on the paper. 皴 follow the fall
    # of the stone, and on a cliff the stone falls.
    fslant = [rng.choice((-1.0, 1.0)) * rng.uniform(0.14, 0.46)
              for _ in range(nf + 1)]

    # 墨分深浅, but in proportion to the thing. `bd` is the depth of the
    # strong band under the overhang and everything else is measured off it —
    # the length of a crease, the height 皴 start at, the length of one rake.
    # Fixed against `h` it was right for a mass of average thickness and
    # wrong for a deep one: at `thk` near its ceiling the band covers a
    # quarter of the mass and the other three quarters sit at the 淡墨 floor,
    # so the silhouette encloses nothing and the whole sheet reads as a pale
    # sail. A rock's dark side is a fraction *of the rock*.
    bd = min(h * 0.52,
             max(h * rng.uniform(0.24, 0.34), mh * rng.uniform(0.38, 0.50)))
    _cliff_wash(cv, xs, lip, top, lt, dp, mot, grn, n,
                folds=folds, fval=fval, bd=bd)

    # 崖有两层 — a nearer block, jutting down out of the overhang near the
    # corner the rock comes from. A cliff drawn as one wedge with one edge has
    # nothing an eye can read as a solid, and no amount of texture inside it
    # helps: what makes it stone is a second form in front of the first,
    # because the 勾 of the back lip then runs *across* the mass as a fold
    # instead of outlining a silhouette.
    #
    # 近实远虚 — and it is the darker of the two. Painted lighter, a shape
    # enclosed by two converging 勾 lines reads as a hole in the cliff. It is
    # also kept clear of the prow: the sheet may have only one lowest corner,
    # and that corner is where the plant hangs from.
    pedge = None
    if rng.random() < 0.80:
        pt = rng.uniform(0.17, 0.36)
        room = lip_at(tp) - max(lip_at(pt * 0.5), lip_at(pt)) - h * 0.04
        dmax = min(h * rng.uniform(0.21, 0.40), room)
        # Two gates, and they pull opposite ways. Too high and the short
        # mass rejects the second layer on most seeds — and the second layer
        # is the whole reason the back lip reads as a fold across a solid
        # rather than as the outline of one. Too low and what gets drawn is
        # a shallow rectangle with a riser at each end, hanging off the side
        # of the rock like a handle. A block has to be deep enough to be a
        # block — deep against its own width, which is why the span came
        # down as the drop went up — and below that, draw nothing.
        #
        # The second gate is the one that mattered: `h * 0.11` is a gate on
        # *depth*, and a block 40 pixels deep is fine at 50 wide and a
        # lozenge glued to the lip at 120. Only the ratio says handle.
        if dmax > h * 0.11 and dmax > abs(span) * pt * 0.52:
            xs2 = [x0 + span * pt * (i / n) for i in range(n + 1)]
            base = [lip_at(pt * (i / n)) for i in range(n + 1)]
            bn = [(0.0, rng.uniform(0.80, 1.00))]
            mm = rng.randint(1, 2)
            for q in range(1, mm + 1):
                bn.append((q / (mm + 1.0) * rng.uniform(0.85, 1.15),
                           rng.uniform(0.55, 1.00)))
            bn.append((1.0, rng.uniform(0.26, 0.50)))
            cl = [bn[0]]
            for t_, v_ in bn[1:]:
                if t_ > cl[-1][0] + 0.05:
                    cl.append((min(1.0, t_), v_))
            if cl[-1][0] < 0.999:
                cl.append((1.0, bn[-1][1]))
            drop = _profile(cl, nz, n, 0.0, dmax, wob=0.10, ph=0.71)
            pedge = [base[i] + max(0.0, drop[i]) for i in range(n + 1)]
            _cliff_wash(cv, xs2, pedge, base, lt, dp, mot, grn, n,
                        alpha=0.92, body=0.60, band=0.28, bd=dmax * 0.85,
                        sink=0.70, fade=0.26, blot=0.34, gpow=1.6, root=0.86)

    # --- 勾 ---------------------------------------------------------------
    gl = mix(lt, dp, 0.70)
    gh = mix(dp, I("zhong"), 0.55)
    # the lip, the heaviest line on the sheet after the blades themselves
    _gou(cv, rng, [(xs[i], lip[i]) for i in range(n + 1)],
         max(1.0, h * 0.014), gl, gh, a0=0.66, a1=0.92, run=(10, 20))
    # 崖顶 — the crest, from where it leaves the top border to the far corner,
    # and then straight on down the far face without lifting the brush: it is
    # one edge of one solid and it turns a corner, so it is one pull. Lighter
    # than the lip — the top of a thing lit from above takes less ink than the
    # underside of it, and more to the point an overhang is what a cliff is,
    # so the overhang gets the ink.
    cmin = max(2.0, h * 0.07)
    crst = [(x0 + span * (i / n) + rng.uniform(-1.0, 1.0), top[i])
            for i in range(n + 1) if top[i] > cmin]
    if len(crst) > 3:
        # 断崖 — the cut face. Dropped straight down from the last crest
        # point it came out a ruled vertical eighty pixels long and one
        # width throughout, which is 界画 and belongs in the pavilion
        # module, not here. A cut face leans *into* the mass (the rock is
        # undercut, which is the whole claim a 悬崖 makes), it is walked
        # down in more than three steps so the 勾 can break and re-wet
        # along it, and it is allowed to wander a couple of pixels.
        cdrop = lip[n] - top[n]
        lean = -span * rng.uniform(0.010, 0.040)
        crst += [(x1 + lean * (q / 6.0) ** 0.75 + rng.uniform(-2.2, 2.2),
                  top[n] + cdrop * (q / 6.0) ** 1.12) for q in range(1, 7)]
        _gou(cv, rng, crst, max(0.9, h * 0.011), gl, gh,
             keep=[ink((p[0] - x0) / span, p[1]) ** 0.5 for p in crst],
             a0=0.50, a1=0.80, run=(16, 30))
    if pedge is not None:
        pg = [(xs2[i], pedge[i]) for i in range(n + 1)]
        # ...and it ends in a cut, not a point: the drop is stopped while it
        # is still a third deep and the line turns back up the face — but
        # obliquely, leaning back over the block the way the big mass's far
        # face leans back over the mass. A vertical riser closes a rectangle.
        plean = span * pt * rng.uniform(0.07, 0.20)
        pg += [(xs2[n] + plean * (q / 4.0) ** 0.8 + rng.uniform(-1.5, 1.5),
                pedge[n] + (base[n] - pedge[n]) * q / 4.0) for q in (1, 2, 3, 4)]
        _gou(cv, rng, pg, max(0.9, h * 0.013), gl,
             mix(dp, I("nong"), 0.50), a0=0.70, a1=0.95, run=(12, 24))
    # the creases. Interior lines are *lighter* than the silhouette — a fold
    # drawn as heavily as an edge cuts the mass into two masses.
    # A crease that runs the full depth of the mass is not a crease, it is a
    # crack, and a near-vertical pale line four hundred pixels long reads as
    # a scratch on the paper. What divides the face is the change of value
    # across it (`fval`); the line only has to say where that change starts.
    for fd in folds:
        ft = fd[0]
        ly = lip_at(ft)
        ex = min(ly * 0.82, bd * rng.uniform(1.3, 2.3))
        pts, ks = [], []
        for q in range(13):
            f = q / 12.0
            yy = ly - ex * f
            # the same curve the value step follows, or the line and the
            # step part company and the face grows a second crease
            tt = _fold_t(fd, ex * f) + rng.uniform(-0.004, 0.004)
            pts.append((x0 + span * tt, yy))
            ks.append((1.0 - f) ** 0.8 * ink(tt, yy))
        _gou(cv, rng, pts, max(0.8, h * 0.009), gl, mix(dp, I("zhong"), 0.30),
             keep=ks, a0=0.30, a1=0.56, run=(4, 9))
    # 石纹 — a crack or two running up off the lip, dying inside the ink
    for _ in range(rng.randint(1, 2)):
        ct = rng.uniform(0.08, 0.72)
        cy = lip_at(ct)
        cx = x0 + span * ct
        pth, aa = [(cx, cy)], -math.pi * 0.5 + rng.uniform(-0.55, 0.55)
        far = min(cy - top_at(ct), bd * 1.4) * rng.uniform(0.35, 0.80)
        while cy > pth[0][1] - far and cy > 4.0:
            s_ = h * rng.uniform(0.05, 0.09)
            aa += rng.uniform(-0.30, 0.30)
            cx += math.cos(aa) * s_
            cy += math.sin(aa) * s_
            pth.append((cx, cy))
        ns = len(pth)
        if ns > 2:
            hw = max(0.9, h * 0.012)
            sweep(cv, pth, [hw * (1.0 - 0.72 * q / (ns - 1.0))
                            for q in range(ns)],
                  [1.0 - 0.75 * q / (ns - 1.0) for q in range(ns)],
                  mix(lt, dp, 0.60), gh,
                  alpha=rng.uniform(0.44, 0.68) * ink(ct, pth[-1][1]) ** 0.6,
                  feather=1.0, core=0.20)

    # --- 皴 ----------------------------------------------------------------
    # Over the whole face, not only along the lip. The first cliff put its
    # texture in a single row just above the edge, which is a tuft of grass
    # growing on a line; a cliff is textured everywhere and *concentrated* at
    # the overhang, which is a distribution, not a place. Pulls are grouped —
    # 皴 are brushed in sets, a few strokes of one hand in one direction — and
    # the direction belongs to the facet, so a crease is also a change of
    # grain. Sampled afresh per stroke they fan out and read as thorns.
    chz = mix(dp, I("zhong"), 0.34)
    for _ in range(rng.randint(15, 21)):
        ct = rng.uniform(0.04, 0.94)
        ly, ty = lip_at(ct), top_at(ct)
        dep = ly - ty
        if dep < h * 0.10:
            continue
        fi = 0
        for q in range(nf):
            if folds[q][0] < ct:
                fi = q + 1
        aa0 = -math.pi * 0.5 + fslant[fi]
        # 下实上虚 — and measured in pixels off the lip, not in fractions of
        # this column's depth. A fraction spreads the sets evenly over
        # whatever height the column happens to have, so on a deep column
        # they climb to where the wash has nothing in it and a short dark
        # dash lands on bare paper: six of those, evenly scattered, are
        # rain. `bd` is where the ink is, so that is where the 皴 are, and
        # the few that climb past it are dimmed by `ink` — which knows how
        # much wash is under them, which is the only thing that settles it.
        cyc = ly - min(dep * 0.82, bd * rng.random() ** 2.2 * 1.25)
        cfa = ink(ct, cyc)
        # 一笔之长, for the whole set: strokes of one hand are the same
        # length and the same angle to within a hair. Re-rolled per stroke
        # they splay, and a splayed set of five short marks is a footprint.
        #
        # 直皴 — and the strokes are *long*. Short detached dashes at this
        # alpha are tally marks, however many are laid down and wherever
        # they are put: what makes 皴 read as the grain of stone instead is
        # that neighbouring pulls overlap, so a set resolves into a patch of
        # dry texture rather than into its members. Length rather than
        # count: twice as many ticks is twice as much rain.
        lbase = min(h * 0.17, dep * 0.52, bd * 1.45) * rng.uniform(0.62, 1.0)
        for _ in range(rng.randint(6, 11)):
            # tight. Spread a set over a tenth of the span and it is not a
            # set, it is six separate marks, and six separate short marks
            # evenly scattered over a pale field is rain.
            sx = x0 + span * (ct + rng.uniform(-0.015, 0.015))
            sy = cyc + rng.gauss(0.0, min(dep * 0.030, bd * 0.09))
            ll = lbase * rng.uniform(0.74, 1.14)
            aa = aa0 + rng.uniform(-0.05, 0.05)
            # A long pull that would run out of the mass is *shortened*, not
            # dropped. Dropping it is what put the gaps in the sets: the
            # columns where the rock is shallow are exactly the ones where
            # the full-length stroke fails, so those columns lost their 皴
            # entirely and the texture pooled in the deep middle.
            ts = (sx - x0) / span
            if not 0.02 < ts < 0.98 \
                    or sy > lip_at(ts) - 1.5 or sy < top_at(ts) + 1.5:
                continue
            for _ in range(5):
                ex2 = sx + math.cos(aa) * ll
                ey2 = sy + math.sin(aa) * ll
                tq = (ex2 - x0) / span
                if 0.02 < tq < 0.98 and top_at(tq) + 1.5 < ey2 \
                        < lip_at(tq) - 1.5:
                    break
                ll *= 0.62
            else:
                continue
            if ll > h * 0.012:
                brush_seg(cv, sx, sy, ex2, ey2,
                          max(0.7, h * 0.0062), max(0.4, h * 0.0022), chz,
                          rng.uniform(0.10, 0.25) * cfa)

    # --- 点苔, and what lives on the rock ------------------------------------
    # Clusters, not a row: spaced evenly along the lip they read as studs on
    # a belt and the edge stops being stone. They go where moss goes — the
    # prow, and the crevices where a crease meets the edge.
    spots = [tp] + [f[0] for f in folds]
    for ct in spots:
        if rng.random() < 0.34:
            continue
        bx, by = x0 + span * (ct + rng.uniform(-0.05, 0.05)), lip_at(ct)
        for _ in range(rng.randint(2, 5)):
            disc(cv, bx + rng.gauss(0.0, h * 0.045),
                 by - rng.uniform(h * 0.004, h * 0.07),
                 max(0.6, h * 0.0095) * rng.uniform(0.7, 1.3),
                 mix(dp, I("nong"), 0.7), rng.uniform(0.26, 0.52), squash=0.7)
    # and on the crest, which is where moss is actually seen from below: the
    # dots sit *on* the drawn top edge, half above it and half below, and a
    # row of small dark marks breaking a long pale line is the oldest way in
    # the language of saying that the line is the top of a rock.
    for _ in range(rng.randint(1, 3)):
        ct = rng.uniform(max(0.02, ct0 + 0.04), 0.96)
        ty = top_at(ct)
        if ty < h * 0.05:
            continue
        bx = x0 + span * ct
        for _ in range(rng.randint(2, 4)):
            disc(cv, bx + rng.gauss(0.0, h * 0.030),
                 ty + rng.gauss(0.0, h * 0.016),
                 max(0.6, h * 0.0085) * rng.uniform(0.7, 1.3),
                 mix(dp, I("nong"), 0.55), rng.uniform(0.22, 0.44), squash=0.7)
    # 垂草 — one or two small tufts hanging from the near half of the lip.
    # They are the cheapest statement of scale on the sheet: a cliff with
    # something little hanging off it is a cliff, and they are kept to the
    # corner side so they never argue with the orchid at the prow.
    for _ in range(rng.randint(0, 2)):
        ct = rng.uniform(0.06, 0.40)
        bx, by = x0 + span * ct, lip_at(ct)
        gl2, gd2 = plane_tone(st, "leaf", k * 0.46)
        for _ in range(rng.randint(3, 5)):
            ll = h * rng.uniform(0.05, 0.12)
            a1_ = math.pi * 0.5 + rng.uniform(-0.75, 0.75)
            px_ = bx + rng.gauss(0.0, h * 0.012)
            pth = [(px_, by - h * 0.004)]
            aa = a1_
            for q in range(3):
                aa += rng.uniform(-0.30, 0.42) * (1.0 if a1_ > 1.57 else -1.0)
                pth.append((pth[-1][0] + math.cos(aa) * ll / 3.0,
                            pth[-1][1] + math.sin(aa) * ll / 3.0))
            hw = max(0.7, h * 0.006)
            sweep(cv, pth, [hw * (1.0 - 0.8 * q / 3.0) for q in range(4)],
                  [1.0, 0.8, 0.55, 0.3], gl2, gd2,
                  alpha=rng.uniform(0.38, 0.62), feather=1.0, core=0.22)
    return [(xs[i], lip[i]) for i in range(n + 1)]


# --------------------------------------------------------------------------
# 一丛 — leaves, root and flowers together, with the measurements that keep
# the group legible
# --------------------------------------------------------------------------

def clump(cv, rng, x, y, L, st, air, n=8, lean=0.0, bias=1.0, hang=False,
          k=1.0, spikes=1, occ=None, stat=None):
    """Draw one 丛 and return its leaf spines.

    The leaves are drawn pale-first so a dark blade always lies over a light
    one and never the other way round — in a 写意 subject the only depth cue
    there is is which stroke is on top of which, and getting it backwards
    makes the clump read as flat.
    """
    plan = leaf_plan(rng, n, bias=bias, hang=hang)
    plan.sort(key=lambda d: d["k"])
    spines = []
    for spec in plan:
        # 根有广狭 — the blades come out of a sheath an inch across, not out
        # of one point. Two percent of the leaf length of scatter at the base
        # is invisible as scatter and is the difference between a plant and a
        # bunch of flowers held in a fist.
        sp = draw_leaf(cv, rng, x + rng.gauss(0.0, L * 0.013),
                       y + rng.uniform(-L * 0.004, L * 0.014),
                       spec, L, st, air, lean=lean,
                       alpha=0.95 * (1.0 if spec["k"] > 0.7 else 0.88))
        spines.append(sp)
        if occ is not None:
            occ.add_path(sp, L * 0.02, L * 0.004, step=4)
    if stat is not None:
        # 凤眼 — measured, not hoped for. Every pair of blades, every
        # crossing, and the angle each crossing makes: a clump wants two or
        # three of them at a good lean, and zero means a fountain while a
        # dozen means a thicket.
        for i in range(len(spines)):
            for j in range(i + 1, len(spines)):
                stat.extend(_cross(spines[i], spines[j]))
    roots(cv, rng, x, y, L, st, air, k=k)

    # --- 花在叶间 — the spikes go where the leaves are not ----------------
    #
    # Both halves of this are measured off the spines that were actually
    # drawn, not off the parameters that drew them. The gap a spike needs is
    # a property of this clump's leaves, and so is the ceiling: 花不出叶
    # cannot be enforced by "make the stalk shorter than L", because the
    # blades curl over and their vertical reach is barely half their length —
    # a stalk of 0.5 L stands clear above every leaf on the sheet, which is
    # how the early sheets grew hyacinths.
    pool = [p for sp in spines for p in sp[::3]]
    ceil_y = min((p[1] for p in pool), default=y) + L * 0.06
    for _ in range(spikes):
        best, best_s = None, None
        for _ in range(20):
            sl = L * rng.uniform(0.26, 0.50)
            if hang:
                # off a cliff the spike hangs too, outward and down
                sa = math.atan2(rng.uniform(0.15, 0.95),
                                (1.0 if bias >= 0 else -1.0))
            else:
                sa = -math.pi * 0.5 + rng.uniform(-0.55, 0.55) + lean * 0.7
            ex = x + math.cos(sa) * sl
            ey = y + math.sin(sa) * sl
            if not (L * 0.08 < ex < cv.w - L * 0.08
                    and L * 0.06 < ey < cv.h - L * 0.06):
                continue
            if not hang and ey - L * 0.10 < ceil_y:   # the top bloom too
                continue
            # Clearance is measured where the *blooms* will be — three
            # points up the top half of the stalk — and not at the stalk's
            # tip. Scoring the tip alone put the stalk in a gap and then hung
            # the flowers off it onto the blade beside it.
            s = 0.0
            for q in (0.55, 0.78, 1.0):
                qx, qy = x + math.cos(sa) * sl * q, y + math.sin(sa) * sl * q
                d = min((math.hypot(qx - px, qy - py) for px, py in pool),
                        default=1e9)
                # a flower wants its own patch of paper, about a sixth of a
                # leaf-length across — but not the whole sheet: a spike that
                # clears every blade by a mile has left the subject.
                s += min(d, L * 0.20) - 0.55 * max(0.0, d - L * 0.26)
            if best_s is None or s > best_s:
                best, best_s = (sa, sl), s
        if best is None:
            continue
        sa, sl = best
        spike(cv, rng, x, y, sa, sl, st, air,
              n=rng.randint(2, 4) if k > 0.9 else rng.randint(1, 2),
              lean=lean, k=0.86 * air["k"])
    return spines


# --------------------------------------------------------------------------
# 题款 — titles, dates, seals
# --------------------------------------------------------------------------

TITLE_A = ["幽兰", "春兰", "香祖", "素心", "空谷", "石畔", "九畹", "楚畹",
           "芳馨", "清芬", "王者香", "露兰", "风兰", "月下", "兰石", "秋兰"]
TITLE_B = ["图", "小景", "清趣", "写意", "图卷", "册页", "小品"]
ACTION = ["写生", "戏作", "写意", "拟古", "画兰", "南窗写兰", "客窗戏墨",
          "并记"]
SEAL_WORDS = ["幽香", "空谷", "九畹", "香祖", "墨缘", "写心", "素心", "芳洁",
              "清芬", "兰友", "一枝", "君子"]


def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into orchid_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS
             + [v["cn"] for v in SCENES.values()]
             + [v["cn"] for v in AIR.values()]
             + [v["cn"] for v in STYLES.values()]
             + list(SCENES) + list(AIR) + list(STYLES)
             + [TIANGAN, DIZHI, "日月年写于图兰春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    """Prefer a blob baked for this vocabulary; fall back to the other
    painters' (the family shares most characters), then to no 题款."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("orchid_font", "bamboo_font", "gongbi_font", "shanshui_font"):
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

    Same machinery as the other painters. 兰 needs it more than most: a clump
    is a small dense thing in a large empty sheet, and the column of writing
    is the counterweight that makes the emptiness deliberate.
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
    r0 = min(W, H) * 0.16
    for fy in (0.34, 0.52, 0.70):
        occ.add(W * 0.5, H * fy, r0)
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.6)
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

def compose(cv, rng, scene_key, air_key, style_key, stat=None):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc = SCENES[scene_key]
    air = dict(AIR[air_key])
    st = STYLES[style_key]
    gb.set_light(rng)
    ground = gb.paint_ground(cv, rng)
    occ = Occupancy(W, H)

    # 虚实 — one half of the sheet carries the plant, the other the writing.
    # The wind blows *into* the empty half: blowing the other way it lays a
    # clump that is already against its border over the border, and the sheet
    # loses the leaves and keeps the emptiness.
    side = rng.choice((0, 1))
    wind = 1.0 if side else -1.0
    lean = air["sweep"] * wind * rng.uniform(0.85, 1.15)

    if air.get("veil"):
        veil(cv, air["veil"][0], air["veil"][1])
    if air.get("moon"):
        mr = S * rng.uniform(0.10, 0.15)
        mx = W * (rng.uniform(0.66, 0.84) if side else rng.uniform(0.16, 0.34))
        my = H * rng.uniform(0.12, 0.28)
        moon(cv, rng, mx, my, mr, dilute(I("dan"), 0.75))
        occ.add(mx, my, mr * 1.1)

    L = S * rng.uniform(*sc["Lf"])
    hang = bool(sc.get("hang"))
    rock_top = None
    roots_xy = []

    if sc.get("ledge"):
        # 崖兰 — the cliff comes in at the top on the subject's side, and the
        # clump hangs off its underside. Everything about the plant is
        # upside down: the blades leave the root nearly horizontally and the
        # long ones fall past the vertical.
        # 崖要大 — but 大 is weight, not area, and the two were traded the
        # wrong way round. At a lip a third of the way down the sheet the
        # mass is five hundred pixels deep, and five hundred pixels of a wash
        # that has to stay out of "dirty" is a fog bank with a drawn line
        # under it: every mark inside it — crease, crack, 皴, the cut face —
        # then stands on paper, because the wash it was supposed to sit in
        # measures two levels. Short the mass to a quarter of the sheet and
        # the same ink per pixel adds up to stone. The rock still outweighs
        # the clump; it does it with tone instead of acreage.
        ey = H * rng.uniform(0.21, 0.29)
        eh = H * rng.uniform(0.26, 0.34)
        sd = -1.0 if side else 1.0
        edge = ledge(cv, rng, sd, ey, eh, st, k=0.95 * air["k"])
        # 根从石罅出 — the plant comes out of a crack, which means out of the
        # lowest part of the underside it can find, and on the open side of
        # the mass rather than back in the corner: the blades have to fall
        # into the empty half of the sheet, not down the border. That point
        # is now a *corner* of the silhouette rather than the bottom of a
        # sag, which is why the plant finally looks attached to something.
        ne = len(edge)
        band = edge[int(ne * 0.40):int(ne * 0.86)]
        rx, ry = max(band, key=lambda p: p[1])
        roots_xy.append((rx, ry + eh * 0.02, 1.0, 1.0 if side else -1.0))
        # the whole mass is occupied, in four bites along its span, so the
        # 题款 is kept off the rock instead of merely off its corner
        for q in range(4):
            ex_, ey_ = edge[int((ne - 1) * (0.12 + 0.25 * q))]
            occ.add(ex_, ey_ * 0.52, max(W * 0.10, eh * 0.30))
    else:
        bx = W * (rng.uniform(0.26, 0.46) if side else rng.uniform(0.54, 0.74))
        by = H * rng.uniform(0.72, 0.84)
        if sc["rock"] and rng.random() < sc["rock"]:
            # 石 — a 坡石 at the foot of the sheet, set *beside* the clump on
            # the side the blades do not run, so it anchors the corner the
            # leaves are leaving. Centred under the clump instead (the first
            # arrangement) the orchid stands on top of the mass with its
            # root at the summit, which is a plant in a tub.
            roff = -1.0 if side else 1.0
            rw = W * rng.uniform(0.34, 0.52)
            rcx = bx + roff * rw * rng.uniform(0.34, 0.52)
            rcx = max(rw * 0.30, min(W - rw * 0.30, rcx))
            ry = H * rng.uniform(0.90, 0.99)
            rh = H * rng.uniform(0.11, 0.17)
            rock_top = stone(cv, rng, rcx, ry, rw, rh, st,
                             k=0.80 * air["k"], moss=0.9)
            occ.add(rcx, ry - rh * 0.4, max(rw, rh) * 0.52)
            # 兰生石畔, or on the shoulder of the stone — both are painted,
            # and the one thing that is not is the root at the very top of
            # the crest in the middle of the mass.
            if rock_top is not None and rng.random() < 0.45:
                n = len(rock_top)
                rx0 = rcx - rw * 0.5
                want = min(W * 0.78, max(W * 0.22, bx))
                j = int(max(0.0, min(float(n - 1), want - rx0)))
                j = max(2, min(n - 3, j))
                by = rock_top[j] + rh * 0.03
                bx = rx0 + j
            else:
                by = H * rng.uniform(0.80, 0.88)
        roots_xy.append((bx, by, 1.0, 1.0 if side else -1.0))
        if sc["clumps"] > 1:
            # 一聚一散 — the companion is smaller, paler and set *along* the
            # ground rather than beside the main clump at the same height;
            # two clumps of a size at one level is a pair of shrubs in a
            # border, which is gardening, not painting.
            dx = W * rng.uniform(0.18, 0.34) * (1.0 if side else -1.0)
            roots_xy.append((min(W * 0.92, max(W * 0.08, bx + dx)),
                             min(H * 0.95, by + H * rng.uniform(-0.03, 0.09)),
                             rng.uniform(0.52, 0.72), -1.0 if side else 1.0))
        if rng.random() < (sc["bank"] if rock_top is None else 0.85):
            for rx_, ry_, kf, _b in roots_xy:
                bank(cv, rng, rx_ + L * rng.uniform(-0.12, 0.12),
                     ry_ + L * rng.uniform(0.01, 0.05),
                     L * rng.uniform(0.68, 1.05) * kf, st,
                     k=0.45 * air["k"], moss=1.0)

    # --- the clumps, 远淡近浓 ---------------------------------------------
    roots_xy.sort(key=lambda r: r[2])              # the small one first
    for rx, ry, kf, bias in roots_xy:
        clump(cv, rng, rx, ry, L * kf, st, air,
              n=max(4, int(rng.randint(*sc["nl"]) * (0.75 + 0.25 * kf))),
              # every clump leans a little even in dead air: a plant
              # standing exactly upright out of exactly level ground is a
              # diagram
              lean=lean + rng.uniform(-0.13, 0.13), bias=bias, hang=hang,
              k=(1.0 if kf > 0.9 else 0.70) * air["k"],
              spikes=(rng.randint(*sc["spikes"]) if kf > 0.9
                      else (1 if rng.random() < 0.5 else 0)),
              occ=occ, stat=stat)

    if air.get("fly"):
        # 风 — a petal or a torn blade tip already travelling. One or two,
        # downwind of the clump, lying the way the wind lies.
        flt, fdp = plane_tone(st, "flower", 0.70 * air["k"])
        rx, ry, _k, _b = roots_xy[-1]
        px, py = rx + L * 0.5 * wind, ry - L * rng.uniform(0.3, 0.7)
        for i in range(rng.randint(*air["fly"])):
            px += L * rng.uniform(0.10, 0.22) * wind
            py -= L * rng.uniform(-0.02, 0.12)
            if not (L * 0.05 < px < W - L * 0.05
                    and L * 0.05 < py < H - L * 0.05):
                break
            ll = L * rng.uniform(0.07, 0.13)
            stroke(cv, rng, px, py,
                   (0.0 if wind > 0 else math.pi) + rng.uniform(-0.5, 0.5),
                   ll, ll * rng.uniform(0.09, 0.16), flt, fdp,
                   alpha=rng.uniform(0.45, 0.70), turn=rng.uniform(-0.8, 0.8),
                   belly=0.4, taper=1.3, root=0.2, dens=0.8, fade=0.3,
                   feather=0.9, core=0.28)
            occ.add(px, py, ll * 0.5)

    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))]) if rng.random() < 0.66 \
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
        description="Generate random Chinese orchid paintings (墨兰).")
    ap.add_argument("-o", "--out", default="orchid.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="幽兰|双丛|兰石|崖兰")
    ap.add_argument("--air", help="静|临风|带露|月下")
    ap.add_argument("--style", help="墨兰|素心|青兰|紫蕊")
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
            print("  %-6s %s" % (k, v["en"]))
        print("花 (the blossom): 五瓣一舌, 点心如品 — "
              + "  ".join("%+.2f" % p[0] for p in PETALS))
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1300")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("墨兰 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.air, a.style)
    return 0


if __name__ == "__main__":
    sys.exit(main())
