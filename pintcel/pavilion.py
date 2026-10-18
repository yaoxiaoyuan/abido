#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pavilion.py — 随机生成中国亭景图 / Random Chinese pavilion scenery, close up.

shanshui.py already paints a 亭, but it paints it the way a landscape painter
does: twenty pixels wide on a bank across the water, four posts and a swept
roof, and the whole thing is a mark that says *somebody comes here*. This
module paints the other picture — the one where the 亭 is the subject and the
mountains are the setting. 松亭, 水亭, 山亭: the 亭 at arm's length, close
enough that the tile courses, the railing and the upturned corners are all
drawn, with somebody standing at the railing (倚栏).

That change of distance forces one genuinely new thing, and it is not a
brush: it is a *projection*. Everything else in these four painters is drawn
flat — a mountain is a height field, a leaf is a field in its own frame, a
fish is a stroke — because nothing in them has square corners. A building
does. The moment you draw a roof you have to decide what happens to the
parallel edges of a box, and Chinese painting answers that question
differently from Western painting: 界画 uses **parallel projection**. The
eaves of a 亭 in a Song album leaf do not converge on a vanishing point;
they stay parallel for ever and the depth axis simply runs off at a constant
angle. So this module carries an `Oblique` — eight lines of arithmetic — and
every point of the building is placed in (X, Y, Z) and projected through it.

    斜测   parallel projection: no vanishing point, ever (§Oblique)
    台基   the stone platform, and the steps up to it
    柱     columns on an n-gon — 四角亭 and 六角亭
    枋     the beams, and 挂落 fretwork hanging under them
    栏杆   the railing: 寻杖, 望柱, 棂条, and a bench
    攒尖   the roof as a *ruled surface* — the tile courses and the wash are
           the same family of curves, which is why they agree (§_slope)
    翘角   the corners that turn up, which is the whole silhouette
    宝顶   the finial
    山水   and then the setting, borrowed wholesale from shanshui.py
    题款   inscription, seals, and a name plaque over the entrance

Infrastructure comes from gongbi.py (PNG, paper, brushes, seals, glyphs) and
the landscape from shanshui.py (ridges, 皴, trees, mist, water, geese). This
module adds the carpentry.

Usage
-----
    python3 pavilion.py                          # one random pavilion
    python3 pavilion.py -n 6
    python3 pavilion.py --scene 水亭 --plan 六角
    python3 pavilion.py --style 浅绛 --seed 7
    python3 pavilion.py --size 900x1200 -o ting.png
    python3 pavilion.py --list
"""

import argparse
import math
import os
import random
import sys
import time

import gongbi as gb
from gongbi import (Canvas, rgb, mix, jitter, disc, brush_seg, brush_path,
                    bezier, fill_contours, paint_column, paint_seal, Occupancy,
                    BrushFont, TIANGAN, DIZHI)
import shanshui as ss
from shanshui import (I, Grain, ridge, Ridge, fbm, tree, mist,
                      water_lines, shore_rocks, waterfall, foot_lobes, geese,
                      birds, STYLES, CUN, PROF)

TAU = math.pi * 2.0


# --------------------------------------------------------------------------
# 斜测 — parallel projection
# --------------------------------------------------------------------------

class Oblique:
    """The projection 界画 draws buildings in: parallel, not perspective.

    World axes are X right, Y *away* from the viewer, Z up. A point lands on
    paper at

        sx = ox + X + Y·cx
        sy = oy - Z - Y·cy

    so the depth axis runs off up-and-to-the-right at a fixed angle and a
    fixed scale, and two edges parallel in the world stay parallel on paper
    however far away they are. No vanishing point, no convergence, no
    foreshortening of X or Z at all. This is not an approximation to
    perspective that a Song painter would have fixed if they could — a
    pavilion drawn this way can be read off like a plan, and the whole
    pleasure of 界画 is that it is simultaneously a picture and a drawing.

    `kd` is how much the depth axis is compressed (half-scale is the usual
    choice, and it is what stops a square floor reading as a lozenge), `az`
    how steeply it climbs. A climbing depth axis means the viewer is *above*
    the eaves, which is the convention: you look down into the pavilion and
    see its floor, and the roof is spread out for you rather than seen edge
    on.
    """

    __slots__ = ("ox", "oy", "cx", "cy")

    def __init__(self, ox, oy, az=0.52, kd=0.52):
        self.ox, self.oy = ox, oy
        self.cx = math.cos(az) * kd
        self.cy = math.sin(az) * kd

    def __call__(self, X, Y, Z):
        return (self.ox + X + Y * self.cx, self.oy - Z - Y * self.cy)

    def facing(self, m):
        """How strongly a wall whose outward normal points along bearing `m`
        turns toward the viewer. Positive is visible.

        The projection direction is the kernel of the map above; toward the
        viewer it is (cx, -1, cy). Dotting that with the normal (cos m,
        sin m, 0) is this one expression, and it is the only hidden-surface
        test the module needs — a convex building has no other way to hide
        anything from itself.
        """
        return math.cos(m) * self.cx - math.sin(m)


def _ngon(n, rot, R):
    """The n corners of a regular plan, as (bearing, x, y)."""
    out = []
    for i in range(n):
        a = rot + i * TAU / n
        out.append((a, math.cos(a) * R, math.sin(a) * R))
    return out


# --------------------------------------------------------------------------
# 屋顶 — the roof
# --------------------------------------------------------------------------

def _eave(m0, m1, Rout, Ze, up, bow, u):
    """A point along one eave, at lateral fraction `u` from corner m0 to m1.

    Two things happen along an eave and both of them are the roof:

    起翘 — the eave is level over most of its run and then lifts in the last
    few feet before the corner. Graded linearly from the middle it reads as
    a shallow arch, which is a Mediterranean pediment, not a 亭; the upturn
    has to be *confined to the corner*, hence the hard exponent.

    The slight outward bow is the carpentry showing: the eave purlin is a
    timber, and a timber that long with tiles on it does not stay straight.
    """
    x0, y0 = math.cos(m0) * Rout, math.sin(m0) * Rout
    x1, y1 = math.cos(m1) * Rout, math.sin(m1) * Rout
    x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    mm = (m0 + m1) * 0.5
    b = bow * Rout * math.sin(math.pi * u)
    t = abs(u * 2.0 - 1.0)
    # Exponent 2.6 confined the whole lift to the last few pixels, so the
    # eave arrived at the corner still level and the 翘角 shot off it at
    # sixty degrees — a pair of horns stuck on either side of a flat hem.
    # At 2.1 the eave is already climbing when it reaches the corner and the
    # tip continues a curve instead of starting a new one.
    return (x + math.cos(mm) * b, y + math.sin(mm) * b,
            Ze + up * t ** 2.1)


def _ray(prj, xin, yin, zin, xo, yo, zo, k=9, bend=0.55, t0=0.0):
    """One tile course, from the inner edge of a roof ring to the eave.

    举折 — a Chinese roof is concave in profile (反宇): steep off the ridge,
    almost flat at the eave. The carpenter gets that by dividing the
    horizontal run evenly between the purlins and reducing the rise of each
    step, so the plan view of a course is a straight line while the
    elevation front-loads the whole drop. That is exactly `t` linear in XY
    and `t**0.55` in Z, and it is the one line in this module that decides
    whether the roof reads as Chinese or as a tent.
    """
    pts = []
    for j in range(k + 1):
        t = t0 + (1.0 - t0) * j / float(k)
        pts.append(prj(xin + (xo - xin) * t, yin + (yo - yin) * t,
                       zin - (zin - zo) * t ** bend))
    return pts


def _slope(cv, rng, prj, m0, m1, Rin, Zin, Rout, Ze, up, bow, ink, pale,
           lit, nray, bend=0.55, tile=True):
    """One face of a roof ring, drawn as a ruled surface.

    A tiled roof is a family of courses running from the ridge down to the
    eave, and the wash that darkens the roof lies along the same family.
    So they are the same loop: draw forty curves from the inner edge to the
    eave, all of them pale, and every third one again in ink. The fill and
    the 瓦垄 agree because there was never a chance for them to disagree —
    no polygon to scan, no edge to anti-alias, and the silhouette comes out
    exactly right at the corners where it matters most.

    `Rin = 0` collapses the inner edge to a point, which is 攒尖 — the
    pyramidal roof of nearly every 亭 ever built. `Rin > 0` is a ring, and
    that is the lower eave of a 重檐亭.
    """
    ix0, iy0 = math.cos(m0) * Rin, math.sin(m0) * Rin
    ix1, iy1 = math.cos(m1) * Rin, math.sin(m1) * Rin
    # The courses do not start at the apex. On a 攒尖 roof every one of them
    # converges on the same point, so a wash painted along them piled up into
    # a black sunburst at the top of the roof and the whole thing read as a
    # parasol. Starting them a fifth of the way out bounds the overlap, and
    # it is also true: the top of the roof is a small platform under the
    # 宝顶, not a mathematical point.
    t0 = 0.0 if Rin > 0.0 else 0.20
    for j in range(nray + 1):
        u = j / float(nray)
        ex, ey, ez = _eave(m0, m1, Rout, Ze, up, bow, u)
        ix, iy = ix0 + (ix1 - ix0) * u, iy0 + (iy1 - iy0) * u
        pts = _ray(prj, ix, iy, Zin, ex, ey, ez, 9, bend, t0)
        # 染 — the wash, laid *along* the courses, so the fill and the tile
        # lines are the same family of curves and cannot disagree.
        #
        # And laid in three passes, not one. At a single flat value the roof
        # came out a grey disc with a comb of lines on it — the largest and
        # most conspicuous shape in the picture, and the only one with no
        # 墨色 in it at all. A tiled roof is not one tone: it is pale where it
        # turns toward the sky at the ridge and it piles up at the eave, where
        # the courses crowd together, the tiles double, and the overhang is in
        # its own shade. Three passes over the outer half of the same curve
        # cost nothing (the curve is already computed) and give 1 : 1.5 : 2
        # from ridge to eave, which is what makes the surface turn.
        w = max(0.8, Rout * 0.034)
        a0 = 0.108 if lit else 0.190
        brush_path(cv, pts, w, w * 1.2, pale, a0)
        k = len(pts) - 1
        brush_path(cv, pts[k // 2:], w * 1.05, w * 1.25, pale, a0 * 0.52)
        brush_path(cv, pts[int(k * 0.76):], w * 1.1, w * 1.3, pale, a0 * 0.44)
        # 顶心 — and close the hole that `t0` leaves.
        #
        # Starting the courses a fifth of the way out stops the sunburst, and
        # the cost of it was a *lens of bare paper either side of the finial*
        # — at the top of the largest shape in the picture, which is the one
        # place on the sheet where white reads as a hole rather than as 留白.
        # A disc over it is wrong twice over (the apex is the highest point of
        # the roof, so a disc centred there hangs in the sky above the ridge,
        # and it came out a moon behind the 宝顶). The cap has to lie *on the
        # surface*, which means it has to be drawn the way the rest of the
        # surface is: along the courses. Every third one, from just clear of
        # the point — the finial's own base covers the last few pixels — at
        # half strength, so a third as many strokes converging at a third of
        # the weight never piles up.
        if t0 > 0.0 and j % 3 == 0:
            cap = _ray(prj, ix, iy, Zin, ex, ey, ez, 16, bend, 0.0)
            brush_path(cv, cap[1:4], w * 0.80, w * 0.95, pale, a0 * 0.55)
        if tile and j % 4 == 2 and 0 < j < nray:
            brush_path(cv, pts, u * 0.85, u * 1.05, ink,
                       (0.26 if lit else 0.36) * rng.uniform(0.85, 1.1))


def _eave_line(cv, rng, prj, m0, m1, Rout, Ze, up, bow, ink, w, drip=True):
    """檐口 — the edge of the roof, and the drip course under it.

    The heaviest line in the building. Everything above it is a wash and a
    comb of courses; this is the one contour the eye uses to read the shape,
    and under it 滴水 — the row of round tile ends that is the single
    detail that says *Chinese roof* at any size above about forty pixels.
    """
    pts = [prj(*_eave(m0, m1, Rout, Ze, up, bow, j / 22.0))
           for j in range(23)]
    # 中实两端虚 — and in two halves, because a long curve of one width is the
    # single most mechanical mark a brush can be asked to make. The eave is
    # the heaviest line in the building and it was also the most ruled: a
    # black ribbon of exactly constant thickness, laid over a landscape in
    # which nothing else has a constant anything. Loaded in the middle and
    # lifting toward each corner, the same curve reads as one pull of a
    # brush, and the 翘角 that continues off either end stops looking welded
    # on.
    h = len(pts) // 2
    brush_path(cv, pts[:h + 1], w * 0.76, w * 1.10, ink, 0.95)
    brush_path(cv, pts[h:], w * 1.10, w * 0.76, ink, 0.95)
    if not drip:
        return
    step = max(2, int(len(pts) / max(3.0, Rout * 0.13)))
    for i in range(0, len(pts), step):
        disc(cv, pts[i][0], pts[i][1] + w * 1.1, w * 0.62, ink,
             rng.uniform(0.55, 0.85))


def _tip(cv, prj, m, Rout, Ze, up, ink, w):
    """翘角 — the corner, turning up and out past the eave.

    嫩戗: the last timber of a corner is a separate piece scarfed on at a
    steeper angle, and it carries the eave up well past the line of the
    hip. Without it a 攒尖 roof is a cone with a scalloped hem. With it the
    roof flies, and that single flick is more of the character of the
    building than the whole tiled surface above it.
    """
    cm, sm = math.cos(m), math.sin(m)
    a = prj(cm * Rout, sm * Rout, Ze + up)
    b = prj(cm * Rout * 1.055, sm * Rout * 1.055, Ze + up + Rout * 0.055)
    c = prj(cm * Rout * 1.085, sm * Rout * 1.085, Ze + up + Rout * 0.125)
    brush_path(cv, [a, b, c], w * 1.05, w * 0.34, ink, 0.92)


def _finial(cv, rng, prj, Za, R, ink, pale):
    """宝顶 — the finial. A neck, a melon, a bead, and a spike."""
    s = R * 0.085
    base = prj(0.0, 0.0, Za)
    for k, (dz, rr, al) in enumerate(((0.0, 0.95, 0.85), (s * 1.0, 1.25, 0.95),
                                      (s * 2.0, 0.70, 0.90))):
        p = prj(0.0, 0.0, Za + dz)
        disc(cv, p[0], p[1] - s * 0.25, s * rr, ink, al)
    top = prj(0.0, 0.0, Za + s * 4.6)
    brush_seg(cv, base[0], base[1] - s * 2.4, top[0], top[1],
              max(0.7, s * 0.34), max(0.4, s * 0.14), ink, 0.88)


def roof(cv, rng, prj, n, rot, r, Zc, ink, pale, tiers, bow, u, lightm):
    """攒尖顶 — the whole roof, drawn back to front.

    Only the faces turned toward the viewer are drawn at all. A convex
    building cannot hide anything from itself except by one face standing in
    front of another, so `prj.facing` is the entire hidden-surface
    algorithm; the `+0.26` is the tilt of the roof planes, which turns a
    face a little more toward a viewer who is above the eaves.
    """
    ov = 1.46                                  # 出檐 — how far the roof flies
    Rout = r * ov
    Ze = Zc + r * 0.05
    up = Rout * 0.115                          # 起翘 at the corners
    Za = Zc + r * (1.00 if tiers == 1 else 1.52)
    lw = u * 2.45                              # 檐口, the heaviest line

    rings = []
    if tiers == 1:
        rings.append((0.0, Za, Rout, Ze, up, 0.55, True))
    else:
        # 重檐 — the lower roof is a *skirt* around a short upper storey, so
        # its inner edge is a ring and not a point, and it is shallower. A
        # skirt pitched like the cap above it reaches higher than the storey
        # it is supposed to leave room for, and the building loses its waist.
        Rs = r * 1.62
        rings.append((r * 0.62, Zc + r * 0.40, Rs, Zc - r * 0.04,
                      Rs * 0.115, 0.78, True))
        Ze2 = Zc + r * 0.76
        Ru = r * 1.02
        rings.append((0.0, Za, Ru, Ze2, Ru * 0.125, 0.55, True))

    def storey():
        """平座 — the short upper storey, seen in the gap between the two
        eaves. Drawn between the rings, not before them: painted first it
        ends up under the skirt's wash and the upper roof floats on the lower
        one with nothing holding it there.

        As a storey and not as posts. One isolated upright per bay, standing
        on the skirt's own surface, read as a stovepipe — the gap between two
        eaves of a 重檐亭 is a little lantern of walls and lattice, and it
        takes a sill and a lintel to say so. They also hide the feet of the
        uprights behind the skirt's inner edge, which is where they belong.
        """
        Rp = r * 0.62
        z0, z1 = Zc + r * 0.14, Zc + r * 0.74
        for i in range(n):
            m0 = rot + i * TAU / n
            m1 = m0 + TAU / n
            if prj.facing((m0 + m1) * 0.5) <= 0.0:
                continue
            x0, y0 = math.cos(m0) * Rp, math.sin(m0) * Rp
            x1, y1 = math.cos(m1) * Rp, math.sin(m1) * Rp
            for z, w, al in ((z0, lw * 0.5, 0.46), (z1, lw * 0.62, 0.70)):
                a, b = prj(x0, y0, z), prj(x1, y1, z)
                brush_seg(cv, a[0], a[1], b[0], b[1], w, w, ink, al)
            for xx, yy in ((x0, y0), (x1, y1)):
                a, b = prj(xx, yy, z0), prj(xx, yy, z1)
                brush_seg(cv, a[0], a[1], b[0], b[1], lw * 0.6, lw * 0.5,
                          ink, 0.68)
            for t in (0.34, 0.66):               # 格窗 — a mullion or two
                a = prj(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z0)
                b = prj(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z1)
                brush_seg(cv, a[0], a[1], b[0], b[1], u * 0.85, u * 0.85,
                          pale, 0.34)

    for ri, (Rin, Zin, Rout_, Ze_, up_, bend, tile) in enumerate(rings):
        if ri == 1:
            storey()
        faces = []
        for i in range(n):
            m0 = rot + i * TAU / n
            m1 = m0 + TAU / n
            mm = (m0 + m1) * 0.5
            if prj.facing(mm) + 0.26 <= 0.0:
                continue
            faces.append((math.sin(mm), m0, m1, mm))
        faces.sort(reverse=True)               # far faces first
        nray = max(8, int(Rout_ * 0.34))
        for _, m0, m1, mm in faces:
            lit = prj.facing(mm) > 0.35 or math.cos(mm - lightm) > 0.2
            _slope(cv, rng, prj, m0, m1, Rin, Zin, Rout_, Ze_, up_, bow,
                   ink, pale, lit, nray, bend, tile)
        for _, m0, m1, mm in faces:
            _eave_line(cv, rng, prj, m0, m1, Rout_, Ze_, up_, bow, ink,
                       lw, drip=Rout_ > r * 1.1)
            # 垂脊 — the hip, a raised capped ridge of tile running from the
            # apex to the corner. Drawn after the courses so it sits on top
            # of them, and heavier than any of them.
            for m in (m0, m1):
                ex, ey, ez = _eave(m0, m1, Rout_, Ze_, up_, bow,
                                   0.0 if m == m0 else 1.0)
                pts = _ray(prj, math.cos(m) * Rin, math.sin(m) * Rin, Zin,
                           ex, ey, ez, 10, bend)
                brush_path(cv, pts, lw * 0.95, lw * 0.80, ink, 0.90)
            for m in (m0, m1):
                _tip(cv, prj, m, Rout_, Ze_, up_, ink, lw)
    _finial(cv, rng, prj, Za, r, ink, pale)


# --------------------------------------------------------------------------
# 木作 — the carpentry below the eaves
# --------------------------------------------------------------------------

def unit(r):
    """笔 — the one unit of line weight in the building, and the reason the
    whole of it looks drawn by one hand.

    This arrived late and it was the single biggest thing wrong with the
    module. The roof, the columns and the finial were all sized off `r`; the
    beams, the 挂落, the railing, the platform, the steps, the plaque and the
    figure were all pinned at literal widths between 0.5 and 1.1 px, written
    while the sheet happened to be 600 wide. Nothing looks wrong about that
    at 600. At the default 900 the near columns are nine pixels of near-black
    and the 棂条 beside them are still 0.55, and at 1400 the ratio is 12:1 —
    so the building stops being a building and becomes *a heavy black roof
    resting on a wire diagram*, against a landscape every mark of which
    scales. A painter does not change brushes between the eave and the
    railing; the hand that drew one drew the other, and the weights it leaves
    are a short ladder with a fixed spacing.

    So: one unit, and every weight in the carpentry is a multiple of it. The
    ladder, top to bottom, is

        檐口 2.45   the eave — the heaviest line in the picture
        柱   4.5    (a column is a mass, not a line, and gets its own figure)
        额枋 1.7    the beam
        台基 1.6    the near edge of the platform
        寻杖 1.35   the handrail
        雀替 1.05   the brace
        棂条 0.78   the balusters
        铺地 0.7    the paving course

    The floor matters as much as the slope: below about half a pixel
    `brush_seg`'s falloff is most of the stroke, so a line asked to be 0.3 px
    wide is not a fine line, it is a grey smudge in the shape of a line.
    """
    return max(0.58, r * 0.0165)


def _frieze(cv, rng, prj, m0, m1, r, Zb, ink, depth, u):
    """挂落 — the fretwork that hangs under the beam of each bay.

    A 亭 is a roof on posts and nothing else, so the only ornament it owns
    is this: a shallow pierced board filling the top of each opening. It is
    also what stops the bay reading as a bare rectangle, and bare
    rectangles are what make a drawn building look like a diagram.
    """
    x0, y0 = math.cos(m0) * r, math.sin(m0) * r
    x1, y1 = math.cos(m1) * r, math.sin(m1) * r
    # 雀替 — a bracket in each top corner of the bay, and nothing across the
    # middle. Two earlier versions ran a member the whole width: a deep one
    # was a hammock slung between the columns, and even a shallow one read
    # as a sagging wire, because a single curve spanning an opening is
    # exactly what a wire looks like. The real joinery at this scale *is*
    # the corner: a short cyma brace where post meets beam, which stiffens
    # the frame and tells the eye the two are joined.
    for u0, sd in ((0.0, 1.0), (1.0, -1.0)):
        bx = x0 + (x1 - x0) * u0
        by = y0 + (y1 - y0) * u0
        dx, dy = (x1 - x0) * sd, (y1 - y0) * sd
        f = depth / max(1e-6, math.hypot(dx, dy)) * 1.5
        a = prj(bx, by, Zb - depth * 1.35)
        b = prj(bx + dx * f * 0.42, by + dy * f * 0.42, Zb - depth * 0.42)
        c = prj(bx + dx * f, by + dy * f, Zb)
        brush_path(cv, [a, b, c], u * 1.05, u * 0.72, ink, 0.60)


def _rail(cv, rng, prj, m0, m1, r, Zc, ink, pale, front, bench, u):
    """栏杆 — 寻杖 on top, 地枋 at the foot, 望柱 at the ends, 棂条 between.

    The railing is where the scale of the building lives. A roof can be any
    size; a handrail is always about waist height, so the moment one is
    drawn the pavilion acquires a *person* standing at it, and everything
    else in the picture is measured against that.
    """
    x0, y0 = math.cos(m0) * r, math.sin(m0) * r
    x1, y1 = math.cos(m1) * r, math.sin(m1) * r
    zt, zb = Zc * 0.34, Zc * 0.055
    # A bay the viewer is looking through the back of gets a quarter of the
    # weight. At equal weight the far railing reads straight through the near
    # one and the pavilion comes out as an x-ray of itself — which is the one
    # failure mode of drawing a building with no z-buffer.
    k_al = 1.0 if front else 0.20
    a0, a1 = prj(x0, y0, zt), prj(x1, y1, zt)
    b0, b1 = prj(x0, y0, zb), prj(x1, y1, zb)
    # 寻杖 is a round timber and it swells a little along its length; 地枋 is
    # a flat plank and does not. Two lines of the same length and the same
    # width, parallel, are a pair of rails on a drawing — the taper is what
    # makes the upper one the one you would put your hand on.
    brush_seg(cv, a0[0], a0[1], a1[0], a1[1], u * 1.45, u * 1.18,
              ink, 0.80 * k_al)
    brush_seg(cv, b0[0], b0[1], b1[0], b1[1], u * 1.0, u * 1.0,
              ink, 0.62 * k_al)
    # 棂条 — a full comb on the near bays, and two on the far ones. A quarter
    # weight is enough to subdue two long lines but not a comb of them: at the
    # default sheet size the far railings put five uprights and a plank across
    # the middle of a floor that was bare paper, and the interior read as a
    # wire crate. Dropping them entirely swapped one defect for another,
    # though — two rails with nothing between them and pale columns at their
    # ends are a pair of wires strung across the room, which is worse than a
    # railing drawn too faintly. Two uprights are what make the pair of lines
    # a railing, and a painter does exactly that: the far side of the balustrade
    # indicated with a post or two and left there.
    k = (4 if r < 70 else 5) if front else 3
    for j in range(1, k):
        t = j / float(k)
        p = prj(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, zt)
        q = prj(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, zb)
        brush_seg(cv, p[0], p[1], q[0], q[1], u * 0.78, u * 0.78,
                  ink, 0.36 * k_al)
    if bench and front:
        # 座凳栏杆 — the rail doubles as a bench, which is what a 亭 is for.
        # One plank, parallel to the rail and set in from it; the first
        # version drew the two end brackets as well and they came out as
        # struts poking through the columns. On a far bay it was a long pale
        # diagonal straight across the middle of the floor — the one mark in
        # the interior with no visible cause, since the railing it belongs to
        # is almost invisible; hence `front` above.
        d0 = prj(x0 * 0.78, y0 * 0.78, zt * 0.80)
        d1 = prj(x1 * 0.78, y1 * 0.78, zt * 0.80)
        brush_seg(cv, d0[0], d0[1], d1[0], d1[1], u * 1.0, u * 1.0,
                  pale, 0.52 * k_al)


def _base(cv, rng, prj, n, rot, r, ink, pale, th, steps_at, u, lightm):
    """台基 — the stone platform, and 踏步 up to it.

    Nothing in this building touches the ground except the platform, and the
    platform is the reason the 亭 looks *placed* rather than parked. It is
    also the only part drawn as a solid: a slab has three visible faces and
    the top one is the floor.
    """
    Rb = r * 1.26
    ring = _ngon(n, rot, Rb)
    top = [prj(x, y, 0.0) for _, x, y in ring]
    # 地面 — the floor, in a tone so light it is barely a tone at all. It is
    # the thing that makes the inside of the 亭 an *inside*: with the floor
    # left as bare paper, every pale line that crosses it — the far railings,
    # the paving course, the far edges of the slab — reads as a wire hanging
    # in air, and the interior goes back to being a crate however carefully
    # those lines are weighted. Over a surface they read as marks *on* a
    # floor. Lighter than the skirt below it, because it faces the sky.
    fill_contours(cv, [top], pale, 0.130)
    # The outline of the platform, edge by edge rather than as one closed
    # path: the far edges are behind the building and have to be pale, or
    # the floor reads as the top of a glass box. Drawn at one weight all
    # round, this was the single thing that made the first renders look like
    # a wireframe crate with a roof on it.
    for i in range(n):
        a, b = top[i], top[(i + 1) % n]
        mm = rot + i * TAU / n + math.pi / n
        near = prj.facing(mm) > 0.0
        w = u * (1.6 if near else 0.95)
        brush_seg(cv, a[0], a[1], b[0], b[1], w, w,
                  ink, 0.75 if near else 0.22)
    # 铺地 — the paving. A concentric course, never chords: chords through
    # the middle of the floor are the diagonals of a box and the eye reads
    # them as such immediately.
    inner = [prj(x * 0.74, y * 0.74, 0.0) for _, x, y in ring]
    brush_path(cv, inner + inner[:1], u * 0.7, u * 0.7, pale, 0.26)
    # the front skirt: only the edges whose outward normal faces the viewer
    for i in range(n):
        m0 = rot + i * TAU / n
        mm = m0 + math.pi / n
        if prj.facing(mm) <= 0.0:
            continue
        x0, y0 = math.cos(m0) * Rb, math.sin(m0) * Rb
        x1, y1 = math.cos(m0 + TAU / n) * Rb, math.sin(m0 + TAU / n) * Rb
        a, b = prj(x0, y0, 0.0), prj(x1, y1, 0.0)
        c, d = prj(x0, y0, -th), prj(x1, y1, -th)
        # The face itself. Outline alone was enough while the 亭 stood on bare
        # paper; on a washed hillside an outlined slab is a slab of glass, and
        # the stair laid over it read as a wireframe lying in the grass. A
        # stone needs a body.
        #
        # It needed more of one than it had. At one flat 0.165 over paper the
        # module had already given back (§_reserve), the 台基 came out *whiter
        # than the hillside it stands on* — a bright box with hairlines round
        # it, which is the one shape in the picture that looks printed. Two
        # things fix it and neither is simply more ink:
        #
        # 面有向背 — the faces of a plinth are at different angles to the
        # light and a stone with two tones on it is a stone, while a stone
        # with one is a card. `lightm` is where the light comes from and it is
        # the same bearing the roof is shaded by, so the building agrees with
        # itself.
        #
        # 下重上轻 — and the wash runs heavier toward the foot. A plinth is
        # dirty where the ground splashes it and it is in its own shadow
        # there, and the gradient is also what stops the bottom edge being a
        # ruled line with paper on both sides of it.
        sh = 0.5 - 0.5 * math.cos(mm - lightm)        # 0 lit, 1 in shade
        al = 0.215 + 0.215 * sh
        ns = max(6, int(abs(b[0] - a[0]) / 1.4))
        for t in range(ns + 1):
            v = t / float(ns)
            p = (a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v)
            q = (c[0] + (d[0] - c[0]) * v, c[1] + (d[1] - c[1]) * v)
            brush_seg(cv, p[0], p[1], q[0], q[1], u * 1.0, u * 1.0, pale, al)
            mz = (p[0] + (q[0] - p[0]) * 0.52, p[1] + (q[1] - p[1]) * 0.52)
            brush_seg(cv, mz[0], mz[1], q[0], q[1], u * 1.0, u * 1.0,
                      pale, al * 0.62)
        brush_seg(cv, c[0], c[1], d[0], d[1], u * 1.9, u * 1.9, ink, 0.80)
        brush_seg(cv, a[0], a[1], c[0], c[1], u * 1.1, u * 1.1, ink, 0.62)
        brush_seg(cv, b[0], b[1], d[0], d[1], u * 1.1, u * 1.1, ink, 0.62)
        # 石纹 — a few courses of stone, and a dry scrub for the shadow. Not
        # on the face the stair comes down: in oblique projection a flight
        # descending in front of a wall always lands on that wall's screen
        # area, so any mark already there ends up crossing the treads and the
        # pair reads as scribble rather than as one thing in front of another.
        if steps_at is not None and abs(m0 - steps_at) < 1e-6:
            continue
        for _ in range(rng.randint(2, 4)):
            t = rng.uniform(0.15, 0.85)
            z = -th * rng.uniform(0.30, 0.85)
            p = prj(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z)
            q = prj(x0 + (x1 - x0) * min(1.0, t + rng.uniform(0.15, 0.45)),
                    y0 + (y1 - y0) * min(1.0, t + rng.uniform(0.15, 0.45)), z)
            brush_seg(cv, p[0], p[1], q[0], q[1], u * 0.85, u * 0.7, pale,
                      rng.uniform(0.25, 0.5))
    if steps_at is None:
        return
    # 踏步 — three steps down from the entrance bay.
    #
    # Four versions of this failed, three of them by adding marks. Bare
    # nosings were a barcode; nosings plus cheek lines were a doormat;
    # risers plus hatching were an open crate. The last and most useful
    # failure was a clean outlined flight that still read as flat, and
    # measuring it said why: the run needed to clear the platform's own front
    # face on paper is about four and a half times the platform height, which
    # is a staircase, not a stoop. So the flight *will* overlap that face,
    # and the only thing that can put it in front is tone. The treads and
    # risers get a wash; the face behind them is left as bare paper. Two
    # values, and the stair stands clear.
    m0 = steps_at
    m1 = m0 + TAU / n
    mm = (m0 + m1) * 0.5
    px, py = math.cos(mm + math.pi * 0.5), math.sin(mm + math.pi * 0.5)
    cx, cy = math.cos(mm), math.sin(mm)
    nst = 3
    run = th * 0.46
    lu = u                      # `P` below takes a step fraction called `u`

    def P(u, k, z):
        # A flight of steps is wider at the bottom than at the top, and that
        # flare is the one piece of information the platform does not have.
        hw = Rb * (0.17 + 0.10 * u)
        d = Rb + run * nst * u
        return prj(cx * d + px * hw * k, cy * d + py * hw * k, z)

    def ln(a, b, w, al):
        brush_seg(cv, a[0], a[1], b[0], b[1], w, w, ink, al)

    def wash(u0, z0, u1, z1, al):
        a0, b0 = P(u0, -1.0, z0), P(u0, 1.0, z0)
        a1, b1 = P(u1, -1.0, z1), P(u1, 1.0, z1)
        ns = max(7, int(max(abs(b0[0] - a0[0]), abs(b1[0] - a1[0]))))
        for t in range(ns + 1):
            v = t / float(ns)
            p = (a0[0] + (b0[0] - a0[0]) * v, a0[1] + (b0[1] - a0[1]) * v)
            q = (a1[0] + (b1[0] - a1[0]) * v, a1[1] + (b1[1] - a1[1]) * v)
            brush_seg(cv, p[0], p[1], q[0], q[1], lu, lu, pale, al)

    for s in range(1, nst + 1):
        u0, u1 = (s - 1) / float(nst), s / float(nst)
        z, zup = -th * s / nst, -th * (s - 1) / nst
        wash(u0, zup, u0, z, 0.56)                  # the riser, in shadow
        wash(u0, z, u1, z, 0.30)                    # the tread, in light
    for s in range(1, nst + 1):                     # the nosings
        v = s / float(nst)
        z = -th * s / nst
        ln(P(v, -1.0, z), P(v, 1.0, z), lu * 1.3, 0.76 if s == nst else 0.58)
    for k in (-1.0, 1.0):                           # 垂带 — one line each side
        ln(P(0.0, k, 0.0), P(1.0, k, -th), lu * 1.15, 0.62)


def scholar(cv, rng, prj, r, Zc, ink, pale, a, rad, u, paper=None):
    """高士 — somebody in it, which is the point of a 亭. Standing at the
    railing, in profile, looking out.

    Six versions. The first five were all seated, and all five failed, in
    this order: a grey wedge with a dot on top; a chess pawn; an ant; a bell;
    a sack with a stopper in it. The proportions were right every time and
    the drawing got better every time, and none of that mattered, because a
    seated figure is a *squat* figure — about as wide as it is tall — and at
    fifty pixels a squat silhouette with a small dark head is a vessel. The
    eye has one reliable cue for `person` at this size and it is not the
    posture, the robe or the head: it is being three times taller than wide.
    So he stands. Which is also what people actually do in a 亭 — the view is
    over the railing, and 倚栏 is the oldest 点景 in the repertoire.
    """
    x, y = math.cos(a) * rad, math.sin(a) * rad
    hs = Zc * 0.66                                  # a person against a column
    sd = 1.0 if rng.random() < 0.5 else -1.0        # which way he is facing
    lu = u                      # `P` below takes a lateral fraction called `u`

    def P(u, v):
        p = prj(x + sd * u * hs, y, v * hs)
        return (p[0], p[1])

    # 背 and 前襟 — two contours, not one half-width about an axis. A body is
    # not a turned vessel and the symmetric construction is why every earlier
    # version read as one.
    back = [(-0.170, 0.000), (-0.148, 0.195), (-0.128, 0.430),
            (-0.118, 0.620), (-0.104, 0.775), (-0.048, 0.845)]
    front = [(0.190, 0.000), (0.158, 0.195), (0.136, 0.430),
             (0.146, 0.620), (0.106, 0.775), (0.048, 0.845)]

    def edge(pts, v):
        for i in range(len(pts) - 1):
            (u0, v0), (u1, v1) = pts[i], pts[i + 1]
            if v0 <= v <= v1:
                return u0 + (u1 - u0) * (v - v0) / max(1e-6, v1 - v0)
        return pts[-1][0] if v > pts[-1][1] else pts[0][0]

    # 留白 — his own paper first, the same as the furniture's (§`body`) and
    # for the same reason, only worse: he stands at the near railing, so the
    # two rails and a 棂条 or three cross him at chest height, and behind
    # those is whatever the hillside was doing. A 0.10 wash over that is not
    # a robe, it is a gauze curtain with a dark bead on top — which is what
    # he was: the handrail ran clean through him and the 皴 of the rock
    # showed inside his sleeve. A person is the one thing in a painting the
    # eye will not accept as transparent.
    # The robe only, not the head: the hull of robe-plus-head is a blunt
    # pentagon that stands a pale box out behind the shoulders, and a disc
    # at 0.90 ink needs nothing cleared under it anyway.
    _reserve(cv, paper, [P(uu, vv) for uu, vv in back + front],
             a=0.90, feather=1.2)
    # 衣 — the robe, as tone first: the body has to be a mass before a line in
    # it is worth drawing, or the dark head has nothing of its own weight to
    # sit on and the figure comes out top-heavy.
    #
    # And the tone has to be paid for twice once the paper is reserved. At
    # 0.10 over the 檐下 wash he was a dim shape in a dim interior and read
    # well enough by accident; clear the paper under him and the same 0.10 is
    # nothing at all — a wire hairpin with a bead on top, standing in a
    # white cut-out of himself. 向背 is the fix and it is free: the robe is
    # a cylinder, so it is darkest at the shaded edge and lightest a third
    # of the way in from the lit one, and three tones across forty pixels
    # are enough to turn an outline into a body.
    step = 0.55 / hs
    v = 0.0
    while v <= 0.845:
        ub, uf = edge(back, v), edge(front, v)
        p, q = P(ub, v), P(uf, v)
        ns = max(3, int(abs(q[0] - p[0]) / 0.7))
        for t in range(ns + 1):
            g = t / float(ns)                     # 0 at the back, 1 in front
            lam = 0.30 + 0.22 * math.cos((g - 0.62) * math.pi)
            x_ = p[0] + (q[0] - p[0]) * g
            y_ = p[1] + (q[1] - p[1]) * g
            brush_seg(cv, x_, y_, x_, y_ + step * hs * 1.1, lu * 0.8,
                      lu * 0.8, pale, lam * 0.22)
        v += step
    # The contours come down to meet it. A black 1.5-unit line round a mass
    # that now has its own tone is a cartoon outline, and the capsule brush
    # beads at every joint at that width.
    brush_path(cv, [P(u, w) for u, w in back], lu * 1.15, lu * 0.95,
               ink, 0.60)
    brush_path(cv, [P(u, w) for u, w in front], lu * 1.05, lu * 0.88,
               ink, 0.54)
    # 下襟 — the hem, and not level: a level hem on a vertical mass is a plinth
    brush_path(cv, [P(-0.170, 0.000), P(-0.04, -0.022), P(0.190, 0.018)],
               lu * 1.20, lu * 1.00, ink, 0.64)
    # 带 — the sash, the one line across the figure, and the thing that says
    # where the waist is. Without it the robe is a sleeve standing up by itself
    brush_seg(cv, *(P(-0.128, 0.455) + P(0.138, 0.455)),
              lu * 1.25, lu * 1.25, ink, 0.46)
    # 头 — the one dark mark on the figure, over the shoulders with a neck's
    # worth of paper under it.
    hd = P(sd * 0.0 - 0.012, 0.912)
    disc(cv, hd[0], hd[1], hs * 0.062, ink, 0.90)
    if rng.random() < 0.6:                          # 巾 — a scholar's cap
        c0, c1 = P(-0.062, 0.968), P(0.040, 0.968)
        brush_seg(cv, c0[0], c0[1], c1[0], c1[1], hs * 0.034, hs * 0.034,
                  ink, 0.74)
    # 袖 — one sleeve, hanging. It is what gives the figure an attitude; run
    # out past the contour it was a limb sticking out of the mass.
    brush_path(cv, [P(0.085, 0.755), P(0.168, 0.640), P(0.150, 0.505)],
               lu * 1.25, lu * 1.0, ink, 0.50)


def furniture(cv, rng, prj, r, ink, pale, a, u, paper=None):
    """石桌石凳 — a stone table and a stool or two, on the far side."""
    x, y = math.cos(a) * r * 0.44, math.sin(a) * r * 0.44
    s = r * 0.30

    def ring(cx, cy, rad, z, al, w=None, half=None):
        """A circle in the world is an ellipse on paper — and in parallel
        projection it is exactly an ellipse, with no foreshortening to get
        wrong. Drawn as a filled `disc` instead, a stone table came out as a
        flat grey smudge; drawn as a *small* contour it came out as a blob,
        because sixteen brush dabs round a six-pixel-tall ellipse all land on
        top of each other. So: bigger, thinner, fewer dabs, and `half` for
        the near rim alone — a slab's underside is only visible in front."""
        lo, hi = (0, 24) if half is None else half
        pts = [prj(cx + math.cos(t * TAU / 24.0) * rad,
                   cy + math.sin(t * TAU / 24.0) * rad, z)
               for t in range(lo, hi + 1)]
        ww = u * 0.85 if w is None else w
        brush_path(cv, pts, ww, ww, ink, al)

    def body(cx, cy, rad, z, al):
        """And a slab needs a body, for the same reason the platform does (see
        `_base`): a top drawn as a contour alone is a hoop of wire, and two
        hoops on a stick are not a table. It showed up only at the default
        sheet size — at 600 px the rings are small enough to read as solid,
        and at 900 the furniture was a wireframe standing in a wireframe.
        A wash alone is not quite enough, though. A wash is transparent, and
        what lies behind the furniture is the far railing: at 0.20 weight its
        two rails are faint, but they are still darker than a tone this light
        and they went straight through the table top, which is how a stone
        slab ends up looking like a pane of glass. So the stone takes its
        paper back first — the same 留白 the whole building stands in — and
        the wash goes over that."""
        pts = [prj(cx + math.cos(t * TAU / 24.0) * rad,
                   cy + math.sin(t * TAU / 24.0) * rad, z)
               for t in range(24)]
        _reserve(cv, paper, pts, a=0.72, feather=1.2)
        fill_contours(cv, [pts], ink, al)

    # Proportion, not size, was the whole problem. A top of radius s sitting
    # at 0.78·s above the floor is a disc twice as wide as it is tall — a
    # floating lid. A stone table is about as tall as it is wide, so the top
    # goes up and the radius comes down, and then the pedestal has room to be
    # seen holding it.
    tr, tz = s * 0.60, s * 1.24
    body(x, y, tr, tz, 0.22)                        # 石桌 — the table top
    ring(x, y, tr, tz, 0.50)
    ring(x, y, tr, tz - s * 0.09, 0.32, half=(12, 24))
    pa, pb = prj(x - tr * 0.40, y, 0.0), prj(x + tr * 0.40, y, 0.0)
    qb, qa = (prj(x + tr * 0.34, y, tz - s * 0.09),
              prj(x - tr * 0.34, y, tz - s * 0.09))
    _reserve(cv, paper, [pa, pb, qb, qa], a=0.72, feather=1.2)
    fill_contours(cv, [[pa, pb, qb, qa]], ink, 0.17)  # 石鼓 — a drum pedestal
    for p, q in ((pa, qa), (pb, qb)):
        brush_seg(cv, p[0], p[1], q[0], q[1], u * 1.1, u * 1.1,
                  ink, 0.50)
    ring(x, y, tr * 0.52, 0.0, 0.30, w=u * 0.7)         # its foot on the floor
    for k in (-1.0, 1.0):                           # 石凳 — a stool each side
        sx = x + math.cos(a + 1.57) * s * 1.30 * k
        sy = y + math.sin(a + 1.57) * s * 1.30 * k
        body(sx, sy, s * 0.30, s * 0.70, 0.19)
        ring(sx, sy, s * 0.30, s * 0.70, 0.42, w=u * 0.78)
        ring(sx, sy, s * 0.30, s * 0.62, 0.26, w=u * 0.7, half=(12, 24))
        p, q = prj(sx, sy, 0.0), prj(sx, sy, s * 0.64)
        brush_seg(cv, p[0], p[1], q[0], q[1], max(1.0, s * 0.16),
                  max(1.0, s * 0.16), ink, 0.34)


def moored_boat(cv, rng, x, y, s, ink, pale):
    """舟 — a small boat moored in the near water, `s` long overall.

    Not `shanshui.boat()`, and for exactly the reason this module exists at
    all: a mark designed for the middle distance does not survive being scaled
    up. That boat is four thin outlines, which is the right drawing at fifty
    pixels — the strokes are thick enough relative to the hull to make a solid
    mark — and at three hundred it is a flat empty parallelogram lying on the
    water, read as a raft, a jetty, or nothing.

    So the near boat is built the way everything else in this module is built:
    a body of tone first, with the lines over it. The things that make it a
    boat rather than a tub are the sheer — the gunwale lifts at both ends and
    the two ends are where the keel meets it — and the 篷, because a 江南小舟
    with no awning is a canoe.
    """
    d = s * 0.078                                   # hull depth amidships
    sh = 1.0 if rng.random() < 0.5 else -1.0        # which way the bow points

    # 舷弧 — the sheer. Lifted too far it is a gondola: a 江南小舟 is nearly
    # flat for most of its length and turns up only at the ends, and a high
    # exponent on the keel is what keeps the bottom flat enough to float on.
    def gun(u):                                     # 船舷 — the gunwale
        return (x + sh * u * s, y - d * 0.40 * (2.0 * abs(u)) ** 2.0)

    def keel(u):
        p = gun(u)
        return (p[0], p[1] + d * (1.0 - (2.0 * abs(u)) ** 3.4))

    us = [-0.5 + i / 24.0 for i in range(25)]
    top = [gun(u) for u in us]
    bot = [keel(u) for u in us]
    fill_contours(cv, [top + bot[::-1]], pale, 0.30)
    brush_path(cv, bot, 1.0, 1.0, ink, 0.70)        # 船底
    brush_path(cv, top, 1.4, 1.4, ink, 0.88)        # and the sheer line
    for v in (0.34, 0.66):                          # 船板 — a plank or two
        brush_path(cv, [(p[0], p[1] + (q[1] - p[1]) * v)
                        for p, q in zip(top, bot)], 0.6, 0.6, pale,
                   rng.uniform(0.35, 0.55))
    # 舱 — the two bulkheads that divide a sampan into wells
    for u in (-0.30, 0.16):
        a, b = gun(u), keel(u)
        brush_seg(cv, a[0], a[1], b[0], b[1], 0.7, 0.7, ink, 0.45)
    # 篷 — the awning, over the stern half. Rounded, ribbed, and filled: an
    # arch of bare line over an open hull is a hoop, not a shelter.
    ua, ub = -0.38, 0.02
    ha = s * 0.118
    arch = []
    for i in range(17):
        t = i / 16.0
        u = ua + (ub - ua) * t
        p = gun(u)
        arch.append((p[0], p[1] - ha * (0.56 + 0.44 * (1.0 - (2.0 * t - 1.0)
                                                       ** 2))))
    fill_contours(cv, [arch + [gun(ub), gun(ua)]], pale, 0.40)
    brush_path(cv, arch, 1.2, 1.2, ink, 0.80)
    brush_seg(cv, arch[0][0], arch[0][1], gun(ua)[0], gun(ua)[1],
              1.0, 1.0, ink, 0.66)
    brush_seg(cv, arch[-1][0], arch[-1][1], gun(ub)[0], gun(ub)[1],
              1.0, 1.0, ink, 0.66)
    for i in (3, 6, 9, 13):                         # 篷骨 — the ribs
        u = ua + (ub - ua) * (i / 16.0)
        a, b = arch[i], gun(u)
        brush_seg(cv, a[0], a[1], b[0], b[1], 0.6, 0.6, ink, 0.40)
    # 篙 — the pole, shipped and leaning on the awning, and the painter at the
    # bow. A moored boat is 野渡无人舟自横; the one thing it keeps is its gear.
    pa = gun(ub)
    brush_seg(cv, pa[0] - sh * s * 0.10, pa[1] + d * 0.30,
              pa[0] + sh * s * 0.26, pa[1] - ha * 0.30, 1.0, 0.7, ink, 0.62)
    # and the waterline: a boat sitting *on* the water is a boat on a shelf
    for u in (-0.5, 0.5):
        p = gun(u)
        for k in range(rng.randint(2, 3)):
            w = s * rng.uniform(0.07, 0.15)
            yy = p[1] + d * rng.uniform(0.25, 0.75)
            brush_seg(cv, p[0] - w, yy, p[0] + w, yy, 0.6, 0.6, pale,
                      rng.uniform(0.30, 0.55))


def bamboo_clump(cv, rng, x, y, h, culm, leaf):
    """丛竹 — a clump of bamboo beside the 亭. 竹亭 is one of the names this
    painter can write and the scene table has always carried a probability for
    it; this is the thing that probability was waiting for.

    Nothing like `bamboo.py`'s machinery, and deliberately: at this size a
    culm is one tapered stroke two pixels wide and a leaf is a flick, so the
    whole clump is about thirty strokes. The two things that have to be right
    are that the culms are *vertical* — bamboo beside a building is a set of
    verticals answering the columns, and leaning them spoils both — and that
    the leaves hang. A bamboo leaf drawn radiating upward is a palm.
    """
    for _ in range(rng.randint(5, 9)):
        xx = x + rng.uniform(-0.5, 0.5) * h * 0.34
        hh = h * rng.uniform(0.60, 1.0)
        ln = rng.uniform(-0.07, 0.07)
        w = max(0.7, hh * 0.017)
        pts = bezier((xx, y), (xx + ln * hh * 0.25, y - hh * 0.42),
                     (xx + ln * hh * 0.65, y - hh * 0.76),
                     (xx + ln * hh, y - hh), 12)
        brush_path(cv, pts, w, w * 0.55, culm, rng.uniform(0.62, 0.85))
        for t in (0.34, 0.52, 0.68, 0.82):          # 节 — the nodes
            p = pts[int(t * (len(pts) - 1))]
            brush_seg(cv, p[0] - w * 0.9, p[1], p[0] + w * 0.9, p[1],
                      w * 0.8, w * 0.8, culm, 0.55)
        # 叶 — and they hang. Three or four per culm, which is what a clump of
        # this size has in a sketch, left it a stand of thin bare verticals:
        # reeds, or saplings. Bamboo is read off its *foliage* — a culm is only
        # a line — so the leaves have to be numerous enough and long enough to
        # make a mass at the top of the clump.
        for _ in range(rng.randint(7, 12)):
            p = pts[int(rng.uniform(0.42, 1.0) * (len(pts) - 1))]
            a = rng.uniform(-1.35, 1.35)
            L = hh * rng.uniform(0.17, 0.30)
            sgn = 1.0 if a > 0 else -1.0
            brush_path(cv, [p,
                            (p[0] + math.sin(a) * L * 0.6,
                             p[1] - math.cos(a) * L * 0.42),
                            (p[0] + math.sin(a) * L * 1.05 + sgn * L * 0.10,
                             p[1] - math.cos(a) * L * 0.42 + L * 0.34)],
                       max(0.8, L * 0.12), 0.4, leaf, rng.uniform(0.45, 0.80))


def _hull(pts):
    """Convex hull, monotone chain. The screen silhouette of a convex prism
    under a parallel projection is exactly the hull of its two end rings —
    no depth sorting, no polygon clipping."""
    pts = sorted(set((round(x, 3), round(y, 3)) for x, y in pts))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append(p)
        return out

    lo, hi = half(pts), half(pts[::-1])
    return lo[:-1] + hi[:-1]


def _reserve(cv, paper, pts, a=0.80, feather=2.0):
    """留白 — give the pavilion back its paper.

    The 亭 is drawn over rock, and rock is a wash: the platform floor came
    out with the hillside showing through it and the whole base read as a
    glass crate sitting on the slope. A painter does not paint the hill
    behind the building at all, and the stdlib equivalent of not painting it
    is to lerp the region back toward the paper that was photographed before
    any rock went down — the same trick `mist` and `waterfall` use.
    """
    if paper is None:
        return
    hull = _hull(pts)
    if len(hull) < 3:
        return
    ys = [p[1] for p in hull]
    y0, y1 = max(0, int(min(ys))), min(cv.h - 1, int(max(ys)) + 1)
    buf = cv.buf
    W = cv.w
    for y in range(y0, y1 + 1):
        xs = []
        for i in range(len(hull)):
            (ax, ay), (bx, by) = hull[i], hull[(i + 1) % len(hull)]
            if (ay <= y < by) or (by <= y < ay):
                xs.append(ax + (bx - ax) * (y - ay) / (by - ay))
        if len(xs) < 2:
            continue
        xa, xb = min(xs), max(xs)
        ia, ib = max(0, int(xa)), min(W - 1, int(xb))
        for x in range(ia, ib + 1):
            e = min(x - xa, xb - x, y - y0, y1 - y)
            al = a * min(1.0, max(0.0, e / feather))
            if al <= 0.004:
                continue
            i = (y * W + x) * 3
            for k in range(3):
                o = buf[i + k]
                buf[i + k] = int(o + (paper[i + k] - o) * al)


def _shade(cv, pts, col, a0, a1, gamma=1.6, feather=2.5, band=None):
    """A graded wash over the screen hull of a set of points: `a0` at the top
    of the region, `a1` at the bottom, eased by `gamma`.

    `band` is the pair of screen y the ramp is measured between, and it is
    needed because the hull is not what the eye calls the region. Under a
    parallel projection that climbs, the top of the column cage's hull is the
    *back* of the building — hidden behind the near beam and the roof — so a
    ramp keyed to the hull spends its whole strong end where nothing can be
    seen and arrives at the visible interior already faded out. Keyed to the
    near beam instead, the shadow is darkest exactly under the eave the
    viewer is looking in past.

    `_reserve` above takes paper back; this is the other half of the same
    idea and the module went without it far too long. Everything in the
    landscape is 染 — a mountain *is* a wash with lines in it — and the 亭 was
    drawn with nothing but lines, so however carefully those lines were
    weighted the building sat on the sheet like a diagram taped to a
    painting. One wash in the right place fixes more of that than any amount
    of work on the drawing: 檐下 — the shadow the roof holds under itself.
    A 亭 is a roof on posts with *air* under it, and air under a roof is dark.
    """
    hull = _hull(pts)
    if len(hull) < 3:
        return
    ys = [p[1] for p in hull]
    ty, by = (min(ys), max(ys)) if band is None else band
    if by - ty < 2.0:
        return
    inv = 1.0 / (by - ty)
    y0 = max(0, int(min(ys)))
    y1 = min(cv.h - 1, int(max(ys)) + 1)
    blend = cv.blend
    for y in range(y0, y1 + 1):
        xs = []
        for i in range(len(hull)):
            (ax, ay), (bx, by_) = hull[i], hull[(i + 1) % len(hull)]
            if (ay <= y < by_) or (by_ <= y < ay):
                xs.append(ax + (bx - ax) * (y - ay) / (by_ - ay))
        if len(xs) < 2:
            continue
        t = min(1.0, max(0.0, (y - ty) * inv)) ** gamma
        av = a0 + (a1 - a0) * t
        if av <= 0.004:
            continue
        xa, xb = min(xs), max(xs)
        for x in range(max(0, int(xa)), min(cv.w - 1, int(xb)) + 1):
            e = min(x - xa, xb - x) / feather
            blend(x, y, col, av * min(1.0, max(0.0, e)))


def plaque(cv, rng, prj, m0, m1, r, Zb, font, text, ink, u):
    """匾额 — the name board over the entrance.

    Every 亭 in China has a name and the name is hung in the entrance bay.
    Two characters, so the board is nearly square; drawn with the same baked
    outlines as the 题款, which means the pavilion is signed in the same
    hand as the painting.
    """
    if font is None:
        return
    mm = (m0 + m1) * 0.5
    x, y = math.cos(mm) * r * 0.98, math.sin(mm) * r * 0.98
    gsz = r * 0.19
    p = prj(x, y, Zb - r * 0.03)
    pad = gsz * 0.30
    w, h = gsz * 2.0 + pad * 3.0, gsz + pad * 2.0
    x0, y0 = p[0] - w * 0.5, p[1]
    for j in range(1, int(h)):                  # the board, washed flat
        brush_seg(cv, x0, y0 + j, x0 + w, y0 + j, 0.6, 0.6, I("qing"), 0.11)
    brush_path(cv, [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h),
                    (x0, y0)], u * 1.15, u * 1.15, ink, 0.72)
    # 横匾 — written across, not down, which is the one place in any of these
    # painters where a line of characters runs left to right.
    for i, ch in enumerate(text[:2]):
        paint_column(cv, rng, font, ch, x0 + pad + i * (gsz + pad),
                     y0 + pad, gsz, alpha=(0.70, 0.88))


def ting(cv, rng, prj, n, rot, r, ink, pale, tiers=1, bow=0.035,
         occupant=True, name=None, font=None, paper=None, lightm=0.0):
    """One 亭, drawn from the ground up and from the back forward.

    Order is everything and there is no z-buffer: the platform, then the
    bays behind, then whoever is sitting in it, then the bays in front, and
    the roof last of all because the roof is above everything and its eave
    is what crops the tops of the columns. Painted in the other order the
    pavilion turns into an x-ray of itself.
    """
    Zc = r * 1.42                              # column height, floor to beam
    u = unit(r)                                # §unit — one ladder of weights
    wc = u * 4.5                               # 柱 — a mass, not a line
    cols = _ngon(n, rot, r)

    # which bay is the entrance: the one most squarely facing the viewer
    bays = []
    for i in range(n):
        m0 = rot + i * TAU / n
        bays.append(((m0 + m0 + TAU / n) * 0.5, m0, i))
    door = max(bays, key=lambda b: prj.facing(b[0]))
    far_bay = min(bays, key=lambda b: prj.facing(b[0]))

    th = r * 0.26
    # 留白 — clear the paper the building stands in, before a line of it is
    # drawn. Down to the platform's foot, not to the floor: a stone slab with
    # nothing in it but an outline is a slab made of glass, and over a green
    # hillside wash that is exactly how it read — the 踏步 a wireframe laid on
    # grass. The reserve may take a bite out of the crest doing this, and 筑台
    # upstream is what keeps the bite small: the shelf is cut dead level under
    # the whole platform, so what the reserve removes is a few pixels of
    # hilltop and the grass at the foot covers the join.
    if paper is not None:
        Rb = r * 1.26
        sil = []
        for i in range(n):
            m = rot + i * TAU / n
            cx_, cy_ = math.cos(m), math.sin(m)
            sil.append(prj(cx_ * r * 0.975, cy_ * r * 0.975, Zc))
            sil.append(prj(cx_ * Rb, cy_ * Rb, 0.0))
            sil.append(prj(cx_ * Rb, cy_ * Rb, -th))
        _reserve(cv, paper, sil, a=0.74)
        # The 踏步 reaches a step and a half beyond the platform, outside that
        # hull, and its two tones are the only thing holding it in front of
        # the face behind it — over a hillside wash they vanished and the
        # flight went back to being a wireframe lying in the grass. Reserved
        # separately rather than by widening the hull: the hull is convex, so
        # taking the stair into it bites a wedge out of the hill on both
        # sides of the flight as well as under it.
        mm = door[1] + math.pi / n
        qx, qy = math.cos(mm + math.pi * 0.5), math.sin(mm + math.pi * 0.5)
        sx, sy = math.cos(mm), math.sin(mm)
        # (`t`, not `u`: `u` is the weight unit and everything below depends
        #  on it. Bound here to a step fraction it leaked out of the loop at
        #  1.0 and silently re-scaled the whole ladder — 额枋, 台基, 栏杆,
        #  雀替 and the entire roof — for every sheet with a `paper` to
        #  reserve against, which is every sheet `compose` renders. At the
        #  default size that is the ladder at 61 % of its weights, and the
        #  building reads as a wireframe; on a small 册页 leaf the same bug
        #  runs the other way and the marks come out half again too heavy.
        #  One letter, and it undid the one thing §unit exists to guarantee.)
        quad = []
        for t, z in ((0.0, 0.0), (1.0, -th)):
            hw, dd = Rb * (0.17 + 0.10 * t), Rb + th * 0.46 * 3 * t
            for k in (-1.0, 1.0):
                quad.append(prj(sx * dd + qx * hw * k,
                                sy * dd + qy * hw * k, z))
        _reserve(cv, paper, quad, a=0.74, feather=1.4)

    # 脚下 — where the stone meets the hill, before the stone is drawn.
    #
    # 留白 gives the building its paper back and leaves a clean edge doing
    # it, and a clean edge between two things is a cut-out: the 台基 came out
    # a white box laid on a brown slope, brighter than anything else in the
    # lower half of the sheet, with the hill passing behind it untouched.
    # Grass along the join (see `compose`) was the first attempt and it is
    # necessary but not sufficient — a dozen flicks cannot say *this object
    # is heavy and standing on that one*. What says it is tone: a hard
    # accretion of ink in the angle, fading off in a few pixels, and a
    # broader smear running away from the light. Drawn before `_base` so the
    # stone goes over its own top edge and the dark stays outside the slab.
    Rb = r * 1.26
    sdir = -1.0 if math.cos(lightm) >= 0.0 else 1.0
    for i in range(n):
        m0 = rot + i * TAU / n
        mm = m0 + math.pi / n
        if prj.facing(mm) <= 0.0:
            continue
        a = prj(math.cos(m0) * Rb, math.sin(m0) * Rb, -th)
        b = prj(math.cos(m0 + TAU / n) * Rb, math.sin(m0 + TAU / n) * Rb, -th)
        for k in range(1, 8):
            t = k / 7.0
            dy = r * 0.075 * t * t
            dx = sdir * r * 0.10 * t
            brush_seg(cv, a[0] + dx * 0.25, a[1] + dy,
                      b[0] + dx, b[1] + dy, max(0.9, r * 0.030),
                      max(0.9, r * 0.030), pale, 0.105 * (1.0 - t) ** 1.3)
        brush_seg(cv, a[0], a[1] + u * 0.8, b[0], b[1] + u * 0.8,
                  u * 1.4, u * 1.4, ink, 0.34)

    _base(cv, rng, prj, n, rot, r, ink, pale, th, door[1], u, lightm)

    # 檐下 — the shade the roof keeps under itself, before a post goes in.
    # The one wash the building needs and the last thing it got. Without it
    # the interior is the brightest region on the sheet: reserved paper, a
    # floor at 0.09, and the far columns and far railing drawn across it as
    # pale grey wires with nothing behind them to be pale *against*. With it
    # the same marks read as carpentry standing in shadow, the near columns
    # come forward because they are dark against dark-grey rather than black
    # against white, and the building acquires the one property a line
    # drawing can never have — an inside.
    void = []
    for i in range(n):
        m = rot + i * TAU / n
        cx_, cy_ = math.cos(m) * r * 0.99, math.sin(m) * r * 0.99
        void.append(prj(cx_, cy_, Zc))
        void.append(prj(cx_, cy_, 0.0))
    # Strong at the beam and almost gone at the floor: the shadow belongs to
    # the roof, not to the floor, and carried all the way down it fills the
    # 亭 with smoke and the platform loses its top. Measured between the
    # *near* beam and the near floor — see `band` in `_shade`.
    _shade(cv, void, pale, 0.40, 0.04,
           band=(prj(0.0, -r, Zc)[1], prj(0.0, -r, 0.0)[1]), gamma=1.25)

    # columns and bays, back to front. A column's depth is its own Y; a
    # bay's is the mean of the two it hangs between.
    order = sorted(range(n), key=lambda i: -math.sin(rot + i * TAU / n))
    drawn_col = set()

    def column(i):
        if i in drawn_col:
            return
        drawn_col.add(i)
        a, x, y = cols[i]
        near = math.sin(a) < 0.0
        # 侧脚 — real columns lean in a little at the top. It is a couple of
        # pixels and it is the difference between a building and a crate.
        a0 = prj(x, y, 0.0)
        a1 = prj(x * 0.975, y * 0.975, Zc)
        # A far column at three quarters the width and half the ink came out a
        # soft grey bar: at that weight `brush_seg`'s falloff is most of the
        # stroke and the column reads as a column of fog. Thinner is further.
        # 收分 — and it is thinner at the top. A Chinese column tapers about
        # a tenth over its height, and the taper is the difference between a
        # post and a length of pipe.
        brush_seg(cv, a0[0], a0[1], a1[0], a1[1],
                  wc if near else wc * 0.52, wc * 0.80 if near else wc * 0.42,
                  ink, 0.90 if near else 0.40)

    for i in order:
        a = rot + i * TAU / n
        column(i)
        column((i + 1) % n)
        m0, m1 = a, a + TAU / n
        mm = (m0 + m1) * 0.5
        if prj.facing(mm) <= 0.0:
            # a bay seen from behind: only its rail shows through the
            # building, and pale, because the near rail is in front of it
            if i != far_bay[2]:
                continue
        # 额枋 — the beam across the bay, doubled: 阑额 under 普拍枋
        p0 = prj(math.cos(m0) * r, math.sin(m0) * r, Zc)
        p1 = prj(math.cos(m1) * r, math.sin(m1) * r, Zc)
        q0 = prj(math.cos(m0) * r, math.sin(m0) * r, Zc - r * 0.085)
        q1 = prj(math.cos(m1) * r, math.sin(m1) * r, Zc - r * 0.085)
        front = prj.facing(mm) > 0.0
        al = 0.88 if front else 0.34
        brush_seg(cv, p0[0], p0[1], p1[0], p1[1], u * 1.7, u * 1.7, ink, al)
        brush_seg(cv, q0[0], q0[1], q1[0], q1[1], u * 1.2, u * 1.2,
                  ink, al * 0.8)
        if front:
            _frieze(cv, rng, prj, m0, m1, r, Zc - r * 0.085, ink, r * 0.20, u)
        if i == door[2]:
            if name:
                plaque(cv, rng, prj, m0, m1, r, Zc - r * 0.10, font, name,
                       ink, u)
        else:
            _rail(cv, rng, prj, m0, m1, r, Zc, ink, pale, front,
                  bench=front, u=u)
        if i == far_bay[2] and occupant:
            # the interior, drawn once the far side of the building is up
            if rng.random() < 0.62:
                # At the railing of a bay the viewer can see into, and not
                # the one with the plaque across it. And clear of the far
                # columns on paper: those are pale grey bars, and one of them
                # ran straight down through the figure, which no amount of
                # work on the figure could survive.
                opts = [b for b in bays
                        if prj.facing(b[0]) > 0.0 and b[2] != door[2]] or [door]
                best, ba = None, opts[0][0]
                for b in opts:
                    for _ in range(5):
                        cand = b[0] + rng.uniform(-0.28, 0.28)
                        fx = prj(math.cos(cand) * r * 0.74,
                                 math.sin(cand) * r * 0.74, 0.0)[0]
                        gap = min(abs(fx - prj(math.cos(rot + j * TAU / n) * r,
                                               math.sin(rot + j * TAU / n) * r,
                                               0.0)[0]) for j in range(n))
                        if best is None or gap > best:
                            best, ba = gap, cand
                scholar(cv, rng, prj, r, Zc, ink, pale, ba, r * 0.74, u,
                        paper)
            else:
                furniture(cv, rng, prj, r, ink, pale,
                          far_bay[0] + rng.uniform(-0.4, 0.4), u, paper)

    roof(cv, rng, prj, n, rot, r, Zc, ink, pale, tiers, bow, u, lightm)
    return Zc


# --------------------------------------------------------------------------
# 山水 — the setting, borrowed from shanshui.py
# --------------------------------------------------------------------------

def rockmass(cv, rng, role, span, hgt, base, anchor, near3, far3, grain,
             mode, per, amp, scrub, bow, scale, ink_d, H,
             peaks=(1, 3), spire=(0.5, 1.2), flat=0.0, tone=None, fade=0.0,
             texture=True, level=None, centre=False):
    """One mass of rock. A thinner version of shanshui's `mass` — this module
    needs a terrace, a cliff and a far ridge, not a whole mountain range, so
    the 石分三面 shoulder machinery is left where it belongs.

    `level` is a hook for 筑台. It is handed the finished crest, as screen
    *y* per column, before a stroke of it is drawn, and may cut a level shelf
    into it. Which is what a terrace is: masons do not find a flat hilltop,
    they make one. Searching a random crest for somewhere level enough to
    stand a building was the long way round, and it never quite worked —
    `ridge` crumples even a sheared plateau by seven per cent of the mass's
    height, so on anything tall enough to fill the bottom of the sheet there
    is no level window the width of a platform anywhere on it."""
    p = PROF[role]
    span = max(12, int(span))
    top = ridge(rng, span, hgt, peaks=rng.randint(*peaks),
                rough=rng.uniform(0.48, 0.62),
                detail=0.30 if role == "far" else 0.42,
                breaks=rng.randint(0, 1) if role == "far" else rng.randint(1, 3),
                spire=spire, flat=flat)
    h0 = hgt * 0.03
    a = 0
    while a < len(top) - 1 and top[a] <= h0:
        a += 1
    b = len(top) - 1
    while b > a and top[b] <= h0:
        b -= 1
    if b - a < 8:
        return None
    top = top[a:b + 1]
    sw = b - a
    s = max(range(len(top)), key=lambda i: top[i])
    # `anchor` normally places the summit, which is how a mountain is aimed.
    # A 台 is aimed by its middle instead: its top is about to be cut level
    # anyway, so what matters is that there is mass on both sides of the
    # building. Anchored by the summit, a ridge whose peak happened to fall
    # near one end hung most of its length off the far side of the sheet and
    # left the pavilion's x within a footprint of the rock's edge.
    x0 = int(anchor - (sw * 0.5 if centre else s))
    lo, hi = p["tone"]
    wob = fbm(rng, sw, 0.6, hgt * 0.012 + 1.5, lo=1, hi=4)
    crest = [base - v for v in top]
    if level is not None:
        level(x0, crest)
    R = Ridge(x0, crest,
              [base + hgt * 0.03 + v for v in wob],
              *(far3 if role == "far" else near3),
              tone=tone if tone is not None else rng.uniform(lo, hi),
              a0=p["a0"], a1=p["a1"], w_light=p["w_light"], w_cur=p["w_cur"],
              w_line=p["w_line"], line=p["lw"] * max(1.0, scale),
              w_crest=p["w_crest"], crest=p["crest"], w_foot=p["w_foot"],
              foot=p["foot"], cun=mode if texture else -1,
              per=per * rng.uniform(0.85, 1.20)
                  * max(0.40, min(1.25, hgt / (H * 0.26))),
              amp=amp, scrub=scrub, bow=bow, grain=grain, fade=fade)
    under = (bytes(cv.buf) if role != "far" and base < H - 2
             and hgt > H * 0.05 else None)
    R.render(cv)
    if under is not None:
        foot_lobes(cv, rng, under, R, ink_d, scale, strength=0.9)
    if texture:
        R.cun_strokes(cv, rng, int(sw * hgt / (900.0 / max(0.6, scale))),
                      scale, ink_d)
    return R


# `ting`, `tx` and `ty` are the subject's size and place on the sheet, and
# they are wider than they look like they need to be on purpose.
#
# Judged one sheet at a time every setting below was defensible: a 亭 at
# 0.125–0.155 of the short side is a good size for a 亭. Judged nine at a
# time (the 册页 test — 虾蟹 §5.1) the set was the same painting nine times,
# because a 1.24× range of size and a fifth of the sheet of travel is not
# variation the eye can see once the subject is the same shape every time.
# The ranges here are about 1.6× in size and a quarter of the sheet in each
# direction, which is the least that reads as nine different compositions.
SCENES = {
    "松亭": dict(
        en="pine pavilion — a spur of rock, old pines, cloud below",
        ting=(0.105, 0.172), tx=(0.26, 0.47), ty=(0.52, 0.72),
        far=(1, 3), pines=(2, 4), water=0.15, fall=0.45, cliff=0.55,
        mist=1.0, geese=0.30, bamboo=0.20, boat=0.0,
    ),
    "水亭": dict(
        en="water pavilion — built at the lip of the water",
        ting=(0.098, 0.158), tx=(0.22, 0.45), ty=(0.48, 0.66),
        far=(2, 4), pines=(1, 3), water=0.95, fall=0.10, cliff=0.20,
        mist=0.8, geese=0.55, bamboo=0.35, boat=0.55,
    ),
    "山亭": dict(
        en="mountain pavilion — a terrace under a wall of rock",
        ting=(0.095, 0.152), tx=(0.24, 0.48), ty=(0.56, 0.76),
        far=(1, 2), pines=(2, 4), water=0.30, fall=0.70, cliff=0.90,
        mist=0.9, geese=0.20, bamboo=0.30, boat=0.0,
    ),
}

PLANS = {"四角": 4, "六角": 6}

TITLE_A = ["松亭", "水亭", "山亭", "溪亭", "湖亭", "竹亭", "幽亭", "野亭",
           "江亭", "云亭", "秋亭", "临流", "听泉", "观瀑", "空亭", "山居"]
TITLE_B = ["图", "小景", "图卷", "册页", "写意", "清趣", "小品", "卷"]
ACTION = ["写意", "戏墨", "拟古", "并记", "笔意", "南窗写意", "客窗戏墨"]
SEAL_WORDS = ["林泉", "闲云", "卧游", "山静", "听泉", "墨缘", "清赏", "得趣",
              "一壑", "松风", "安之"]
PLAQUE = ["松风", "听泉", "观瀑", "清风", "水心", "爱晚", "闲云", "山静",
          "临流", "一壑"]


def vocabulary():
    """Every character this painter can write."""
    words = (TITLE_A + TITLE_B + ACTION + SEAL_WORDS + PLAQUE
             + list(SCENES) + list(PLANS) + list(STYLES)
             + [v["cn"] for v in CUN.values()]
             + [TIANGAN, DIZHI, "日月年写于图亭山水春夏秋冬"])
    return sorted(set("".join(words)))


def _load_font():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for mod in ("pavilion_font", "shanshui_font", "gongbi_font"):
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
    """题款 — as shanshui does it: measure the block, then let 留白 place it."""
    W, H = cv.w, cv.h
    gs = S * rng.uniform(0.040, 0.055)
    sz = gs * rng.uniform(0.52, 0.66)
    lead, lead_s = rng.uniform(1.12, 1.26), rng.uniform(1.18, 1.34)
    season = "春夏秋冬"[rng.randrange(4)]
    sign = _ganzhi(rng) + season + "日" + ACTION[rng.randrange(len(ACTION))]
    gap = gs * 0.45
    sign_dy = gs * rng.uniform(1.1, 2.0)
    seal_s = gs * rng.uniform(0.80, 1.0)
    bw = gs + gap + sz
    bh = max(len(title) * gs * lead,
             sign_dy + len(sign) * sz * lead_s + gs * 0.25 + seal_s)
    margin = S * 0.035
    r0 = min(W, H) * 0.20
    for fy in (0.26, 0.42, 0.58, 0.74):
        occ.add(W * 0.5, H * fy, r0, ink=False)
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin, corner_bias=1.7)
        paint_column(cv, rng, FONT, title, bx + gap + sz, by, gs, lead=lead)
        end = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, sz,
                           lead=lead_s, alpha=(0.62, 0.84))
        paint_seal(cv, rng, FONT, SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                   bx + (sz - seal_s) * 0.5, end + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
    if rng.random() < 0.55:
        s2 = S * rng.uniform(0.036, 0.050)
        for cx, cy in ((W * 0.07, H * 0.89), (W * 0.89 - s2, H * 0.89),
                       (W * 0.07, H * 0.07), (W * 0.89 - s2, H * 0.07)):
            if occ.overlap(cx, cy, cx + s2, cy + s2, s2 * 0.5) == 0.0:
                paint_seal(cv, rng, FONT,
                           SEAL_WORDS[rng.randrange(len(SEAL_WORDS))],
                           cx, cy, s2)
                break


def compose(cv, rng, scene_key, plan_key, style_key, cun_key):
    W, H = cv.w, cv.h
    S = min(W, H)
    sc, st = SCENES[scene_key], STYLES[style_key]
    cun = CUN[cun_key]
    lang = ss.set_light(rng)
    # 受光 — which side of everything is lit. `shanshui.LIGHT` points *toward*
    # the source in screen coordinates, so its x sign is the side the light
    # comes from; the building is shaded off the same number as the hills, or
    # the 亭 is lit from the right on a sheet whose mountains are lit from the
    # left, which is a thing no amount of drawing can recover from.
    lightm = 0.0 if math.cos(lang) >= 0.0 else math.pi
    ground = gb.paint_ground(cv, rng)
    paper = bytes(cv.buf)
    grain = Grain(rng)
    occ = Occupancy(W, H)
    scale = S / 1000.0

    ink_n, ink_d, ink_q = I("nong"), I("dan"), I("qing")
    tree_ink = jitter(rgb(st["tree"]), rng, 6)
    leaf_ink = jitter(rgb(st["leaf"]), rng, 8)
    near3 = tuple(rgb(c) for c in st["near"])
    far3 = tuple(rgb(c) for c in st["far"])
    mode = cun["mode"]
    per = rng.uniform(*cun["per"]) * max(0.7, scale)
    amp = rng.uniform(*cun["amp"])
    scrub, bow = cun["scrub"], st["bow"]
    flip = 1 if rng.random() < 0.5 else -1

    def mx(f):                                 # mirror a fraction of the width
        return f if flip > 0 else 1.0 - f

    n = PLANS[plan_key]
    r = S * rng.uniform(*sc["ting"])
    tx = W * mx(rng.uniform(*sc["tx"]))
    ty = H * rng.uniform(*sc["ty"])
    # The projection's two constants are settled here rather than at the 亭,
    # because the width of the platform *on paper* depends on them, and the
    # shelf cut into the terrace has to be exactly that wide.
    az, kd = rng.uniform(0.44, 0.60), rng.uniform(0.46, 0.58)
    # ...and so is the plan's rotation, for the same reason.
    rot = -math.pi * 0.5 - math.pi / n + rng.uniform(-0.20, 0.20)
    # 平面宽 — the platform's half width *on paper*, which is the width the
    # shelf has to be planed to and nothing wider.
    #
    # This is measured, not estimated, and the two estimates it replaces are
    # between them the whole of the **mesa** — the flat-topped trapezoid the
    # terrace kept coming out as, with a perfectly horizontal skyline
    # sticking out either side of the building.
    #
    #   `1.26r·(1 + cos az·kd)` = 1.9r adds the depth offset to the radius as
    #   though the near and far corners were at opposite ends of a diameter.
    #   They are not: `sx = X + Y·cx`, so the widest point of a *circle* of
    #   radius Rb is `Rb·√(1 + cx²)`.
    #
    #   And that, 1.37r, is still wrong, because the plan is a hexagon and
    #   not a circle: only one rotation in six puts a vertex at the widest
    #   bearing, and the rest of the time the true half width is as little as
    #   0.87 of it. Measured over the vertices it averages about 1.2r.
    #
    # 1.9 against 1.2 is fifty per cent of surplus shelf, cut dead level,
    # either side of the 亭 — and no amount of work on the shoulders can help
    # while the flat itself is half as long again as the thing standing on it.
    cxp = math.cos(az) * kd
    foot = r * 1.26 * max(abs(math.cos(rot + i * TAU / n)
                              + cxp * math.sin(rot + i * TAU / n))
                          for i in range(n))

    # ---- 远山 — ridges across the top, pale, behind everything ---------
    for k in range(rng.randint(*sc["far"])):
        hgt = H * rng.uniform(0.05, 0.13)
        base = H * rng.uniform(0.14, 0.34)
        rockmass(cv, rng, "far", W * rng.uniform(0.40, 0.95), hgt, base,
                 W * rng.uniform(0.08, 0.92), near3, far3, grain, mode, per,
                 amp, scrub, bow, scale, ink_d, H, peaks=(1, 3),
                 spire=(0.5, 1.1), fade=hgt * 0.30)
    mist(cv, rng, paper, H * rng.uniform(0.26, 0.36), H * 0.05,
         0.6 * sc["mist"])

    # Whether the sheet has water in it is decided here and not below,
    # because it is the one thing that says how far down the terrace goes: a
    # 水亭 stands on a bank with its foot in the river, a 山亭 on ground that
    # runs off the bottom of the paper. And the terrace is what settles the
    # pavilion's height on the sheet, so it has to be settled before the cliff
    # behind it, which is hung off that height.
    wet = rng.random() < sc["water"]
    # ...and the terrace is big. It was a modest bank for a while, out of
    # necessity rather than choice — the seat had to be *found* on the crest,
    # and only a low mass has a plateau level enough to find one on. The cost
    # was a composition: a building on a mound with a third of the sheet empty
    # below it, which is a picture of a pavilion and not a landscape. With the
    # shelf cut to fit (below) the terrace can be as tall as the sheet wants,
    # so: down past the bottom edge when the ground is dry, down to the
    # waterline when it is not.
    if wet:
        base_t = H * rng.uniform(0.82, 0.93)
        hgt_t = H * rng.uniform(0.26, 0.36)
    else:
        base_t = H * rng.uniform(1.00, 1.08)
        hgt_t = H * rng.uniform(0.30, 0.42)
    ty = base_t - hgt_t * 0.80 - r * 0.26         # provisional; the shelf rules
    # 水平 — where the water will lie, decided here rather than down in the
    # water section where it is drawn, because the trees go in before it and a
    # tree needs to know. The terrace's lower slopes run below the waterline;
    # a pine whose foot is on that part of the crest is standing in the river,
    # and one did — rooted at the shoreline with its needles over the water,
    # the same "nothing under it" defect as the tree rooted off the end of the
    # mass, with the difference that there *is* something under it and the
    # something is water.
    y_w = base_t - hgt_t * rng.uniform(0.10, 0.34)

    # ---- 崖 — a wall of rock behind the pavilion, on the far side ------
    # The band of mist across the cliff's foot is decided before the cliff is
    # drawn, because the waterfall has to end in it. See below.
    y_mist = ty - r * rng.uniform(1.0, 1.9)
    if rng.random() < sc["cliff"]:
        hgt = H * rng.uniform(0.30, 0.46)
        base = ty - r * rng.uniform(0.1, 0.6)
        rockmass(cv, rng, "mid", W * rng.uniform(0.40, 0.78), hgt, base,
                 W * mx(rng.uniform(0.62, 0.95)), near3, far3, grain, mode,
                 per, amp, scrub, bow, scale, ink_d, H, peaks=(1, 2),
                 spire=(0.62, 1.30), fade=hgt * 0.18)
        if rng.random() < sc["fall"]:
            fx = W * mx(rng.uniform(0.66, 0.90))
            y_top = base - hgt * rng.uniform(0.3, 0.6)
            # 瀑入云 — and it ends *in* the mist, not three per cent of the
            # sheet below the cliff's foot.
            #
            # `waterfall` works by lifting the rock back toward paper along a
            # wavering column and darkening the banks either side of it. High
            # on the face, where the rock is dark, that is water. Carried on
            # down past the foot it crosses ground that is already nearly
            # paper, so the lift does nothing and the two banks are all that
            # is left: a soft grey column three hundred pixels long standing
            # in the open with no rock around it and no pool under it — the
            # one mark on these sheets that people read as a smudge rather
            # than as a thing. A fall has to *stop* somewhere, and the only
            # honest places are a pool and a cloud. This one gets the cloud,
            # which is also why the mist's height is settled above: it is
            # drawn last, over the end of the fall, and dissolves it.
            y_bot = min(base + H * 0.03, y_mist + H * 0.012)
            if y_bot - y_top > H * 0.06:
                waterfall(cv, rng, paper, fx, y_top, y_bot,
                          max(2.4, S * 0.012), ink_d)
    mist(cv, rng, paper, y_mist, H * 0.045, 0.55 * sc["mist"])

    # ---- 坡 — the terrace the 亭 stands on -----------------------------
    # The rock and the 亭 are fitted to each other, and the fitting is done by
    # cutting the rock (see `cut_shelf`): a pavilion dropped onto whatever
    # crest the generator happened to make stands on one leg with the other in
    # mid-air, and raising it to clear the worst column puts the platform in
    # the sky.
    # The crest has to land on the underside of the platform, and the mass
    # has to go *down* from there, not up: built with its base at the
    # pavilion's feet the rock became a bank whose top edge ran across the
    # sheet above the pavilion floor, and the 亭 read as standing in a pit —
    # or, where the bank met water, as moored in it. `ridge` returns heights
    # above a base line, so the base is the sum and the crest is what we
    # actually want to control.
    # `spire` well under 1 and `flat` high: this mass is a 台, and the first
    # version asked for spire 1.1–1.5, which is a peak. Between the narrow
    # summit and the random 15–34 % plateau there was hardly any level ground
    # on it at all, so the 亭 ended up perched on a sliver of rock.
    cyk = math.sin(az) * kd
    ext = foot * (1.46 / 1.26)                 # 出檐 — the roof, see `foot`
    drop = r * 0.26 + (r * 1.26 + r * 0.26 * 1.38) * cyk
    rise = r * 2.45 + r * 1.46 * cyk
    shelf = {}

    def cut_shelf(x0, crest):
        """筑台 — find the best place on this crest for the 亭 and then cut it
        level. Masons do not go looking for a flat hilltop; the flatness is
        the work. Which also disposes of the whole class of bug that came of
        searching: there is no longer any such thing as a seat that doesn't
        quite fit, so there is no fallback to get wrong."""
        n = len(crest)
        rx1 = x0 + n - 1
        # The footprint has to sit clear of both ends of the mass, and not
        # merely inside them. A ridge tapers to its base line at each foot, so
        # a platform set `foot` from the end still has the sloping toe under
        # one half of it — and since the shelf is cut down to the lowest
        # ground beneath the platform, that one column at base level drags the
        # whole shelf down to the mass's own foot. Which is what happened: the
        # cut planed off the entire left half of the terrace, and the 亭 came
        # out standing on bare paper with the surviving rock beside it.
        edge = foot + min(n * 0.14, max(foot * 0.45, r * 0.5))
        windows = ((max(x0 + edge, W * 0.02 + ext),
                    min(rx1 - edge, W * 0.98 - ext)),
                   (x0 + edge, rx1 - edge),      # ...or overrun the margins
                   (max(x0 + foot, W * 0.02 + ext),
                    min(rx1 - foot, W * 0.98 - ext)),
                   (float(x0), float(rx1)))      # ...or take what there is
        for a, b in windows:
            if a <= b:
                break

        def at(cxp):
            i0 = max(0, int(cxp - foot) - x0)
            i1 = min(n - 1, int(cxp + foot) - x0)
            seg = crest[i0:i1 + 1] or [crest[max(0, min(n - 1, int(cxp) - x0))]]
            # The shelf is cut down to the lowest ground under it, never
            # filled up to the highest: cut and the platform sits in a notch
            # of its own making, fill and it stands on an invented pedestal.
            return max(seg)

        hi_t = min(crest)
        best, bx, byy = None, None, None
        for i in range(49):
            cxp = a + (b - a) * i / 48.0
            yy = at(cxp) - r * 0.26 + r * 0.03
            if not (rise + H * 0.045 < yy and yy + drop < H * 0.975):
                continue
            # High on the mass, and near where the composition wanted it.
            sco = (yy - hi_t) * 0.85 + abs(cxp - tx) * 1.0
            if best is None or sco < best:
                best, bx, byy = sco, cxp, yy
        if bx is None:                          # nowhere on the sheet: centre it
            bx = (a + b) * 0.5
            byy = min(max(at(bx) - r * 0.26 + r * 0.03,
                          rise + H * 0.045), H * 0.975 - drop)
        # Cut. The shelf runs the width of the platform plus a hand's breadth,
        # with a short ramp at each shoulder so the 台 is built into the hill
        # rather than sawn out of it.
        top = byy + r * 0.26
        # Level under the slab and nowhere else, and then a **talus**: a cut
        # that climbs away from the bench at a constant angle until it runs
        # into whatever the hill was already doing.
        #
        # The first version planed level for `foot + r*0.10` and then climbed
        # back over a fixed ramp of `r*0.22`. Both halves of that were wrong.
        # The level part was too long (see `foot` in `compose`), and the ramp
        # was far too short to reach the hill: it stopped a hundred pixels
        # above the bench with the natural crest still two hundred above
        # *that*, so the cut ended in a dead vertical wall three hundred
        # pixels tall and perfectly straight — a quarry face, in a landscape
        # where nothing else has a straight edge anywhere.
        #
        # A bench cut into a slope does not have a length. It has an *angle*,
        # and it ends wherever the hillside happens to come down to meet it,
        # which is a different place on each side and no place at all when
        # the hill was already lower. So: a slope per side, drawn
        # independently, with a long slow wobble along it, and no bound on
        # how far it runs — `max` stops it the moment the hill is below it.
        hard = foot + r * 0.04
        tal = (rng.uniform(0.42, 0.88), rng.uniform(0.42, 0.88))
        wob = (rng.uniform(0.0, TAU), rng.uniform(0.0, TAU))
        per_ = max(4.0, r * rng.uniform(0.22, 0.38))
        # `max` because y grows downward: the shelf is cut *down* to the
        # lowest ground under it, and anywhere the hill is already lower than
        # the shelf — a dip at the shoulder — it stays where it is.
        for i in range(n):
            s = (x0 + i) - bx
            d = abs(s)
            if d <= hard:
                crest[i] = max(crest[i], top)
                continue
            k = 0 if s < 0.0 else 1
            e = d - hard
            z = top - e * tal[k] - math.sin(e / per_ + wob[k]) * r * 0.045
            crest[i] = max(crest[i], z)
        shelf["x"], shelf["y"] = bx, byy

    terr = rockmass(cv, rng, "near", W * rng.uniform(0.86, 1.30),
                    hgt_t, base_t,
                    tx + r * rng.uniform(-0.3, 0.3), near3, far3, grain,
                    mode, per, amp, scrub, bow, scale, ink_d, H,
                    # `flat` was 0.90 — nine tenths of the crest a plateau —
                    # and that is a fossil of the version that *searched* the
                    # crest for somewhere level to stand the 亭 on. Nothing
                    # searches any more: `cut_shelf` makes the level ground.
                    # Left at 0.90 all it did was guarantee that the terrace
                    # was already a table before the shelf was cut into it,
                    # which is the half of the mesa that survived fixing
                    # `foot`. A 台 is a hill with a bench cut in it.
                    peaks=(1, 2), spire=(0.52, 0.98), flat=0.22,
                    level=cut_shelf, centre=True)
    if terr is not None:
        # More 点 than a mountain would get. The shelf planes the crest dead
        # level over the whole width of the platform, and a level crest with a
        # flat wash under it is a field; the dots are what keep the top of the
        # 台 a piece of ground.
        terr.moss(cv, rng, int(26 * scale + 10), ink_n, scale)
        terr.register(occ)
        if shelf:
            tx, ty = shelf["x"], shelf["y"]

    # ---- 树 — pines behind the pavilion, and bamboo beside it ----------
    pine_xy = []
    # The 亭 is the subject, so the trees have to get out of its way. Placed
    # by band alone one landed directly behind the roof and its branches came
    # out of the ridge like a dead bush growing from the tiles; another
    # dropped its crown over the whole interior. Trees are therefore kept a
    # clear margin to either side of the building in x — which is also how
    # they end up framing it.
    # They also have to stand on the ground. A tree whose x fell off the end
    # of the terrace used to get the platform's own height for want of a
    # crest to read, which put a bare 枯 tree rooted in mid-air half way up
    # the left margin — the one mark in the picture with nothing under it.
    # A trunk is not a tree, though, and a clearance measured to the trunk is
    # the defect this code had longest. The 出檐 is r*1.46 and the oblique
    # spreads it a little further across the sheet, so the roof's own half
    # width is about r*1.60; the old test asked for r*1.70 between the trunk
    # and the centre of the building — seventeen pixels of paper between the
    # trunk and the eave tip, into which a crown that measures 0.45·h to
    # either side (0.60 at the ninth decile; measured, not guessed, over forty
    # trees of each kind) was then painted. So every tall pine put its needles
    # over the tiles, and the one thing the roof must not have crossing it is
    # the one thing that always did.
    #
    # What a clearance has to leave room for is therefore the crown, and the
    # crown's width is set by `h` — so `h` is what gives, not the position.
    # The gap between eave and trunk is measured first and the tree is grown
    # to fit it: out at the margin it can be the full height, and close in it
    # is a smaller tree or no tree at all. That is also how the trees come to
    # frame the building instead of standing in a row with it — 疏密 out of
    # arithmetic.
    eave = r * 1.46 * math.hypot(1.0, math.cos(az) * kd)
    for k in range(rng.randint(*sc["pines"])):
        px, py, h = None, None, None
        for _ in range(14):
            side = mx(rng.uniform(0.58, 0.96)) if k % 2 == 0 \
                else mx(rng.uniform(0.04, 0.30))
            cand = W * side
            gap = abs(cand - tx) - eave - r * 0.10
            if gap <= 0.0 or any(abs(cand - q) <= r * 0.55
                                 for q, _ in pine_xy):
                continue
            hh = min(H * rng.uniform(0.17, 0.30), gap / 0.55)
            if hh < H * 0.11:                # too little room to be a tree
                continue
            cy = terr.crest_y(cand) if terr is not None else None
            if cy is None:
                if terr is not None:
                    continue                 # off the mass: nothing to stand on
                cy = ty + r * rng.uniform(0.0, 0.45) - 1.0
            if wet and cy > y_w - r * 0.06:
                continue                     # that part of the crest is river
            px, py, h = cand, cy + 1.0, hh
            break
        if px is None:
            continue
        tree(cv, rng, px, py, h, rng.choice(("松", "松", "松", "枯")),
             tree_ink, leaf_ink, scale, alpha=0.92)
        pine_xy.append((px, py - h))
        occ.add(px, py - h * 0.6, h * 0.34)

    # ---- 亭 ------------------------------------------------------------
    prj = Oblique(tx, ty, az=az, kd=kd)        # `rot` was settled with `foot`
    Zc = ting(cv, rng, prj, n, rot, r, ink_n, ink_d,
              tiers=1 if rng.random() < 0.78 else 2,
              bow=0.03 + rng.random() * 0.03,
              name=PLAQUE[rng.randrange(len(PLAQUE))] if rng.random() < 0.55
              else None, font=FONT, paper=paper, lightm=lightm)
    occ.add(tx, ty - r * 1.1, r * 1.9)

    # 草 — tufts along the foot of the platform. The reserved paper leaves a
    # clean edge where the building meets the hill, and a clean edge between
    # two things is a cut-out: the 亭 looked pasted on. A dozen flicks of
    # grass across the join is what a painter puts there, and it is the only
    # mark in the picture whose job is to be ambiguous.
    yb = ty + r * 0.26
    # The join is the *silhouette of the platform's foot*, and that is not the
    # shelf level. `ty + r*0.26` is where the cut was made, which on screen is
    # about half way up the slab: the far side of a slab drawn in parallel
    # projection climbs, so its bottom ring runs from well below that line at
    # the near corner to well above it at the back. Clamped to the cut, a
    # third of the tufts came out standing *on the pavilion floor* along its
    # back edge — grass growing out of the paving, which is worse than no
    # grass at all because it is the one mark here whose job is to be
    # ambiguous and it was being unambiguously wrong.
    Rb, thp = r * 1.26, r * 0.26
    ring_f = [prj(math.cos(rot + i * TAU / n) * Rb,
                  math.sin(rot + i * TAU / n) * Rb, -thp) for i in range(n)]

    def slab_y(x):
        """Screen y of the underside of the platform at this x, or None."""
        lo = None
        for i in range(n):
            (ax, ay), (bx_, by_) = ring_f[i], ring_f[(i + 1) % n]
            if ax == bx_ or not (min(ax, bx_) <= x <= max(ax, bx_)):
                continue
            yy = ay + (by_ - ay) * (x - ax) / (bx_ - ax)
            lo = yy if lo is None else max(lo, yy)
        return lo

    for _ in range(int(rng.randint(12, 20))):
        gx = tx + r * rng.uniform(-1.8, 1.8)
        # On the hill where the hill is visible, and on the stone's own foot
        # where the stone is in front of it. Whichever is lower on paper wins,
        # because that is the one the eye reads as the contact.
        cy_ = terr.crest_y(gx) if terr is not None else None
        gy = yb if cy_ is None else cy_
        sy = slab_y(gx)
        if sy is not None:
            gy = max(gy, sy)
        # ...and kept short and dry. Unclamped and at full length these went
        # up the crest beside the building and stood at floor height: a row of
        # tall dark spikes inside the 亭, which read as a picket fence.
        h = r * rng.uniform(0.045, 0.105)
        for _ in range(rng.randint(2, 4)):
            a = -math.pi * 0.5 + rng.uniform(-0.65, 0.65)
            brush_seg(cv, gx, gy, gx + math.cos(a) * h, gy + math.sin(a) * h,
                      max(0.5, r * 0.012), 0.35, ink_n,
                      rng.uniform(0.22, 0.46))

    # 竹 — a clump at the corner of the terrace. Drawn after the building, so
    # it stands in front of the platform's skirt rather than behind it: the
    # near corner of a 台 is where a painter puts bamboo, because it is the one
    # place a vertical can overlap the architecture without hiding any of it.
    if rng.random() < sc["bamboo"]:
        for sd in (1.0, -1.0) if rng.random() < 0.5 else (-1.0, 1.0):
            bx = tx + sd * r * rng.uniform(1.75, 2.45)
            if not (W * 0.045 < bx < W * 0.955):    # not half off the sheet
                continue
            cy_ = terr.crest_y(bx) if terr is not None else None
            if terr is not None and cy_ is None:
                continue
            bamboo_clump(cv, rng, bx, (cy_ if cy_ is not None else yb) + 1.0,
                         r * rng.uniform(1.15, 1.85),
                         mix(tree_ink, leaf_ink, 0.45), leaf_ink)
            break

    # ---- 水 — the lower register ---------------------------------------
    # Water laps at the foot of the terrace, a little above it: set by the
    # pavilion's own size instead it either ran across the rock halfway up or
    # left a strip of bare paper between bank and river.
    boat_x, boat_s = None, 0.0
    if wet and y_w < H - H * 0.06:
        water_lines(cv, rng, y_w, H * 0.99, ink_d, 1.0)
        shore_rocks(cv, rng, y_w + H * 0.01, H * 0.98, ink_n, scale,
                    k=rng.randint(2, 4))
        if rng.random() < sc["boat"]:
            # 舟 — and its size comes from *where it is*, not from the sheet.
            #
            # Inherited from shanshui, where a boat sits in the middle distance
            # and is a 50 px mark, it stayed a 50 px mark here — floating in
            # the near water, which is nearer to the viewer than the terrace,
            # a few feet in front of a 亭 whose own standing figure is 114 px
            # tall. Its boatman came out 12 px. That is 远小近大 inverted: the
            # nearer figure an eighth the height of the farther one, and the
            # one thing in the picture that cannot be read as anything but a
            # mistake.
            #
            # The pavilion sets the scale of the sheet, and it sets it at
            # roughly 70 px to the metre: r*2 across the column ring is about
            # 3.5 m. So a small boat at this distance is 250–350 px — a third
            # of the width. That is simply what a boat in the foreground of a
            # close-up *is*, and painters draw it that way: moored, cropped by
            # the edge, half hidden behind the near bank (which is drawn after
            # it, so it overlaps correctly).
            #
            # And at that length it cannot be `shanshui.boat` any more: see
            # `moored_boat` above for why four thin outlines that are a solid
            # mark at fifty pixels are an empty parallelogram at three hundred.
            # 空舟 — and no boatman either. `boat` draws its figure at 0.22 of
            # the hull, which is a seven-metre boat for a man; at a hull length
            # that is finally right, the man is half the height of the scholar
            # in the 亭 and the same inversion comes back one level down. An
            # empty moored boat is 野渡无人舟自横 and a better painting.
            #
            # 水上 — and it has to be *in the water*. The terrace runs down to
            # the waterline and usually well past it, and the boat is drawn
            # after the terrace, so one placed by (x, y) alone comes out
            # moored on dry rock half way up the bank with the hillside
            # showing through its hull. Which is a fault nobody sees while
            # they are looking at the boat and nobody can un-see afterwards.
            # Three samples along the hull, because at a third of the sheet's
            # width the bow can be afloat while the stern is aground.
            def aground(x, y):
                """Is this point inside the terrace rather than on water?"""
                if terr is None:
                    return False
                cy_ = terr.crest_y(x)
                return cy_ is not None and y + r * 0.10 > cy_

            for _ in range(14):
                v = rng.uniform(0.28, 0.76)      # 0 at the far shore, 1 near
                bx_ = W * mx(rng.uniform(0.34, 0.78))
                bs_ = r * (2.0 + 1.1 * v)
                by_ = y_w + (H - y_w) * v
                if any(aground(bx_ + bs_ * f, by_) for f in (-0.5, 0.0, 0.5)):
                    continue
                boat_x, boat_s = bx_, bs_
                moored_boat(cv, rng, bx_, by_, bs_, ink_n, ink_d)
                break
        # 对岸 — a low far shore across the water, so the water has a far edge
        # as well as a near one and reads as a reach rather than as a tone.
        if rng.random() < 0.65 and y_w < H * 0.88:
            rockmass(cv, rng, "far", W * rng.uniform(0.55, 1.15),
                     H * rng.uniform(0.022, 0.045), y_w + H * 0.008,
                     W * rng.uniform(0.05, 0.95), near3, far3, grain, mode,
                     per, amp, scrub, bow, scale, ink_d, H, peaks=(1, 3),
                     spire=(0.4, 0.9), texture=False)
        # 近岸 — a bank across the bottom of the sheet, in front of the water.
        # Without it the sheet was a building on a mound with a quarter of the
        # paper empty below it, and three registers is what makes the middle
        # one read as middle distance: near bank, then water, then the terrace
        # and the 亭. Not optional, and not a corner either — at a third of
        # the sheet's width it was a green pyramid sitting in the river.
        #
        # Two things about it were wrong for a long time. Its high end is
        # anchored in the corner *away* from the 亭: the pavilion takes one
        # side and the inscription the opposite top corner, which leaves the
        # opposite bottom corner holding nothing but a seal, and the bank is
        # what closes that diagonal. And it is tall — at a twelfth of the
        # sheet its crest sat below 0.90H, so the trees and reeds sampled from
        # it fell off the bottom edge and the bank was two green blisters with
        # nothing on them. It can afford the height precisely because its
        # summit is in the corner, where there is no building to bury.
        nb = rockmass(cv, rng, "near", W * rng.uniform(0.70, 1.15),
                      H * rng.uniform(0.115, 0.195), H * 1.04,
                      W * (rng.uniform(0.04, 0.34) if tx > W * 0.5
                           else rng.uniform(0.66, 0.96)), near3, far3, grain,
                      mode, per, amp, scrub, bow, scale, ink_d, H,
                      peaks=(1, 3), spire=(0.55, 1.0), flat=0.4)
        if nb is not None:
            nb.moss(cv, rng, int(12 * scale + 5), ink_n, scale)
            for _ in range(rng.randint(1, 3)):
                # Clear of the 亭 — but by the crown, not by the trunk, and
                # measured to the column ring rather than to the eave. These
                # are foreground trees, drawn after the building and therefore
                # in front of it, and a near branch crossing the eaves is what
                # puts the building *behind* them; a crown over the interior
                # is what buries it. So they may graze the roof and nothing
                # more, and like the pines they lose height rather than
                # position when the bank gives them no room.
                pt, ht = None, None
                for _ in range(10):
                    sx, sy = nb.sample(rng, 1, lo=0.12, hi=0.88)[0]
                    gap = abs(sx - tx) - r * 1.26
                    if gap <= 0.0:
                        continue
                    # ...and off the boat. The bank is drawn over the water and
                    # over the moored boat, which is the right order — the boat
                    # is *behind* the bank — but a tree rooted on the low tail
                    # of the bank where the boat happens to lie comes up out of
                    # the hull. At fifty pixels the boat was too small for this
                    # to ever happen; at three hundred it is a third of the
                    # width of the sheet and two of the three near trees landed
                    # in it.
                    if boat_x is not None and abs(sx - boat_x) < boat_s * 0.60:
                        continue
                    hh = min(H * rng.uniform(0.09, 0.17), gap / 0.55)
                    if hh < H * 0.07:
                        continue
                    pt, ht = (sx, sy), hh
                    break
                if pt is None:
                    continue
                tree(cv, rng, pt[0], pt[1], ht,
                     rng.choice(("丛", "柳", "枯")), tree_ink, leaf_ink,
                     scale, alpha=0.95)
            for _ in range(rng.randint(5, 11)):     # 芦 — reeds along the edge
                sx, sy = nb.sample(rng, 1, lo=0.04, hi=0.96)[0]
                hh = H * rng.uniform(0.016, 0.036)
                for _ in range(rng.randint(2, 4)):
                    aa = -math.pi * 0.5 + rng.uniform(-0.5, 0.5)
                    brush_seg(cv, sx, sy, sx + math.cos(aa) * hh,
                              sy + math.sin(aa) * hh, max(0.5, 0.9 * scale),
                              0.35, ink_n, rng.uniform(0.3, 0.6))
    else:
        # 坡石 — a near bank across the bottom corner instead of water
        rockmass(cv, rng, "near", W * rng.uniform(0.50, 0.95),
                 H * rng.uniform(0.10, 0.18), H * 1.02,
                 W * (rng.uniform(0.02, 0.30) if tx > W * 0.5
                      else rng.uniform(0.70, 0.98)), near3, far3, grain, mode,
                 per, amp, scrub, bow, scale, ink_d, H, peaks=(1, 2),
                 spire=(0.9, 1.3))
        # And they stand on it, and clear of the building. Spread across the
        # whole width they were rooted on bare paper wherever the bank wasn't;
        # worse, one rooted at the bottom edge with a fifth of the sheet's
        # height in it reaches the floor of the 亭, and a bare 枯 crown then
        # grows up through the railing and fills the interior. A foreground
        # tree is *meant* to overlap the terrace — that is what puts the
        # terrace behind it — but not the building on top of it.
        # The clearance is by the crown and to the column ring, as on the near
        # bank above: a foreground branch may cross the eaves, a crown may not
        # stand in the interior.
        for _ in range(rng.randint(1, 3)):
            bx, bh = None, None
            for _ in range(12):
                cand = W * (rng.uniform(0.03, 0.34) if tx > W * 0.5
                            else rng.uniform(0.66, 0.97))
                gap = abs(cand - tx) - r * 1.26
                if gap <= 0.0:
                    continue
                hh = min(H * rng.uniform(0.10, 0.20), gap / 0.55)
                if hh < H * 0.07:
                    continue
                bx, bh = cand, hh
                break
            if bx is None:
                continue
            tree(cv, rng, bx, H * rng.uniform(0.93, 1.0), bh,
                 rng.choice(("阔", "丛", "柳", "枯")), tree_ink, leaf_ink,
                 scale, alpha=0.95)

    # ---- 雁 or 飞鸟 -----------------------------------------------------
    if rng.random() < sc["geese"]:
        geese(cv, rng, occ, W, H, S, ink_d, ink_q)
    elif rng.random() < 0.45:
        birds(cv, rng, W * rng.uniform(0.25, 0.75), H * rng.uniform(0.10, 0.24),
              4.0 * scale, rng.randint(2, 5), ink_d)

    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))])
    inscribe(cv, rng, occ, S, title)
    return title, ground


def paint(path, size=(900, 1200), seed=None, scene=None, plan=None,
          style=None, cun=None, quiet=False):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    pk = plan or sorted(PLANS)[rng.randrange(len(PLANS))]
    yk = style or sorted(STYLES)[rng.randrange(len(STYLES))]
    ck = cun or sorted(CUN)[rng.randrange(len(CUN))]
    for key, tbl, what in ((sk, SCENES, "scene"), (pk, PLANS, "plan"),
                           (yk, STYLES, "style"), (ck, CUN, "cun")):
        if key not in tbl:
            raise SystemExit("unknown %s %r; try --list" % (what, key))

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, pk, yk, ck)
    ss.age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-26s %s  %s·%s亭·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, sk, pk, CUN[ck]["cn"],
                 STYLES[yk]["cn"], ground, seed, size[0], size[1],
                 time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate random Chinese pavilion (亭) scenery.")
    ap.add_argument("-o", "--out", default="pavilion.png")
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("-s", "--seed", type=int)
    ap.add_argument("--scene", help="松亭|水亭|山亭")
    ap.add_argument("--plan", help="四角|六角")
    ap.add_argument("--style", help="水墨|浅绛|青绿")
    ap.add_argument("--cun", help="披麻|斧劈|雨点|折带|卷云")
    ap.add_argument("--size", default="900x1200", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        print("景 (scene):")
        for k, v in SCENES.items():
            print("  %-6s %s" % (k, v["en"]))
        print("平面 (plan):")
        for k, v in PLANS.items():
            print("  %-6s %d-sided" % (k, v))
        print("皴法 / 设色: as shanshui.py")
        return 0

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except Exception:
        raise SystemExit("--size must look like 900x1200")

    root, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    print("亭 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s_%02d%s" % (root, i + 1, ext)
        seed = None if a.seed is None else a.seed + i
        paint(out, (w, h), seed, a.scene, a.plan, a.style, a.cun)
    return 0


if __name__ == "__main__":
    sys.exit(main())
