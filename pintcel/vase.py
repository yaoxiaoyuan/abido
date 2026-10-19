#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vase.py — 瓶花 / 清供: a porcelain vase of flowers, painted 工笔 on silk.

The eleventh painter in the set, and the first whose subject is a *made*
thing. Everything before it grew — a branch, a fish, a pine — and grew with
the happy property that no two instances need agree about anything. A thrown
pot is the opposite: it is a solid of revolution, it is bilaterally
symmetric to within the thickness of a brush line, its curve was pulled in
one continuous movement, and the eye knows all three facts before it knows
what flower is in it. Nothing in the family's toolkit does that, so the
vessel is built here from first principles and only the flowers are borrowed.

What is borrowed is most of the painting, though. `gongbi.py` already has
six species with their petals, leaves, buds, pigments and 罩染 glaze, and
`paint_blossom`/`paint_leaf`/`leaf_sites` do not care whether the wood they
hang on came out of the ground or out of a bottle. So:

    器      this module      profile table, 旋坯 shading, 釉, 开片, 青花,
                            圈足, 木座, the 三才 arrangement of the spray
    花      gongbi.py        花头, 花苞, 叶, 枝, 设色, 蝶蜂, 绢底, 题款

Three decisions carry the whole of the vessel.

**The form is a table, not a silhouette.** `PROFILES` gives each 器型 as a
dozen `(height, radius)` control points through which a Catmull-Rom spline is
run once into a per-scanline radius table. Every later operation — the fill,
the glaze, the decoration, the mouth, the foot, where the flowers go — reads
that one table, so the pot cannot disagree with itself.

**The surface is addressed in azimuth, not in pixels.** A point on the pot is
`(phi, t)`: how far round the axis, and how far up. `Body.pt` projects it,
`x = cx + r(t)·sin(phi)`. Decoration drawn in that space is compressed toward
the silhouette exactly as it should be, wraps out of sight at `phi = ±pi/2`,
and takes its light from the same `cos(phi - phi_L)` the body does. Drawn in
screen space instead, a 缠枝莲 scroll is a decal on a flat oval.

**The glaze goes on last.** 青花 is underglaze cobalt and 开片 is a crack in
the glaze, so the body and its decoration are painted first and
`Body.gloss()` lays the specular band and the reflected-light rim over
everything afterwards. The highlight crossing the decoration is the single
cue that says "there is a layer of glass on top of this", and it costs one
extra pass over the same pixels.

    python3 vase.py --list
    python3 vase.py -s 7 --size 900x1300 -o /tmp/v.png
    python3 vase.py --vase 玉壶春 --ware 青花 --flower peony -o /tmp/v.png
"""

import argparse
import math
import os
import random
import sys
import time

from gongbi import (
    Canvas, Noise, Occupancy, BrushFont,
    P, rgb, mix, hsv_mix, jitter, saturate, clamp01, bell, to_hsv,
    disc, brush_seg, brush_path, bezier, tangent, knuckle,
    draw_branch, paint_ground, set_light,
    SPECIES, glaze_palette, paint_blossom, paint_bud, paint_leaf, leaf_sites,
    paint_butterfly, paint_bee, paint_column, paint_seal, corner_seal,
    TIANGAN, DIZHI, SEAL_WORDS, age,
)

TAU = math.pi * 2.0


# --------------------------------------------------------------------------
# 器型 — the forms, as profile control points
# --------------------------------------------------------------------------
#
# Each entry is a list of `(t, r)`: `t` runs 0 at the foot-ring to 1 at the
# lip, `r` is the half-width as a fraction of the pot's widest point. They
# are read as control points of a Catmull-Rom spline, so the curve passes
# through all of them and the potter's profile is whatever the list says —
# which is the only honest way to get five *named* shapes rather than five
# samples of one parametric blob.
#
# Two rules the lists all obey, because the kiln obeys them:
#
#   - the last two or three points are the lip, and a lip is always a local
#     reversal — a rolled 盘口, a flared 撇口, a thickened 唇口. A profile
#     that just stops is a cut pipe.
#   - `r` at t=0 is the foot ring, and it is never the widest point except
#     on 觚. A pot stands on less than it carries.

PROFILES = {
    # 梅瓶 — the plum vase: tiny mouth, broad shoulder carried high, long
    # taper to a small foot. The mouth is small on purpose; it was a wine
    # bottle before it was a flower vase, and it holds one branch.
    "梅瓶": [(0.00, 0.47), (0.04, 0.44), (0.07, 0.46), (0.20, 0.57),
             (0.38, 0.73), (0.56, 0.90), (0.70, 0.99), (0.78, 1.00),
             (0.86, 0.88), (0.91, 0.56), (0.945, 0.30), (0.965, 0.255),
             (0.985, 0.25), (1.00, 0.295)],
    # 玉壶春 — pear body, slender waisted neck, trumpet mouth. The classic
    # 瓶花 vessel: the flare of the mouth is what lets a spray sit wide.
    "玉壶春": [(0.00, 0.40), (0.04, 0.37), (0.08, 0.41), (0.18, 0.60),
               (0.30, 0.82), (0.40, 0.92), (0.46, 0.94), (0.56, 0.86),
               (0.66, 0.66), (0.74, 0.47), (0.82, 0.36), (0.88, 0.335),
               (0.93, 0.37), (0.97, 0.47), (1.00, 0.58)],
    # 胆瓶 — the gallbladder: one unbroken curve from a round low belly into
    # a long plain neck. No shoulder, no lip to speak of. The hardest of the
    # five to draw and the easiest to get wrong, because any kink shows.
    "胆瓶": [(0.00, 0.36), (0.04, 0.34), (0.08, 0.40), (0.18, 0.60),
             (0.30, 0.79), (0.39, 0.86), (0.47, 0.84), (0.57, 0.71),
             (0.67, 0.54), (0.76, 0.39), (0.85, 0.29), (0.93, 0.245),
             (0.98, 0.24), (1.00, 0.27)],
    # 花觚 — the beaker, after the bronze 觚: a trumpet over a waist over a
    # splayed foot, and the only form here that is widest at the top.
    "花觚": [(0.00, 0.96), (0.03, 0.80), (0.07, 0.62), (0.14, 0.49),
             (0.24, 0.40), (0.36, 0.355), (0.46, 0.35), (0.54, 0.37),
             (0.62, 0.42), (0.72, 0.52), (0.82, 0.67), (0.91, 0.84),
             (0.97, 0.96), (1.00, 1.00)],
    # 贯耳瓶 — the tubular-handled vase, after a bronze 投壶: a square-ish
    # swelling body, a straight neck with two tube handles, a flat lip.
    "贯耳瓶": [(0.00, 0.54), (0.04, 0.51), (0.08, 0.60), (0.16, 0.72),
                (0.26, 0.78), (0.40, 0.80), (0.52, 0.785), (0.62, 0.72),
                (0.70, 0.56), (0.76, 0.42), (0.80, 0.375), (0.92, 0.37),
                (0.96, 0.385), (1.00, 0.44)],
}

# `hw` is the *half*-width at the widest point, as a fraction of the vessel's
# height — so a pot whose width is 45 % of its height has hw=0.225. Getting
# this wrong by a factor of two is easy and does not look like an error: it
# looks like a different, dumpier pot. Every 器型 here is between 0.21 and
# 0.25, because a tall vessel is the whole reason a 瓶花 composition works —
# the vase supplies the vertical that the spray then leans off.
VASES = {
    "梅瓶":   dict(cn="梅瓶", en="plum vase", prof="梅瓶", hw=0.222,
                   squash=0.26, ears=0.0, foot=0.030),
    "玉壶春": dict(cn="玉壶春", en="pear-shaped vase", prof="玉壶春", hw=0.238,
                   squash=0.27, ears=0.0, foot=0.028),
    "胆瓶":   dict(cn="胆瓶", en="gallbladder vase", prof="胆瓶", hw=0.214,
                   squash=0.25, ears=0.0, foot=0.026),
    "花觚":   dict(cn="花觚", en="beaker vase", prof="花觚", hw=0.228,
                   squash=0.30, ears=0.0, foot=0.034),
    "贯耳瓶": dict(cn="贯耳瓶", en="tubular-handled vase", prof="贯耳瓶",
                   hw=0.246, squash=0.26, ears=1.0, foot=0.030),
}


# --------------------------------------------------------------------------
# 窑口 — the wares
# --------------------------------------------------------------------------
#
# A glaze is four colours and a gloss, and the four are not free:
#
#   body   what the glaze reads as in full even light
#   deep   the same glaze at the terminator — *not* grey. A celadon in
#          shadow goes blue-green and a copper red goes near-black plum;
#          sliding every ware toward the same grey is what makes rendered
#          pottery look like plastic.
#   hi     the specular. Porcelain reflects the window, not its own colour,
#          so this is near-white for a glossy ware and barely above the
#          body colour for a matte one.
#   rim    the colour of the lip, where the glaze runs thin over the clay
#          and the body shows through. 灯草口 on a 霁红, 芒口 on a 定窑.

WARES = {
    "青花": dict(cn="青花", en="blue-and-white",
                 body="#EFEDDF", deep="#7E8C86", hi="#FFFFFF", rim="#E8E6D8",
                 gloss=0.95, pool=0.16, crackle=0.0, decor="qinghua",
                 rimline=0.0, ink="#27508C"),
    "龙泉": dict(cn="龙泉", en="Longquan celadon",
                 body="#BBCBB4", deep="#4E6B62", hi="#F4FBF2", rim="#D8DCC4",
                 gloss=0.82, pool=0.34, crackle=0.35, decor="incise",
                 rimline=0.0, ink="#5E7E70"),
    "霁红": dict(cn="霁红", en="sacrificial red",
                 body="#9E2A20", deep="#38100F", hi="#FFE9DC", rim="#F0E2CC",
                 gloss=0.98, pool=0.42, crackle=0.0, decor=None,
                 rimline=1.0, ink="#5A1410"),
    "哥窑": dict(cn="哥窑", en="Ge ware, 金丝铁线",
                 body="#E2D8BE", deep="#6E6450", hi="#F8F3E6", rim="#BCA886",
                 gloss=0.42, pool=0.20, crackle=1.0, decor=None,
                 rimline=0.0, ink="#5A5244"),
    "甜白": dict(cn="甜白", en="sweet white",
                 body="#F6F3E8", deep="#B8BCAC", hi="#FFFFFF", rim="#EFEBDA",
                 gloss=0.70, pool=0.12, crackle=0.0, decor="incise",
                 rimline=0.0, ink="#CFCDBC"),
}


# --------------------------------------------------------------------------
# 章法 — what goes on the sheet
# --------------------------------------------------------------------------

# `dens` is the scene's 疏密: a multiplier on the *sheet's* flower, bud and
# leaf budget. It exists because the counts in `gongbi.SPECIES` are written
# for one plant filling one sheet, and a vase holds three or four stems. A
# scene that wants to be empty cannot get there by drawing fewer stems — two
# stems at full count is as busy as four — it has to ask for fewer flowers.

SCENES = {
    # The plain vase of flowers: vessel on bare paper, generous spray.
    "瓶花": dict(cn="瓶花", en="a vase of flowers",
                 vh=(0.34, 0.42), vx=(0.40, 0.60), stand=0.25, fruit=0.0,
                 reach=(1.05, 1.40), mixed=0.0, nbranch=(3, 3), dens=1.00),
    # 清供 — "pure offering": the New Year arrangement, vessel on its carved
    # 木座 with fruit set beside it. The companion is what makes it a 清供
    # rather than a 瓶花, so it is not optional here. The fruit and the stand
    # are themselves mass, so the spray is pulled back to leave room for them.
    "清供": dict(cn="清供", en="a scholar's offering", vh=(0.30, 0.37),
                 vx=(0.36, 0.52), stand=1.0, fruit=1.0,
                 reach=(0.90, 1.20), mixed=0.45, nbranch=(3, 3), dens=0.85),
    # 满堂 — the full sheet: 杂花, two species in one vessel, which is what a
    # 岁朝图 does and what nothing else in this family can do, because only
    # here are the flowers detached from their own tree. `mixed=1.0` only
    # *tries*; whether it succeeds depends on the season having a partner.
    "满堂": dict(cn="满堂", en="a full arrangement", vh=(0.28, 0.35),
                 vx=(0.42, 0.58), stand=0.55, fruit=0.30,
                 reach=(1.10, 1.45), mixed=1.0, nbranch=(3, 4), dens=1.40),
    # 孤瓶 — 疏: one or two stems, the vessel off to one side, most of the
    # paper empty. The 留白 is the subject.
    "孤瓶": dict(cn="孤瓶", en="one vase, few stems", vh=(0.32, 0.40),
                 vx=(0.26, 0.40), stand=0.30, fruit=0.0,
                 reach=(0.80, 1.10), mixed=0.0, nbranch=(2, 2), dens=0.60),
}

TITLE_A = ["清供", "岁朝", "瓶花", "案头", "书斋", "四时", "寒香", "满堂",
           "供春", "折枝", "玉壶", "胆瓶"]
TITLE_B = ["图", "清趣", "小景", "图卷", "写生", "小品", "册页", "清玩"]
ACTION = ["写生", "戏作", "写意", "并记", "南窗写生", "客窗戏墨", "拟古",
          "仿宋人笔意", "案上写"]
VASE_SEALS = ["清供", "瓶隐", "案头清", "墨缘", "闲心", "知春", "长乐",
              "清赏", "古意", "三友", "安之"]


# --------------------------------------------------------------------------
# 旋坯 — the profile, run once into a scanline table
# --------------------------------------------------------------------------

def _catmull(ctrl, t):
    """Catmull-Rom through `(t, r)` control points, clamped at the ends.

    A pot's curve has to pass *through* the points it is described by: the
    shoulder of a 梅瓶 is at a particular height and a particular width, and
    a bezier that merely leans toward it gives a different pot. The ends are
    clamped by duplicating the first and last control point, so the lip and
    the foot do not get a phantom tangent from outside the vessel.
    """
    n = len(ctrl)
    if t <= ctrl[0][0]:
        return ctrl[0][1]
    if t >= ctrl[-1][0]:
        return ctrl[-1][1]
    i = 0
    while i < n - 2 and ctrl[i + 1][0] < t:
        i += 1
    p0 = ctrl[max(0, i - 1)]
    p1, p2 = ctrl[i], ctrl[i + 1]
    p3 = ctrl[min(n - 1, i + 2)]
    span = p2[0] - p1[0]
    u = 0.0 if span <= 1e-9 else (t - p1[0]) / span
    u2 = u * u
    u3 = u2 * u
    return 0.5 * ((2.0 * p1[1])
                  + (-p0[1] + p2[1]) * u
                  + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * u2
                  + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * u3)


class Body:
    """One thrown vessel: the radius table, the lighting, and the surface map.

    Screen geometry is `cx` (the axis), `ytop` (the lip) and `ybot` (the
    foot). `t` runs 0 at the foot to 1 at the lip, so it is the profile's own
    parameter and nothing has to remember which way up the table is.

    The light is taken from gongbi's global `LIGHT`, re-expressed as an
    azimuth about the pot's axis, so petal, leaf, bark and glaze all model
    from one window. A pot lit from the left among flowers lit from the right
    is the fastest way to make a collage out of a painting.
    """

    def __init__(self, cv, rng, form, cx, ytop, ybot, hw, squash, light_a):
        self.cv, self.rng = cv, rng
        self.cx, self.ytop, self.ybot = cx, ytop, ybot
        self.H = max(8.0, ybot - ytop)
        self.hw = hw                       # half-width at the widest point
        self.squash = squash               # how far above the rim the eye is
        ctrl = PROFILES[form]
        # 手拉坯 — a thrown wall is smooth, and smooth is the point. The only
        # irregularity is one slow wobble of well under a percent, which is
        # the hand on the rib rather than a bumpy pot; per-row noise here
        # reads as a cast flaw and ruins the form (song §1.4, the same
        # lesson at the other end of its scale).
        ph = rng.uniform(0, TAU)
        amp = rng.uniform(0.002, 0.006)
        n = max(16, int(self.H) + 1)
        self.n = n
        self.r = []
        for i in range(n + 1):
            t = i / n
            w = 1.0 + amp * math.sin(t * math.pi * rng.uniform(1.4, 2.6) + ph)
            self.r.append(max(0.5, _catmull(ctrl, t) * hw * w))
        self.rmax = max(self.r)
        self.rtop = self.r[n]
        self.rbot = self.r[0]
        # 口沿 — the mouth is an ellipse, and it is part of the *silhouette*,
        # not an ornament laid on top of one. Its centre and semi-minor axis
        # are wanted by the wall painter, the contour, the decoration clip
        # and the spray, so they are worked out once, here.
        self.ry = max(0.5, self.rtop * squash)
        self.mcy = self.ytop + self.ry
        # 笔 — the unit every decoration width is quoted in. A 回纹 is a
        # fixed fraction of the pot it is painted on, not a fixed number of
        # pixels: the first draft quoted line widths absolutely and the whole
        # scheme shrank to a scatter of dust on anything over 500 px.
        self.du = self.rmax / 52.0
        # 光 — gongbi's LIGHT is a screen vector; its horizontal component is
        # how far round the pot the window sits. Clamped off dead centre: a
        # light exactly on the axis gives a highlight down the middle of the
        # vessel, which is the one position that flattens it.
        s = math.cos(light_a)
        self.phiL = max(-1.05, min(1.05, math.asin(max(-0.92, min(0.92, s)))))
        if abs(self.phiL) < 0.34:
            self.phiL = 0.34 * (1.0 if self.phiL >= 0 else -1.0)

    # ---- geometry -------------------------------------------------------

    def y_of(self, t):
        return self.ybot - (self.ybot - self.ytop) * clamp01(t)

    def t_of(self, y):
        return clamp01((self.ybot - y) / (self.ybot - self.ytop))

    def r_of(self, t):
        f = clamp01(t) * self.n
        i = int(f)
        if i >= self.n:
            return self.r[self.n]
        g = f - i
        return self.r[i] + (self.r[i + 1] - self.r[i]) * g

    def wall_r(self, y):
        """Half-width of the *outer wall* on paper row `y`.

        Above the rim ellipse's two side points the vessel's silhouette is
        no longer the wall: it is the far half of the mouth, arching up
        behind the opening. Filling every row out to `r_of(t)` right up to
        `ytop` gives the pot a flat top, and what shows to either side of
        the opening is a pair of grey wings standing past the lip — which
        is exactly what draft 3 drew on the wide-mouthed 花觚.
        """
        r = self.r_of(self.t_of(y))
        d = (self.mcy - y) / self.ry
        if d > 0.0:
            if d >= 1.0:
                return 0.0
            r = min(r, self.rtop * math.sqrt(1.0 - d * d))
        return r

    def pt(self, phi, t):
        """Where `(azimuth, height)` lands on the paper."""
        return self.cx + math.sin(phi) * self.r_of(t), self.y_of(t)

    def visible(self, phi, slack=0.055):
        return -math.pi * 0.5 + slack < phi < math.pi * 0.5 - slack

    def off(self, phi, t, dx, dy):
        """`(phi, t)` for a point `dx` right and `dy` *up* from `(phi, t)`,
        both measured in paper-pixels along the pot's own surface.

        Azimuth and height are not the same unit, and a motif specified in
        them is stretched by whatever `r/H` happens to be: draft 2's 缠枝莲
        came out a zigzag and its lotus a flat fan for exactly that reason.
        Anything with a *shape* — a petal, a cloud head, a leaf — is built
        in these surface pixels, and only the band layout stays in `t`.
        """
        return phi + dx / max(1.0, self.r_of(t)), t + dy / self.H

    # ---- lighting -------------------------------------------------------

    def lit(self, u):
        """Diffuse value at `u = sin(phi)`, in 0..1. No specular: that is a
        separate pass laid over the decoration."""
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        phi = math.asin(u)
        d = math.cos(phi - self.phiL)
        if d < 0.0:
            d = 0.0
        # 0.20 ambient, because a pot on silk sits in a lit room and a
        # terminator that goes to black is a studio photograph, not 设色.
        # The exponent under 1 widens the lit face and tightens the roll-off,
        # which is what a *glazed* wall does: the sheen spreads the light.
        v = 0.20 + 0.80 * (d ** 0.78)
        # 边缘转深 — the glaze is seen edge-on at the silhouette and reads
        # thicker and darker there, on *both* flanks. This is the line that
        # makes the form turn away rather than end, and on a white ware it is
        # doing nearly all the modelling on its own: a 甜白 or 青花 body is
        # already paper-coloured, so there is no highlight to see and the
        # roundness has to be carried entirely by the two dark flanks.
        a = abs(u)
        if a > 0.72:
            v *= 1.0 - 0.52 * ((a - 0.72) / 0.28) ** 1.35
        return v

    def bounce(self, u):
        """反光 — light thrown back onto the shaded flank from the paper.

        Without it a vessel has a lit side and a dead side and reads as a
        half-cylinder stuck to the sheet. The band sits *inside* the dark
        silhouette edge, never on it, which is why it is keyed off `u`
        against the light's own sign rather than off the shading value.
        """
        s = -1.0 if self.phiL > 0 else 1.0
        z = (u * s - 0.42) / 0.44
        if z <= 0.0 or z >= 1.6:
            return 0.0
        return math.sin(min(1.0, z) * math.pi) ** 1.6

    def spec(self, u, t, gloss):
        """The window in the glaze: a narrow vertical streak, strongest on
        the belly where the wall is closest to vertical."""
        if u < -1.0:
            u = -1.0
        elif u > 1.0:
            u = 1.0
        d = math.cos(math.asin(u) - self.phiL)
        if d <= 0.0:
            return 0.0
        # Two lobes: a tight core that is the window itself and a broad
        # sheen around it. One lobe gives either a hard stripe (high
        # exponent) or a fog (low), and porcelain shows both at once.
        return gloss * (0.72 * d ** 26 + 0.28 * d ** 7) * \
            (0.45 + 0.55 * math.sin(clamp01(t) * math.pi))

    # ---- the fill -------------------------------------------------------

    def fill(self, ware, alpha=1.0):
        """釉色 — the body, with 积釉 pooling but no highlight yet."""
        cv = self.cv
        base, deep = rgb(ware["body"]), rgb(ware["deep"])
        pool = ware["pool"]
        # A value ramp is cheap and the pot is thousands of pixels wide, so
        # tabulate it once over u rather than per pixel.
        NU = 384
        ramp = []
        for i in range(NU + 1):
            u = -1.0 + 2.0 * i / NU
            ramp.append(self.lit(u))
        grain = Noise(self.rng, 6, 14)
        # 桔皮 — orange peel. A thick ash or copper glaze does not come out
        # of the kiln as a mirror; it is dimpled, and the dimples are what
        # tell the eye it is glass over clay rather than painted tin. The
        # field is sampled in `(azimuth, t)`, so the dimples foreshorten
        # into the silhouette with everything else.
        peel = Noise(self.rng, 22, 40)
        # keyed mostly off `pool`, not off gloss: it is the *thick* glazes
        # that dimple. A clear 青花 glaze is nearly a mirror, and draft 3's
        # gloss-led coefficient put the heaviest texture on the one ware
        # that should have none.
        pk = 0.07 * ware["gloss"] + 0.34 * pool
        y0 = max(0, int(self.ytop))
        y1 = min(cv.h, int(self.ybot) + 1)
        cx = self.cx
        for py in range(y0, y1):
            t = self.t_of(py + 0.5)
            # the wall is clipped by the mouth near the top, but the *light*
            # still runs over the full radius of the pot at that height, so
            # the two are kept apart: `rf` shades, `r` covers.
            rf = self.r_of(t)
            r = self.wall_r(py + 0.5)
            if r < 0.6:
                continue
            # 积釉 — glaze runs down and gathers where the wall turns in, so
            # the foot and the inside of the neck's curve are deeper. Taken
            # off the *profile's own slope*, which is where a potter would
            # point: it darkens exactly the concavities, and automatically
            # does nothing on a straight-walled 觚.
            dr = (self.r_of(min(1.0, t + 0.02)) - self.r_of(max(0.0, t - 0.02)))
            turn = clamp01(dr / max(1.0, self.rmax * 0.10))
            dk = pool * (0.55 * turn + 0.45 * clamp01((0.22 - t) / 0.22))
            inv = 1.0 / rf
            px0 = max(0, int(cx - r - 1))
            px1 = min(cv.w, int(cx + r + 2))
            for px in range(px0, px1):
                u = (px + 0.5 - cx) * inv
                cov = r - abs(px + 0.5 - cx) + 0.5
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                v = ramp[int((u + 1.0) * 0.5 * NU) if -1.0 <= u <= 1.0
                         else (0 if u < 0 else NU)]
                v *= 1.0 - dk
                v *= 1.0 + pk * (peel.at((u + 1.0) * 0.5, t) - 0.5)
                c = mix(deep, base, clamp01(v))
                cv.blend(px, py, c, cov * alpha)
        # 窑变 — a few soft tonal drifts in the glaze, or the wall is a
        # gradient and nothing else. Kept inside the silhouette.
        rng = self.rng
        for _ in range(int(self.H * 0.09)):
            t = rng.random()
            r = self.r_of(t)
            u = rng.uniform(-0.80, 0.80)
            rr = self.rmax * rng.uniform(0.14, 0.34)
            g = grain.at(rng.random(), t)
            disc(self.cv, cx + u * r, self.y_of(t), rr,
                 mix(base, deep, 0.35 + 0.3 * g), (0.05 + 0.13 * pool) * alpha,
                 squash=rng.uniform(0.5, 1.1), soft=0.9)
        # 流釉 — the glaze is a liquid and it ran. Each streak follows one
        # *azimuth*, so it curves with the wall and narrows into the
        # silhouette with it. Without them a thick glaze like 霁红 or 龙泉
        # comes out an airbrushed cone: a smooth ramp is the one thing a
        # 1300°C melt never is.
        hi = rgb(ware["hi"])
        # How much a glaze ran is `pool` and nothing else, so both the count
        # and the strength are quoted off it. Draft 3 used the full `deep`
        # at a flat 0.075 and gave the two white wares a wall of grey
        # smears — a clean 青花 jar looked like it had been left in a shed.
        strength = 0.030 + 0.13 * pool
        for _ in range(int(3 + 13 * pool)):
            u0 = rng.uniform(-0.86, 0.86)
            ta = rng.uniform(0.18, 0.97)
            tb = max(0.02, ta - rng.uniform(0.22, 0.70))
            wid = self.rmax * rng.uniform(0.05, 0.14)
            c = mix(base, deep, 0.45 + 0.4 * pool) if rng.random() < 0.58 \
                else mix(base, hi, 0.50)
            span = abs(self.y_of(tb) - self.y_of(ta))
            k = max(2, int(span / max(1.0, wid * 0.5)))
            for i in range(k + 1):
                tt = ta + (tb - ta) * i / k
                # fades in at the top and pools at the bottom of the run
                f = math.sin(math.pi * (i / k)) ** 0.55
                disc(self.cv, cx + u0 * self.r_of(tt), self.y_of(tt),
                     wid * (0.7 + 0.5 * (i / k)), c, strength * f * alpha,
                     soft=0.95)

    def gloss(self, ware):
        """The layer of glass: specular streak and 反光, over everything."""
        cv = self.cv
        hi = rgb(ware["hi"])
        g = ware["gloss"]
        base = rgb(ware["body"])
        bounce_c = mix(base, hi, 0.45)
        y0 = max(0, int(self.ytop))
        y1 = min(cv.h, int(self.ybot) + 1)
        cx = self.cx
        for py in range(y0, y1):
            t = self.t_of(py + 0.5)
            rf = self.r_of(t)
            r = self.wall_r(py + 0.5)
            if r < 0.6:
                continue
            inv = 1.0 / rf
            px0 = max(0, int(cx - r - 1))
            px1 = min(cv.w, int(cx + r + 2))
            for px in range(px0, px1):
                u = (px + 0.5 - cx) * inv
                cov = r - abs(px + 0.5 - cx) + 0.5
                if cov <= 0.0:
                    continue
                if cov > 1.0:
                    cov = 1.0
                sp = self.spec(u, t, g)
                bo = self.bounce(u) * (0.10 + 0.16 * g)
                if sp > 0.004:
                    cv.blend(px, py, hi, min(1.0, sp) * cov)
                if bo > 0.004:
                    cv.blend(px, py, bounce_c, bo * cov)

    def outline(self, ware, k=1.0):
        """勾线 — the contour. 工笔 draws the thing before it colours it, and
        a vessel without a line round it is a render, not a painting."""
        ink = mix(rgb(ware["deep"]), P("mo"), 0.55)
        w = max(0.75, self.H * 0.0022) * k
        # The contour is the wall's own edge and it stops where the wall
        # does — at the rim ellipse's side point. Run it to t=1 and it
        # crosses the opening and draws a line over the far rim.
        tend = self.t_of(self.mcy)
        for s in (-1.0, 1.0):
            pts = []
            steps = max(24, int(self.H / 3.0))
            for i in range(steps + 1):
                t = tend * i / steps
                pts.append((self.cx + s * self.r_of(t), self.y_of(t)))
            # 一边重一边轻 — the shaded flank carries the heavier line. Equal
            # weight on both sides is a technical drawing.
            lit_side = 1.0 if self.phiL > 0 else -1.0
            a = 0.30 if s == lit_side else 0.52
            brush_path(self.cv, pts, w * (0.85 if s == lit_side else 1.12),
                       w * (0.8 if s == lit_side else 1.0), ink, a)


# --------------------------------------------------------------------------
# 口 and 足 — the two ellipses that turn a silhouette into a vessel
# --------------------------------------------------------------------------

def mouth(cv, rng, body, ware, dark=True):
    """口 — the opening, seen from a little above.

    Three marks, and leaving out any one of them costs the whole illusion:
    the *cavity*, which is the inside of the pot and is the darkest thing in
    the painting; the *far rim*, which is the top edge of the wall seen
    across the hole and catches the light; and the *near rim*, the strip of
    glazed lip between the cavity and the outer contour. A single stroked
    ellipse gives a hole cut in paper.
    """
    rx = body.rtop
    ry = body.ry
    cy = body.mcy
    deep = rgb(ware["deep"])
    # 唇厚 — a thrown wall has a thickness, and the lip is the end-grain of
    # it. Without it the cavity runs to the contour and the mouth is a hole
    # punched in paper; with it there is a ring of glazed porcelain to
    # catch the light, which is the whole read of "you are looking into a
    # pot" rather than "through one".
    wall = max(1.2, rx * 0.085)
    ri = max(1.0, rx - wall)
    # The cavity is dark but it is not ink: an interior of white porcelain
    # in a lit room goes to a warm neutral, and taking it to black makes a
    # 花觚 look like a bowl of mud. Draft 3 did exactly that at 0.84.
    # How black it goes is a function of the *opening*, not a constant. A
    # 梅瓶's thumb-wide mouth is a slot and nothing comes back out of it; a
    # 花觚's trumpet shows you most of the far inner wall, lit. Flat at
    # 0.62 the beaker came out carrying a tub of tar.
    openness = clamp01((rx - 24.0) / 70.0)
    cav = mix(mix(deep, P("zheshi"), 0.22), P("mo"),
              (0.62 if dark else 0.32) - 0.24 * openness)
    disc(cv, body.cx, cy, ri, cav, 0.90, squash=body.squash)
    # 内壁 — what you see *across* the opening is the far inner wall, which
    # faces you and takes the light; straight down the throat, under the
    # near lip, is the one direction nothing comes back from. Draft 3 had
    # these two the other way round.
    disc(cv, body.cx, cy - ry * 0.34, ri * 0.82,
         mix(cav, mix(rgb(ware["body"]), deep, 0.52), 0.45 + 0.25 * openness),
         0.55 + 0.18 * openness, squash=body.squash, soft=0.55)
    disc(cv, body.cx, cy + ry * 0.40, ri * 0.74, mix(cav, P("mo"), 0.55),
         0.52, squash=body.squash * 0.92, soft=0.6)
    # the lip itself: a band between the outer ellipse and the cavity.
    #
    # `phiL` is an azimuth about the pot's axis, so the light's horizontal
    # position on the *rim ellipse* — where the parameter runs round the
    # ellipse rather than round the pot — is at `acos(sin(phiL))`. Using
    # `phiL` directly here puts the bright part of the lip a quarter turn
    # from the bright part of the wall under it.
    aL = math.acos(max(-1.0, min(1.0, math.sin(body.phiL))))
    rim_c = rgb(ware["rim"])
    # The band is walked at better than one sample per pixel of its own
    # width. Draft 3 stepped it at roughly its own disc radius and the lip
    # came out beaded, like a pie crust round the mouth.
    rm = rx - wall * 0.5
    w = max(0.9, wall * 0.62)
    steps = max(40, int(math.pi * rm / max(0.6, w * 0.45)))
    for i in range(steps + 1):
        a = math.pi * (i / steps)            # the near half of the ellipse
        x = body.cx + math.cos(a) * rm
        y = cy + math.sin(a) * rm * body.squash
        lit = 0.45 + 0.55 * clamp01(0.5 + 0.5 * math.cos(a - aL))
        disc(cv, x, y, w, mix(rim_c, rgb(ware["hi"]), 0.35 * lit),
             0.55 + 0.35 * lit)
    # the far rim, lit from behind the hole
    for i in range(steps + 1):
        a = math.pi + math.pi * (i / steps)
        x = body.cx + math.cos(a) * rm
        y = cy + math.sin(a) * rm * body.squash
        lit = 0.4 + 0.6 * clamp01(0.5 + 0.5 * math.cos(a + aL))
        disc(cv, x, y, w * 0.82,
             mix(rim_c, rgb(ware["hi"]), 0.55 * lit), 0.42 + 0.40 * lit)
    return cy


def foot(cv, rng, body, ware):
    """圈足 — the ring the pot stands on, and the shadow it sits in.

    The underside of the foot is never visible from eye level, so the mark
    is the *front* arc of the base ellipse plus the dark line where the
    unglazed ring meets the floor. The crisp dark bottom edge is also what
    stops a vase from looking as though it is floating.
    """
    rx = body.rbot
    ry = rx * body.squash
    cy = body.ybot - ry * 0.55
    ink = mix(rgb(ware["deep"]), P("mo"), 0.60)
    steps = max(20, int(rx * 0.8))
    pts = []
    for i in range(steps + 1):
        a = math.pi * (i / steps)
        pts.append((body.cx + math.cos(a) * rx, cy + math.sin(a) * ry))
    brush_path(cv, pts, max(0.9, rx * 0.055), max(0.9, rx * 0.055), ink, 0.62)
    # 露胎 — the biscuit foot ring, unglazed and warmer than the glaze
    biscuit = mix(P("zheshi"), rgb(ware["body"]), 0.45)
    for i in range(steps + 1):
        a = math.pi * (i / steps)
        x = body.cx + math.cos(a) * rx * 0.985
        y = cy + math.sin(a) * ry * 0.98 - max(1.0, rx * 0.055)
        disc(cv, x, y, max(0.8, rx * 0.05), biscuit, 0.30)
    # the contact shadow
    disc(cv, body.cx, cy + ry * 0.55, rx * 1.18,
         mix(P("danmo"), P("zheshi"), 0.4), 0.17,
         squash=body.squash * 0.78, soft=0.85)


def ears(cv, rng, body, ware):
    """贯耳 — the two tubular handles on the neck of a 贯耳瓶.

    They are cylinders stuck to a cylinder, so they are lit by the same
    azimuth field as the body: the near one catches the window, the far one
    does not. Sized off the neck, never off the sheet, or they drift.
    """
    t0, t1 = 0.700, 0.905
    rn = body.r_of((t0 + t1) * 0.5)
    y0, y1 = body.y_of(t1), body.y_of(t0)
    base, deep, hi = rgb(ware["body"]), rgb(ware["deep"]), rgb(ware["hi"])
    ink = mix(deep, P("mo"), 0.55)
    # 贯耳 are the feature the form is *named* for, so they are sized to be
    # seen: draft 3's tubes were a third of the neck's radius and read as
    # two small tabs stuck to the side.
    w = max(3.0, rn * 0.52)
    for s in (-1.0, 1.0):
        # 贯耳 are luted onto the *outside* of the neck, so the tube's axis
        # stands clear of the wall by about its own radius. Draft 1 put it
        # at 0.92 of the neck radius — inside the pot — and it came out as
        # a pale blister on the neck rather than a handle.
        x = body.cx + s * (rn + w * 0.42)
        # the tube is a little cylinder, lit across its own width
        for i in range(int(w) + 2):
            u = -1.0 + 2.0 * i / (int(w) + 1)
            cov = math.sqrt(max(0.0, 1.0 - u * u))
            if cov <= 0.02:
                continue
            # its own azimuth, offset by where it sits on the pot
            v = body.lit(max(-1.0, min(1.0, s * 0.78 + u * 0.38)))
            brush_seg(cv, x + u * w * 0.5, y0, x + u * w * 0.5, y1,
                      w / (int(w) + 1) * 1.3, w / (int(w) + 1) * 1.3,
                      mix(deep, base, clamp01(v)), 0.95)
        # the two ends, and the hole bored through
        for yy, sq in ((y0, body.squash * 1.5), (y1, body.squash * 1.5)):
            disc(cv, x, yy, w * 0.5, mix(mix(deep, base, 0.75), hi, 0.18),
                 0.95, squash=sq)
            disc(cv, x, yy, w * 0.27, mix(deep, P("mo"), 0.55), 0.80,
                 squash=sq)
        for d in (-1.0, 1.0):
            brush_seg(cv, x + d * w * 0.5, y0, x + d * w * 0.5, y1,
                      max(0.7, w * 0.07), max(0.7, w * 0.07), ink, 0.45)


# --------------------------------------------------------------------------
# 纹饰 — decoration, drawn in azimuth space so it wraps the wall
# --------------------------------------------------------------------------

def _mark(cv, body, ware, phi, t, r, alpha, c=None):
    """One dot of decoration, shaded and hidden by the body's own curvature.

    Everything in this section goes through here, which is the point: a
    motif is specified in `(phi, t)` and never in pixels, so it compresses
    into the silhouette, disappears round the back, and takes its value from
    the same light as the glaze under it. 青花 is *under* the glaze, so it
    darkens with the wall; a decal does not, and that is exactly what the
    eye catches.

    `r` is quoted in `body.du`, the pot's own brush unit — see `Body.du`.
    """
    if not body.visible(phi):
        return
    x, y = body.pt(phi, t)
    # a band drawn right under the lip runs off the wall at the flanks,
    # where the silhouette has already become the mouth ellipse
    if y < body.mcy and abs(x - body.cx) > body.wall_r(y):
        return
    v = body.lit(math.sin(phi))
    col = rgb(ware["ink"]) if c is None else c
    disc(cv, x, y, r * body.du,
         mix(mix(col, rgb(ware["deep"]), 0.35), col, clamp01(v)),
         alpha * (0.40 + 0.60 * v))


def _stroke(cv, body, ware, pts, w0, w1, alpha, c=None):
    """A run of `(phi, t)` samples laid down as one tapered mark.

    The samples are *re-spaced on the paper* before they are drawn. A motif
    is specified by as few control points as it takes to say what it is — a
    fret's upright is two — and two dots a centimetre apart is not a line.
    Densifying here rather than in each motif also means the azimuth
    compression near the silhouette is paid for automatically: the same
    segment gets more samples when it is drawn face-on and fewer when it is
    seen edge-on, which is the opposite of what fixed-step sampling does.
    """
    n = len(pts) - 1
    if n < 1:
        return
    step = max(0.7, body.du * 0.55)
    for i in range(n):
        p0, p1 = pts[i], pts[i + 1]
        x0, y0 = body.pt(*p0)
        x1, y1 = body.pt(*p1)
        k = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for j in range(k):
            g = j / k
            f = (i + g) / n
            _mark(cv, body, ware, p0[0] + (p1[0] - p0[0]) * g,
                  p0[1] + (p1[1] - p0[1]) * g, w0 + (w1 - w0) * f, alpha, c)
    _mark(cv, body, ware, pts[-1][0], pts[-1][1], w1, alpha, c)


def bowstring(cv, body, ware, t, w=1.0, alpha=0.55, c=None):
    """弦纹 — one incised or painted line clean round the pot."""
    steps = max(24, int(body.r_of(t) * 1.6))
    pts = [(-1.56 + 3.12 * i / steps, t) for i in range(steps + 1)]
    _stroke(cv, body, ware, pts, w, w, alpha, c)


def fret(cv, body, ware, t, n=18, h=0.018, alpha=0.6):
    """回纹 — the key-fret band, the border that frames every other band."""
    for k in range(n):
        p0 = -math.pi * 0.5 + math.pi * (k + 0.12) / n
        p1 = -math.pi * 0.5 + math.pi * (k + 0.88) / n
        if not body.visible(p0):
            continue
        up = (k % 2 == 0)
        a, b = (t, t + h) if up else (t + h, t)
        _stroke(cv, body, ware, [(p0, a), (p1, a)], 0.9, 0.9, alpha)
        _stroke(cv, body, ware, [(p1, a), (p1, b)], 0.9, 0.9, alpha)


def plantain(cv, body, ware, t0, t1, n=9, alpha=0.62):
    """蕉叶纹 — the ring of upright plantain leaves that stands on a neck.

    A tall lens with a midrib, one per slot of azimuth, alternating long and
    short. The alternation is not decoration-for-its-own-sake: a 蕉叶 band
    on a real neck interleaves a tall leaf and a short one so the band has
    two edges rather than a comb's single straight line.
    """
    ink = rgb(ware["ink"])
    pale = mix(ink, rgb(ware["body"]), 0.55)
    hpx = (t1 - t0) * body.H
    wpx = TAU / n * body.r_of((t0 + t1) * 0.5) * 0.46
    for k in range(n):
        phi = -math.pi + TAU * (k + 0.5) / n
        if not body.visible(phi):
            continue
        tall = (k % 2 == 0)
        L = hpx * (1.0 if tall else 0.62)
        _lens(cv, body, ware, phi, t0, math.pi * 0.5, L, wpx,
              1.1, 0.40, alpha, ink, fill=pale if tall else None)
        _stroke(cv, body, ware, [(phi, t0), body.off(phi, t0, 0.0, L * 0.92)],
                0.9, 0.3, alpha * 0.85, ink)


def ruyi(cv, body, ware, t, n=7, h=0.045, alpha=0.6):
    """如意云头 — the lappet collar on a shoulder: a ring of cloud heads.

    Each head hangs *down* from the band line: two lobes swelling out from
    the shoulders and meeting in a point below, with an inner echo of the
    same outline drawn a little inside it. The echo is what makes it read as
    a 云头 and not as a scallop — the doubled contour is the motif.
    """
    ink = rgb(ware["ink"])
    # 宽高相称 — a 云头 is about as deep as it is half-wide. `h` is the room
    # the band has, not the shape's proportion, so the *count* is what gives
    # way: if seven heads would each be 100 px across in a 15 px band, draw
    # more and narrower ones. Draft 2 honoured `n` and produced a row of
    # flat chevrons.
    hpx = max(4.0, h * body.H)
    circ = TAU * body.r_of(t) * 0.47
    # A lappet is roughly three parts wide to two deep, so the count that
    # fits is the one that makes its *slot* about 1.4 times the band's
    # depth. Draft 3 used a coefficient that left each head three times as
    # wide as it was deep and the collar came out a shallow running wave.
    n = max(5, min(16, int(round(1.45 * circ / hpx))))
    # and the half-width is pulled inside its slot, because the bow throws
    # the lobe out past this line: at the full half-slot, adjacent heads
    # merge into one continuous garland.
    wpx = circ / n * 0.70
    pale = mix(ink, rgb(ware["body"]), 0.58)

    def lobe(s, sc, f):
        """A point `f` of the way along one lobe, in surface pixels from the
        head's own top-centre: shoulder -> tip, bowed along the *perpendicular*
        of that chord. Draft 2 bowed it along x, which on a wide shallow
        lappet is almost along the chord itself, so the lobe stretched
        instead of swelling and the collar came out a row of chevrons."""
        sx, sy = s * wpx * sc, 0.0
        dx, dy = -sx, -hpx * sc
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * s, dx / ln * s
        # the swell is biased toward the shoulder so the lobes still meet in
        # a *point*: a symmetric bow rounds both ends equally and the collar
        # comes out a row of horseshoes.
        b = math.sin(math.pi * f ** 0.70) * ln * 0.38
        return sx + dx * f + nx * b, sy + dy * f + ny * b

    for k in range(n):
        phi = -math.pi + TAU * (k + 0.5) / n
        if not body.visible(phi):
            continue
        # 分水 — flood the head, then outline it. Two contours round an
        # empty middle read as a chain of loops; the wash is what makes a
        # row of these hang as lappets. It is laid between the two lobes at
        # matching `f`, so it takes the cloud's own swelling silhouette —
        # a straight taper here paints a trapezoid, which is worse than no
        # wash at all.
        # chords every half brush-unit: spaced by a fixed count they leave
        # the head ruled with pale stripes instead of flooded
        nf = max(6, min(64, int(hpx / max(0.6, body.du * 0.45))))
        for i in range(1, nf + 1):
            f = i / (nf + 1.0)
            lx, ly = lobe(-1.0, 1.0, f)
            rx_, _ry = lobe(1.0, 1.0, f)
            _stroke(cv, body, ware,
                    [body.off(phi, t, lx, ly), body.off(phi, t, rx_, ly)],
                    1.0, 1.0, alpha * 0.34, pale)
        for sc, w, al in ((1.0, 0.95, 1.0), (0.60, 0.70, 0.72)):
            for s in (-1.0, 1.0):
                pts = [body.off(phi, t, *lobe(s, sc, i / 12.0))
                       for i in range(13)]
                _stroke(cv, body, ware, pts, w, w * 0.8, alpha * al, ink)


def scroll(cv, body, ware, t0, t1, n=8, alpha=0.70):
    """缠枝莲 — the scrolling lotus that fills the belly of a 青花 vase.

    One sine of vine running round the pot, a flower head at every crest, a
    leaf in every trough, and a curl of tendril off each. The vine is drawn
    across a *wider* azimuth range than is visible and clipped by `_mark`,
    so the pattern genuinely continues round the back instead of being
    squeezed to fit the front.
    """
    tm = (t0 + t1) * 0.5
    # `n` counts crests round the *whole* pot, so the visible half-circle
    # shows n/2 of them. Four or five across the front is a scroll; two is a
    # ribbon, which is what draft 1 drew. The amplitude is deliberately a
    # sixth of the band and not a third: the vine's job is to carry the
    # heads, and a vine that swings the full height of its own band leaves
    # no room for them and reads as a zigzag in its own right.
    amp = (t1 - t0) * 0.11
    half = math.pi / n                     # azimuth owned by one crest
    hpx = (t1 - t0) * body.H               # the band, in paper pixels
    steps = 240
    vine = []
    for i in range(steps + 1):
        phi = -2.0 + 4.0 * i / steps
        vine.append((phi, tm + amp * math.sin(phi * n)))
    _stroke(cv, body, ware, vine, 0.7, 0.7, alpha * 0.60)
    ink = rgb(ware["ink"])
    pale = mix(ink, rgb(ware["body"]), 0.50)
    # the head is as big as the band and the crest spacing will let it be
    R = min(hpx * 0.40, half * 0.78 * body.r_of(tm))
    k = -n
    while k <= n:
        crest = (math.pi * 0.5 + k * math.pi) / n
        up = (k % 2 == 0)
        k += 1
        if abs(crest) > 2.0:
            continue
        ct = tm + amp * (1.0 if up else -1.0)
        if up:
            _lotus(cv, body, ware, crest, ct, R, alpha, ink, pale)
        else:
            _scroll_leaf(cv, body, ware, crest, ct, R * 0.92, alpha, ink)
        # 卷草 — the tendril that makes it a *scroll* and not a garland
        for s in (-1.0, 1.0):
            cur = []
            for i in range(10):
                f = i / 9.0
                cur.append(body.off(crest, ct, s * R * 1.5 * f,
                                    (-1.0 if up else 1.0) * R * 0.9
                                    * math.sin(f * 2.4)))
            _stroke(cv, body, ware, cur, 0.65, 0.25, alpha * 0.65)


def _lens(cv, body, ware, phi, t, a, L, wid, w0, w1, alpha, c, fill=None):
    """One petal/leaf: a lens from `(phi, t)` pointing along `a`, `L` px long.

    Two bowed edges that start together, part to `wid`, and meet again at
    the tip. Nearly all 青花 drawing is made of this one shape — petal,
    sepal, leaf, cloud lobe — so it is worth having once and building the
    motifs out of it rather than out of free strokes. `a` is measured in
    surface pixels, so 0 is round the pot and +pi/2 is up the pot.
    """
    ca, sa = math.cos(a), math.sin(a)
    for s in (-1.0, 1.0):
        pts = []
        for i in range(11):
            f = i / 10.0
            # the 0.7 power pushes the widest point past halfway, which is
            # where a lotus petal's shoulder actually sits
            o = s * wid * math.sin(math.pi * f) ** 0.70
            pts.append(body.off(phi, t, ca * L * f - sa * o,
                                sa * L * f + ca * o))
        _stroke(cv, body, ware, pts, w0, w1, alpha, c)
    if fill is not None:
        # 分水 is a *flood*. The chords are spaced on the paper, not by a
        # fixed count: six of them across a big petal rules it with stripes,
        # which is the one thing a wash must not look like.
        nf = max(5, min(64, int(L / max(0.6, body.du * 0.45))))
        for i in range(1, nf + 1):
            f = i / (nf + 1.0)
            o = wid * math.sin(math.pi * f) ** 0.70
            p0 = body.off(phi, t, ca * L * f + sa * o, sa * L * f - ca * o)
            p1 = body.off(phi, t, ca * L * f - sa * o, sa * L * f + ca * o)
            _stroke(cv, body, ware, [p0, p1], 0.9, 0.9, alpha * 0.42, fill)


def _lotus(cv, body, ware, phi, t, R, alpha, ink, pale):
    """One 莲 head of a 缠枝莲, face on: a rosette seen from above.

    Two rings and a boss. The outer five petals are washed pale and splay
    outward, the inner three stand up in full cobalt. 分水 — the 青花
    painter outlines and then floods — so the outer ring is drawn light and
    filled lighter, and only the middle of the flower is at full strength.
    """
    for i in range(5):
        a = math.pi * 0.5 + (i - 2) * 0.80
        _lens(cv, body, ware, phi, t, a, R, R * 0.30, 0.85, 0.40,
              alpha * 0.80, pale, fill=pale)
    for i in range(3):
        a = math.pi * 0.5 + (i - 1) * 0.72
        _lens(cv, body, ware, phi, t, a, R * 0.60, R * 0.22, 0.95, 0.45,
              alpha, ink)
    _mark(cv, body, ware, phi, t, 1.4, alpha, ink)


def _scroll_leaf(cv, body, ware, phi, t, R, alpha, ink):
    """The curled leaf pair that sits in each trough of the vine."""
    for s in (-1.0, 1.0):
        _lens(cv, body, ware, phi, t, -math.pi * 0.5 + s * 0.58,
              R * 1.05, R * 0.20, 0.95, 0.35, alpha * 0.9, ink)


def qinghua(cv, rng, body, ware):
    """The whole 青花 scheme, laid out by what part of the pot it is on.

    Bands are not decoration *placed* on a vase, they are decoration placed
    on its *parts*: a fret on the foot, a scroll on the belly, lappets on
    the shoulder, plantain on the neck, a line under the lip. The heights
    come out of the profile, so the scheme re-fits itself to a 觚 and a
    梅瓶 without a table of special cases.
    """
    # 分段 — the pot is divided before anything is drawn on it. `tb` is the
    # widest point and `neck` the first height above it where the wall has
    # pulled in to half its maximum; those two numbers are enough to find
    # every band on any of the five forms.
    tb = max(range(body.n + 1), key=lambda i: body.r[i]) / body.n
    # 觚形 — a beaker is widest at the *lip*, so `tb` lands at the top and
    # every band below it computes to nothing. Draft 5 shipped a 花觚 with
    # a fret round its foot and three hands of bare white above it. A
    # beaker is recognised by its *waist*, not by its widest point: an
    # interior minimum far narrower than either end. It then divides into
    # the three registers of the bronze 觚 it copies, not into belly,
    # shoulder and neck, which it does not have.
    i0, i1 = int(body.n * 0.20), int(body.n * 0.80)
    tw = min(range(i0, i1 + 1), key=lambda i: body.r[i]) / body.n
    if body.r_of(tw) < 0.78 * min(body.r_of(0.06), body.r_of(0.94)):
        _gu(cv, rng, body, ware, tw)
        return
    neck = tb
    while neck < 0.96 and body.r_of(neck) > body.rmax * 0.52:
        neck += 0.01
    # The main band is a *band*, roughly a third of the pot, centred on the
    # belly. Draft 1 ran it from the foot to the neck: a scroll stretched
    # over half the height has crests a hand apart and reads as a ribbon
    # tied round the vase, not as 缠枝莲.
    belly0 = max(0.11, tb - 0.17)
    belly1 = min(neck - 0.05, tb + 0.16)
    if belly1 - belly0 > 0.12:
        bowstring(cv, body, ware, belly0 - 0.018, 0.75, 0.55)
        bowstring(cv, body, ware, belly1 + 0.018, 0.75, 0.55)
        scroll(cv, body, ware, belly0, belly1, n=rng.choice((5, 6, 7)))
        # 如意云头 — the lappet collar hangs from the shoulder *into* the
        # space above the main band, so it is placed off `belly1` and not
        # off the neck: on a 梅瓶 the shoulder and the neck are 0.1 apart
        # and on a 胆瓶 they are 0.3.
        sh = belly1 + 0.055
        if neck - sh > 0.045:
            ruyi(cv, body, ware, sh, n=rng.choice((6, 7, 8)),
                 h=min(0.072, (neck - sh) * 0.80))
    if 0.94 - neck > 0.085:
        plantain(cv, body, ware, neck + 0.035, 0.935,
                 n=rng.choice((8, 9, 10)))
    fret(cv, body, ware, 0.040, n=rng.choice((14, 16, 18)), h=0.014)
    bowstring(cv, body, ware, 0.082, 0.7, 0.45)
    bowstring(cv, body, ware, 0.972, 0.8, 0.50)


def _gu(cv, rng, body, ware, tw):
    """The 青花 scheme for a beaker: trumpet, waist drum, splayed foot.

    A 觚 has three registers and they are read as three, so the waist
    carries the one figured band and both flares carry 蕉叶 standing on
    their narrow end. Running a 缠枝莲 up a flare instead puts the scroll's
    crests a finger apart at one end and a hand apart at the other.
    """
    b0 = max(0.10, tw - 0.11)
    b1 = min(0.90, tw + 0.10)
    bowstring(cv, body, ware, b0 - 0.016, 0.75, 0.55)
    bowstring(cv, body, ware, b1 + 0.016, 0.75, 0.55)
    scroll(cv, body, ware, b0, b1, n=rng.choice((5, 6)))
    plantain(cv, body, ware, b1 + 0.055, 0.925, n=rng.choice((9, 10, 11)))
    lo = max(0.075, b0 - 0.21)
    if b0 - 0.045 - lo > 0.09:
        plantain(cv, body, ware, lo, b0 - 0.045, n=rng.choice((9, 10, 11)))
    fret(cv, body, ware, 0.030, n=rng.choice((16, 18, 20)), h=0.013)
    bowstring(cv, body, ware, 0.972, 0.8, 0.50)


def incise(cv, rng, body, ware):
    """划花 — combed and incised decoration under a 青瓷 or 甜白 glaze.

    The glaze pools in the cut, so the line is *darker* than the wall on a
    celadon and barely visible on a white ware. It is never an outline: it
    is a groove, so it gets no ink, only the glaze's own colour deepened.
    """
    c = mix(rgb(ware["body"]), rgb(ware["deep"]), 0.60)
    tb = max(range(body.n + 1), key=lambda i: body.r[i]) / body.n
    bowstring(cv, body, ware, min(0.95, tb + 0.20), 1.3, 0.40, c)
    bowstring(cv, body, ware, max(0.06, tb - 0.24), 1.3, 0.40, c)
    # 篦划 — a comb of *parallel* curved cuts round the belly. The comb is
    # one tool with many teeth, so the cuts never cross; draft 3 gave each
    # one a wide S-swing in azimuth and they tangled into a scrawl in the
    # middle of the pot that read as graffiti.
    n = rng.randint(9, 14)
    lean = rng.uniform(0.10, 0.16) * (1.0 if rng.random() < 0.5 else -1.0)
    t0 = max(0.05, tb - 0.16)
    for k in range(n):
        phi0 = -1.70 + 3.4 * (k + 0.5) / n
        pts = []
        for i in range(10):
            f = i / 9.0
            pts.append((phi0 + lean * f * f, t0 + 0.27 * f))
        _stroke(cv, body, ware, pts, 1.0, 0.6, 0.30, c)


def crackle(cv, rng, body, ware, k=1.0):
    """开片 — the craze. 金丝铁线: iron wires and golden threads.

    A craze is a *graph*, not a texture: a crack runs until it meets
    another crack and stops there, and what the eye actually reads is the
    cells between them. Two independent populations of free-floating
    segments do not make cells — draft 4 drew exactly that, and a 哥窑 jar
    came out strewn with straw.

    So the long dark 铁线 are laid down first and *kept*, and every short
    pale 金丝 is then drawn from a point on one of them to a point on
    another. It costs one list of nodes and buys the whole pattern: a line
    with both ends anchored reads as a crack, and a line with two free ends
    never does.
    """
    iron = mix(rgb(ware["deep"]), P("mo"), 0.45)
    gold = mix(P("zheshi"), P("jin"), 0.45)
    nodes = []                       # (which crack, phi, t, surface x, y)
    # The walk steps in *surface pixels*, through `off`, like every other
    # motif here. Stepping in raw `(phi, t)` makes the stride six times
    # longer vertically than horizontally on a slim pot, and the craze
    # combs itself into a set of parallel verticals.
    step = max(1.0, body.du * (1.5 + 1.5 * k))
    # `k` is how loudly the ware crazes, and it has to move three things at
    # once: the alpha, the count, and the *length*. Moving the count alone
    # turns a 龙泉's faint net into twenty full-strength cracks, and a long
    # walk at any alpha reads as a scratch rather than a cell wall — what
    # the eye calls 开片 is small cells, not long lines.
    av = k ** 0.85
    run = int(5 + 11 * k)
    for ci in range(max(6, int(44 + 44 * k))):
        phi = rng.uniform(-1.45, 1.45)
        t = rng.uniform(0.04, 0.96)
        a = rng.uniform(0, TAU)
        pts = [(phi, t)]
        for _ in range(run):
            a += rng.gauss(0.0, 0.13)
            phi, t = body.off(phi, t, math.cos(a) * step,
                              math.sin(a) * step)
            if not (0.015 < t < 0.985) or abs(phi) > 1.5:
                break
            pts.append((phi, t))
        if len(pts) < 2:
            continue
        _stroke(cv, body, ware, pts, 0.80, 0.80, 0.42 * av, iron)
        for p, q in pts:
            nodes.append((ci, p, q, p * body.r_of(q), q * body.H))
    if len(nodes) < 8:
        return
    # 金丝 is 哥窑's and nobody else's: a celadon crazes in one colour, and
    # threading gold through a 龙泉 paints a warmer second net over the
    # first until the glaze stops looking like one glaze. The *linking*
    # still has to happen on every ware, though — cracks with two free ends
    # are dashes, and dashes are straw. So a quiet ware links in its own
    # iron, and only a loud one links in gold.
    thread = gold if k >= 0.6 else iron
    # Each thread runs to its *nearest* node on another crack. Picking the
    # pair at random instead lets the threads pile into whatever corner the
    # 铁线 happened to cluster in and leaves the rest of the pot bare — a
    # cobweb in one place and nothing anywhere else.
    lo, hi = body.du * 2.5, body.du * 16.0
    for i0 in range(0, len(nodes), 3):
        if rng.random() > 0.58:
            continue
        ci, p0, q0, ax, ay = nodes[i0]
        best = None
        for cj, p1, q1, bx, by in nodes:
            if cj == ci:
                continue
            d = math.hypot(bx - ax, by - ay)
            if lo <= d <= hi and (best is None or d < best[0]):
                best = (d, p1, q1, bx - ax, by - ay)
        if best is None:
            continue
        d, p1, q1, dxs, dys = best
        bow = rng.gauss(0.0, 0.11)
        pts = []
        for i in range(7):
            f = i / 6.0
            b = math.sin(math.pi * f) * d * bow
            pts.append(body.off(p0 + (p1 - p0) * f, q0 + (q1 - q0) * f,
                                -dys / d * b, dxs / d * b))
        _stroke(cv, body, ware, pts, 0.58, 0.58, 0.36 * av, thread)


# --------------------------------------------------------------------------
# 木座 and 果 — what the vessel stands on, and what stands beside it
# --------------------------------------------------------------------------

def stand(cv, rng, body, S):
    """木座 — the carved hardwood base a 清供 vessel always sits on.

    Three parts and no more: a top rail the pot sits in, an apron with a
    scalloped lower edge, and the feet. It is deliberately small and dark —
    it exists to lift the vessel off the paper and to give the bottom of the
    composition a horizontal, and a stand that competes with the pot it
    carries is a stand nobody asked for.
    """
    # A 木座 is cut to its vessel's foot, but it also has to clear the
    # vessel's belly or the pot looks as though it is standing on a coin.
    # Off `rbot` alone a 花觚 — whose foot is nearly its widest point — gets
    # a stand half again as wide as the whole painting's subject.
    rx = max(body.rbot * 1.20, min(body.rbot * 1.46, body.rmax * 1.05))
    ry = rx * body.squash
    cy = body.ybot + ry * 0.10
    aL = math.acos(max(-1.0, min(1.0, math.sin(body.phiL))))
    wood_l = mix(P("zheshi"), P("mo"), 0.42)
    wood_d = mix(P("mo"), P("yanzhi"), 0.22)
    hgt = max(6.0, rx * 0.34)
    # top rail — an ellipse seen the same way as the foot, with a lip
    disc(cv, body.cx, cy, rx, mix(wood_l, P("mo"), 0.25), 0.92,
         squash=body.squash)
    # The apron is a *solid* band with a scalloped bottom, and the two are
    # drawn as one column per step so the scallop is a silhouette rather
    # than an outline. Draft 1 let the scallop swing 0.55–1.00 of the height
    # with a cusp at each node, and a 木座 with deep cusps all round reads
    # as a fringed skirt, not as carved hardwood: 壸门 are shallow.
    steps = max(40, int(rx * 1.6))
    cw = max(0.8, rx * 2.0 / steps * 0.80)
    for i in range(steps + 1):
        a = math.pi * (i / steps)
        x = body.cx + math.cos(a) * rx
        y0 = cy + math.sin(a) * ry
        lit = 0.35 + 0.65 * clamp01(0.5 + 0.5 * math.cos(a - aL))
        c = mix(wood_d, wood_l, lit)
        f = i / steps
        hh = hgt * (0.88 + 0.12 * math.cos(f * math.pi * 6.0))
        brush_seg(cv, x, y0, x, y0 + hh, cw, cw, c, 0.96)
        # the dark line along the carved edge
        disc(cv, x, y0 + hh, cw * 0.75, mix(wood_d, P("mo"), 0.45), 0.55)
    # 如意足 — three feet, the middle one at the front
    for f in (0.14, 0.5, 0.86):
        a = math.pi * f
        x = body.cx + math.cos(a) * rx * 0.88
        y = cy + math.sin(a) * ry * 0.88 + hgt * 0.92
        lit = 0.35 + 0.65 * clamp01(0.5 + 0.5 * math.cos(a - aL))
        disc(cv, x, y, hgt * 0.34, mix(wood_d, wood_l, lit), 0.94,
             squash=1.35)
    disc(cv, body.cx, cy + ry + hgt * 1.15, rx * 1.25,
         mix(P("danmo"), P("zheshi"), 0.35), 0.16,
         squash=body.squash * 0.7, soft=0.9)
    return cy + ry + hgt


def fruit(cv, rng, x, y, r, S, kind=None):
    """果 — a persimmon or a pomegranate set beside the vessel.

    A 清供 is an offering, and an offering is a vessel *and* something to
    offer. Two or three fruit at the foot also do a compositional job: they
    put a small round weight on the paper beside a tall vertical one, which
    is the oldest still-life device there is.
    """
    kind = kind or rng.choice(("柿", "石榴"))
    if kind == "柿":
        base, deep = rgb("#E08A22"), rgb("#8C3A08")
    else:
        base, deep = rgb("#D4604A"), rgb("#7A1E18")
    hi = mix(base, P("baifen"), 0.55)
    sq = 0.82 if kind == "柿" else 0.92
    # the body, shaded on the same azimuth as the vase
    lx = -0.55 if rng.random() < 0.5 else 0.55
    disc(cv, x, y, r, mix(base, deep, 0.35), 0.95, squash=sq)
    disc(cv, x + lx * r * 0.34, y - r * 0.30, r * 0.74, base, 0.85, squash=sq,
         soft=0.5)
    disc(cv, x + lx * r * 0.46, y - r * 0.40, r * 0.34, hi, 0.45, squash=sq,
         soft=0.9)
    disc(cv, x - lx * r * 0.52, y + r * 0.22, r * 0.52, deep, 0.28, squash=sq,
         soft=0.9)
    brush_path(cv, [(x - r * 0.96, y + r * 0.1 * sq),
                    (x, y + r * sq * 0.99), (x + r * 0.96, y + r * 0.1 * sq)],
               0.9, 0.9, mix(deep, P("mo"), 0.4), 0.35)
    if kind == "柿":
        # 柿蒂 — the four-lobed calyx, which is the whole of its character
        g_l = mix(P("caolv"), P("mo"), 0.35)
        for k in range(4):
            a = k * math.pi * 0.5 + rng.uniform(-0.2, 0.2)
            brush_path(cv, [(x, y - r * sq * 0.85),
                            (x + math.cos(a) * r * 0.42,
                             y - r * sq * 0.80 + math.sin(a) * r * 0.26),
                            (x + math.cos(a) * r * 0.62,
                             y - r * sq * 0.72 + math.sin(a) * r * 0.40)],
                       r * 0.17, r * 0.05, g_l, 0.85)
        disc(cv, x, y - r * sq * 0.86, r * 0.13, mix(g_l, P("mo"), 0.4), 0.8)
    else:
        # 石榴 — the crown of sepals at the blossom end
        for k in range(5):
            a = -math.pi * 0.5 + (k - 2) * 0.34
            brush_seg(cv, x, y - r * sq * 0.82,
                      x + math.cos(a) * r * 0.30,
                      y - r * sq * 0.82 + math.sin(a) * r * 0.38,
                      r * 0.10, r * 0.03, deep, 0.9)


# --------------------------------------------------------------------------
# 插花 — the spray, on the 三才 plan
# --------------------------------------------------------------------------
#
# 天地人. A branch arrangement is three stems and the three are not equal:
# 主枝, the tall one, sets the height and stands nearly upright; 客枝, two
# thirds of it, leans well out to one side; 使枝, a third of it, comes
# forward and down on the other. Every other length and angle is read off
# those. Three stems of similar length fanned symmetrically is a bouquet
# from a shop, and it is what this produced before the plan went in.

THREE = [
    # (length, lean out, how far the tip is allowed to wander)
    (1.00, 0.10, 0.30),      # 主 — heaven
    (0.66, 0.72, 0.42),      # 客 — man
    (0.34, 1.18, 0.50),      # 使 — earth
    (0.52, 0.95, 0.46),      # a fourth, for 满堂 only
]

# Which side of the axis each of them takes, as a multiple of the one `side`
# the arrangement picks. 主 and 客 share a side and 使 counterweights it: the
# whole point of 三才 is that the mass is *off* centre and one short stem
# holds it. Alternating sides — which is what draft 1 did, 客 left, 使 right
# — gives a symmetric V, and a symmetric V is a shop bouquet.
SIDES = (1.0, 1.0, -1.0, -1.0)


def stems(rng, body, mcy, S, reach, n):
    """Where the wood goes: `n` skeletons rising out of the mouth.

    They start *inside* the pot — a little below the rim plane and within
    the mouth's own radius — so that the near lip, redrawn afterwards, cuts
    them off exactly where a real stem disappears into a vase. Started at
    the rim they sprout from the edge of the hole like candles on a cake.
    """
    rm = body.rtop
    side = rng.choice((1.0, -1.0))
    out = []
    for i in range(n):
        L, lean, wander = THREE[i if i < len(THREE) else -1]
        ln = body.H * reach * L
        s = side * SIDES[i if i < len(SIDES) else -1]
        x0 = body.cx + s * rm * rng.uniform(0.12, 0.52)
        y0 = mcy + rm * body.squash * rng.uniform(0.1, 0.6)
        a = -math.pi * 0.5 + s * lean * rng.uniform(0.80, 1.18)
        # 使枝 goes out and *down*: it is the stem that reaches below the
        # mouth and ties the flowers back to the vessel.
        if i == 2:
            a = s * rng.uniform(0.20, 0.62) + (0.0 if s > 0 else math.pi)
        tx = x0 + math.cos(a) * ln
        ty = y0 + math.sin(a) * ln
        # the curve: stems leave the mouth nearly vertical whatever they do
        # later, because the neck of the pot holds them there.
        q1 = (x0 + math.cos(-math.pi * 0.5 + s * 0.12) * ln * 0.34,
              y0 + math.sin(-math.pi * 0.5 + s * 0.12) * ln * 0.34)
        q2 = (tx - math.cos(a + s * rng.uniform(0.5, 1.1)) * ln * 0.32,
              ty - math.sin(a + s * rng.uniform(0.5, 1.1)) * ln * 0.32)
        tx += rng.gauss(0, ln * wander * 0.16)
        ty += rng.gauss(0, ln * wander * 0.12)
        path, _ = knuckle(rng, bezier((x0, y0), q1, q2, (tx, ty), 26),
                          k=rng.randint(2, 3), amp=0.17, start=0.22)
        out.append(path)
    return out


def on_pot(body, mcy, x, y, pad=0.0):
    """Is `(x, y)` lying on the face of the vessel?

    A cut stem crossing in front of its own vase is normal and is left
    alone — it is how the eye knows the stems go *into* the pot. A leaf or
    a blossom lying flat on the belly is not: it has no stalk going
    anywhere, it covers the 青花, and it reads as a sticker. So the wood is
    free to cross and the foliage is not.
    """
    if y < mcy + body.rtop * body.squash * 0.5 or y > body.ybot:
        return False
    # `pad` widens or narrows the test. Negative for a blossom, which may
    # overlap the rim a little and still read correctly — a head is opaque
    # and obviously in front. Positive for a leaf, by half the blade's own
    # width, because the caller can only hand over points on the leaf's
    # *axis* and a leaf is a wide thing: a 菊花 blade at `wr` 0.6 whose
    # midrib runs a hair clear of the silhouette still lies half across the
    # neck, and being a thin translucent wash it reads as a decal with the
    # red showing through it.
    #
    # Either way it is *bounded* below at 0.45 r: a 荷叶 wider than the vase
    # would otherwise shrink the test radius past zero and be declared off a
    # pot it is lying flat across. That is how the 贯耳瓶 ended up wearing
    # one like a bib.
    r = body.r_of(body.t_of(y))
    return abs(x - body.cx) < max(r * 0.45, r + pad)


def head_R(body, S, spec):
    """花与瓶称 — how big one blossom of `spec` may be on this vessel.

    gongbi sizes a head against the *sheet*, which is right in gongbi, where
    the flower is the subject and fills the paper. Here the flower is in a
    pot, and 花与瓶称 is a relation between two objects rather than between
    one object and the paper.

    It is a relation between their *widths*, and that is the correction. The
    first version of this cap measured the head against `body.H` at 0.27,
    which sounds like the same idea and is not: a vase is three or four times
    as tall as it is wide, so a cap stated in its height let a 牡丹 come out
    nearly twice as wide as the belly it was standing over. Two peonies like
    that and the vessel is a stem with two cabbages on it. Against `rmax` the
    widest head on the sheet is just under the width of the pot's belly,
    which is where the manuals put it.

    The cap only bites on the big species — 牡丹, 荷花, 菊花, 山茶. A 梅花 at
    `size` 0.52 is nowhere near it and is left entirely alone, which is
    correct: 梅 is a small flower and a vase of plum is a lot of small
    flowers.
    """
    return min(S * 0.145 * spec["size"], body.rmax * 0.92)


def leaf_L(body, S, spec):
    """The same relation for a leaf: `(Lmin, Lmax)` in paper pixels.

    A 荷叶 is specified at up to 1.3 of gongbi's own leaf unit, which on a
    560 px sheet is a blade longer than a 花觚 is wide. Leaves are worse than
    blossoms when they are oversized, because a leaf is a flat slab of one
    colour: three of them across a vase is a hedge with a pot behind it, and
    that is most of what read as "cluttered". Capped at 1.9 `rmax` the
    longest blade is a little under the belly's full width.
    """
    lo, hi = spec["leaf"]["L"]
    cap = body.rmax * 1.55
    return min(S * lo * 0.19, cap * 0.70), min(S * hi * 0.19, cap)


# How a species' sheet budget is shared between its stems, by 三才 rank.
# The 主枝 carries the arrangement; the others answer it. Flat weights give
# three equal sprays in one pot, which is three bouquets, not one.
RANK_W = (1.00, 0.62, 0.42, 0.30)

# Leaves get a little more than gongbi's own count, because that count is
# per *plant* and a vase is several cut stems. Only a little, though. The
# first attempt at this put the multiplier at 1.65 and the 菊花 sheets came
# back with one flower behind nine big flat green slabs: a leaf carries far
# more visual weight than its count suggests, since it is a solid shape in
# a single saturated colour and a blossom is a graded wash. 叶不夺花.
LEAF_SHARE = 1.05


def _share(total, weights):
    """Split an integer budget by weight, largest remainder first.

    Rounding each share independently loses the last flower about half the
    time, and the flower it loses is the one on the 使枝 — the short stem
    whose whole job is to tie the mass back to the pot.
    """
    s = sum(weights) or 1.0
    raw = [total * w / s for w in weights]
    out = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - out[i],
                   reverse=True)
    for i in order[:max(0, total - sum(out))]:
        out[i] += 1
    return out


def companion(rng, key):
    """宾 — which second species may share the vase, or `None`.

    杂花 is the one liberty this painter has that the rest of the family
    cannot take: everywhere else the flowers are attached to their own
    tree, and a plum and a camellia on one branch would be a lie. In a vase
    they are cut stems in water and the 岁朝图 mixes four.

    But the liberty was being taken as "any other species", and any other
    species is how an orange 梅花 ended up beside a hot-pink 山茶 — two
    flowers that do not open in the same month, in two reds that fight. A
    cut arrangement is still a *season*; the season is what the picture is
    about and what its 题款 says.

    Restricted to the season, gongbi's six species offer exactly the two
    pairs the painters actually painted: 春 牡丹 + 玉兰 is 玉堂富贵, 冬 梅
    + 山茶 is an 岁寒 pairing. 荷花 and 菊花 are alone in their seasons and
    so always stand alone — which is also right. Nobody puts anything next
    to a lotus.
    """
    season = SPECIES[key]["season"]
    pool = [k for k in sorted(SPECIES)
            if k != key and SPECIES[k]["season"] == season]
    return pool[rng.randrange(len(pool))] if pool else None


def colourways(rng, keys):
    """One 设色 per species per sheet, with the 宾 tuned to the 主.

    Two faults, one fix. The palette used to be drawn inside the per-stem
    loop, so a species that happened to get two stems got two palettes and
    the sheet came back with a violet 菊 beside a yellow one. They are the
    same plant; a plant has one colour.

    And the companion's colourway is not free either. 牡丹 + 玉兰 is a legal
    pair on the calendar and can still be a crimson against a cold violet.
    Of the companion's three colourways the one nearest the host's in hue is
    taken — which is how the 玉堂富贵 pictures are in fact coloured, the
    second flower a paler statement of the first rather than an argument
    with it.

    The 罩染 glaze is drawn once from a fixed sub-stream and applied to both,
    so the two species share an air as well as a hue. That is the entire
    purpose of a 罩染 and it was being defeated by calling `glaze_palette`
    twice.
    """
    gseed = rng.random()
    host = keys[0]
    cw = SPECIES[host]["colors"]
    pick = cw[rng.randrange(len(cw))]
    out = {SPECIES[host]["cn"]: glaze_palette(random.Random(gseed), pick)}
    h0 = to_hsv(pick[1])[0]
    for k in keys[1:]:
        best = min(SPECIES[k]["colors"],
                   key=lambda c: abs(((to_hsv(c[1])[0] - h0 + 180.0) % 360.0)
                                     - 180.0))
        out[SPECIES[k]["cn"]] = glaze_palette(random.Random(gseed), best)
    return out


def spray(cv, rng, body, mcy, S, occ, specs, pals, sc, mist):
    """The flowers: 枝, 叶, 花头, 花苞 — all of them gongbi's.

    `specs` is one or two species, the host first; `pals` maps a species'
    `cn` to the one palette it wears on this sheet. Two species is 杂花 —
    see `companion` for why the second is not drawn freely.

    Three quantities decide whether this comes out as an arrangement or as
    a thicket, and all three are stated against the *vessel* or against the
    *sheet*, never against a stem: how big a head may be (`head_R`), how
    long a leaf may be (`leaf_L`), and how many of each there are in total
    (the budget below). Per-stem counts are what made the frame busy — the
    counts look modest written down and then get drawn three times.
    """
    W, H = cv.w, cv.h
    reach = rng.uniform(*sc["reach"])
    n = rng.randint(*sc["nbranch"])

    # 留白 — the arrangement must fit the paper. `reach` is a multiple of
    # the vessel's height, which is the right unit for the *shape* of a
    # 三才 spray and says nothing at all about the sheet; off a tall vase
    # it put the 主枝's blossom clean through the top edge. So it is cut
    # back until the longest stem, plus the head it will carry, clears a
    # margin. `leaf_sites`' `sheet` argument cannot do this job: it is a
    # preference, not a clamp.
    room = mcy - head_R(body, S, specs[0]) * 1.30 - H * 0.05
    reach = max(0.55, min(reach, room / max(1.0, body.H)))
    paths = stems(rng, body, mcy, S, reach, n)
    scale = S / 1000.0

    # 宾主 — which stem belongs to which species.
    #
    # The old rule was `specs[i % len(specs)]`: on three stems that gives
    # the host two and the companion one, but on the four of a 满堂 it
    # gives them two each, and two-and-two is not 杂花 — it is two bouquets
    # sharing a pot. The companion gets exactly one stem and it is the 客
    # 枝, the middle rank, never the 主枝 that sets the height of the
    # picture. 宾不夺主.
    owner = [0] * len(paths)
    if len(specs) > 1 and len(paths) > 1:
        owner[1] = 1

    # 疏密 — the budget, which belongs to the sheet and not to the stem.
    #
    # gongbi's `blossoms`, `buds` and `leaf.n` are written for one plant
    # filling one sheet. Drawn once per stem, a 牡丹 at 5–8 leaves came out
    # of a three-stem 满堂 with twenty-four of them, and twenty-four peony
    # leaves is a shrub with a vase behind it. Each species now rolls its
    # count *once*, scales it by the scene's 疏密, and shares the result
    # out among its own stems by 三才 rank.
    quota = [dict(bl=0, bu=0, lf=0) for _ in paths]
    for si, spec in enumerate(specs):
        mine = [i for i in range(len(paths)) if owner[i] == si]
        if not mine:
            continue
        sub = 1.0 if si == 0 else 0.55        # 宾不夺主, in count as in size
        w = [RANK_W[min(i, len(RANK_W) - 1)] for i in mine]
        # A floor of two heads once the host has three stems. 疏 is a
        # virtue and one flower on three stems is not 疏, it is unfinished:
        # the other two stems end in buds and the sheet reads as an
        # arrangement that something happened to. 孤瓶, which really does
        # want a single head, only ever draws two stems and so never
        # reaches the floor.
        floor = 2 if (si == 0 and len(mine) >= 3) else 1
        want = dict(
            bl=max(floor,
                   round(rng.randint(*spec["blossoms"]) * sc["dens"] * sub)),
            bu=round(rng.randint(*spec["buds"]) * sc["dens"] * sub),
            lf=round(rng.randint(*spec["leaf"]["n"]) * sc["dens"] * sub
                     * LEAF_SHARE),
        )
        for field, total in want.items():
            for i, got in zip(mine, _share(int(total), w)):
                quota[i][field] = got

    heads = []
    for i, path in enumerate(paths):
        spec = specs[owner[i]]
        bw0, bw1 = spec["branch"]["w"]
        # A cut stem is thinner than the same species' standing branch: it
        # is a 折枝, chosen for the vase, and 13 px of plum trunk coming out
        # of a 胆瓶 neck is a log in a bottle.
        bw0 *= scale * 0.62
        bw1 *= scale * 0.70
        hosts = [(path, bw0, bw1)]
        twigs = []
        for _ in range(rng.randint(1, 3)):
            j = rng.randrange(int(len(path) * 0.30), int(len(path) * 0.92))
            bx, by = path[j]
            a = tangent(path, j) + rng.choice((1, -1)) * rng.uniform(0.35, 0.95)
            ln = S * rng.uniform(0.09, 0.22) * reach
            q3 = (bx + math.cos(a) * ln, by + math.sin(a) * ln)
            q1 = (bx + math.cos(a) * ln * 0.35, by + math.sin(a) * ln * 0.35)
            q2 = (q3[0] - math.cos(a + 0.9) * ln * 0.25,
                  q3[1] - math.sin(a + 0.9) * ln * 0.25)
            tw, _ = knuckle(rng, bezier((bx, by), q1, q2, q3, 12),
                            k=rng.randint(1, 2), amp=0.22, start=0.25)
            twigs.append(tw)
            hosts.append((tw, bw0 * 0.70, bw1 * 0.72))
        for tw in twigs:
            draw_branch(cv, rng, tw, bw0 * 0.70, bw1 * 0.72, spec)
            occ.add_path(tw, bw0 * 0.70, bw1 * 0.72)
        draw_branch(cv, rng, path, bw0, bw1, spec)
        occ.add_path(path, bw0, bw1)

        lf = spec["leaf"]
        lmin, lmax = leaf_L(body, S, spec)
        mar = S * 0.018
        for bx, by, a, L, bur in leaf_sites(
                rng, hosts, quota[i]["lf"], lmin, lmax,
                sep={"trifoliate": 0.98, "round": 1.06}.get(lf.get("form"),
                                                            0.80),
                sheet=(W, H)):
            mx = bx + math.cos(a) * L * 0.5
            my = by + math.sin(a) * L * 0.5
            # The whole blade is tested, not just its middle. A leaf is a
            # long thing: pivoted so that its midpoint clears the pot by a
            # hair, it still lies right across the belly, and the 贯耳瓶
            # came out wearing one like a bib.
            half = L * lf["wr"] * 0.38       # the blade's own half-width
            if any(on_pot(body, mcy, bx + math.cos(a) * L * f,
                          by + math.sin(a) * L * f, half)
                   for f in (0.30, 0.55, 0.80, 1.0)):
                continue
            # And tested against the paper edge for the same reason — a
            # blade pointing out of the sheet is a cut blade. `sheet=` only
            # biases where `leaf_sites` looks; it does not refuse.
            tx, ty = bx + math.cos(a) * L, by + math.sin(a) * L
            if not (mar < tx < W - mar and mar < ty < H - mar):
                continue
            paint_leaf(cv, rng, spec, bx, by, a, L,
                       alpha=rng.uniform(0.9, 1.0), depth=bur)
            occ.add(mx, my, L * 0.55)
        heads.append((spec, hosts, i))

    # 花头 — after every branch and leaf on the sheet, so a blossom is never
    # cut by a stem drawn later. 聚散 is judged across the *whole* spray:
    # two heads from different stems that touch merge exactly as readily as
    # two from one.
    flowers = []
    buds = []
    for spec, hosts, i in heads:
        colors = pals[spec["cn"]]
        base_R = head_R(body, S, spec) * (1.0 if owner[i] == 0 else 0.78)
        sites = []
        for path, _a, _b in hosts:
            sites.append(path[-1])
            if len(path) > 8 and rng.random() < 0.65:
                sites.append(path[int(len(path) * rng.uniform(0.55, 0.88))])
        rng.shuffle(sites)
        nb = quota[i]["bl"]
        got = 0
        for sx, sy in sites:
            if got >= nb:
                break
            R = base_R * (1.0 if got == 0 else rng.uniform(0.58, 0.88))
            # The jitter is stated in the head's own radius, not in the
            # sheet's width. At 0.035 W it was up to 20 px on a 560 px
            # sheet, which is most of a small flower's diameter: the head
            # slid off the node it was placed on and sat in clear air with
            # its twig ending somewhere behind it. A blossom may lean off
            # its tip; it may not leave it.
            sx += rng.uniform(-0.34, 0.34) * R
            sy += rng.uniform(-0.34, 0.34) * R
            pad = R * 1.45
            sx = min(W - pad, max(pad, sx))
            sy = min(H - pad, max(pad, sy))
            if on_pot(body, mcy, sx, sy, -R * 0.35):
                continue
            sep = 0.80 if spec["size"] < 0.6 else 1.06
            if any(math.hypot(sx - fx, sy - fy) < (R + fr) * sep
                   for fx, fy, fr, _c, _o in flowers):
                continue
            flowers.append((sx, sy, R, colors,
                            1.0 if got == 0 else rng.uniform(0.82, 1.0)))
            got += 1
        for _ in range(quota[i]["bu"]):
            path, _a, _b = hosts[rng.randrange(len(hosts))]
            j = rng.randrange(max(1, int(len(path) * 0.45)), len(path))
            bx, by = path[j]
            a = tangent(path, j) + rng.uniform(-0.8, 0.8)
            buds.append((spec, bx, by, base_R * rng.uniform(0.28, 0.46), a,
                         colors))
        # 无头之枝 — a cut stem that stops in mid-air is the one fault the
        # eye catches every single time, and the counts above can leave one
        # whenever the blossom budget runs out before the tips do. So every
        # tip that ended bare gets a bud. It is cheaper to add a bud than to
        # justify a twig pointing at nothing.
        for path, _a, _b in hosts:
            tx, ty = path[-1]
            if on_pot(body, mcy, tx, ty):
                continue
            if any(math.hypot(tx - fx, ty - fy) < fr * 1.15
                   for fx, fy, fr, _c, _o in flowers):
                continue
            if any(math.hypot(tx - bx, ty - by) < br * 2.2
                   for _s, bx, by, br, _ang, _c in buds):
                continue
            buds.append((spec, tx, ty, base_R * rng.uniform(0.26, 0.40),
                         tangent(path, len(path) - 1), colors))

    for spec, bx, by, R, a, colors in buds:
        paint_bud(cv, rng, spec, bx, by, R, colors, a,
                  alpha=rng.uniform(0.92, 1.0))
        occ.add(bx + math.cos(a) * R * 0.5, by + math.sin(a) * R * 0.5, R * 1.2)
    flowers.sort(key=lambda f: f[2])
    out = []
    for sx, sy, R, colors, op in flowers:
        dh = rng.uniform(-7.0, 7.0)
        sk = rng.uniform(0.94, 1.08)
        from gongbi import hue_rotate
        cols = (hue_rotate(colors[0], dh, sk), hue_rotate(colors[1], dh, sk),
                hue_rotate(colors[2], dh, sk) if colors[2] else None)
        paint_blossom(cv, rng, _owner(heads, sx, sy, specs), sx, sy, R, cols,
                      openness=op)
        occ.add(sx, sy, R * 1.12)
        out.append((sx, sy, R))
    return out


def _owner(heads, sx, sy, specs):
    """Which species a settled head belongs to.

    The heads are pooled across stems so that 聚散 can be judged over the
    whole spray, and pooling loses the species. Nearest host wins, which is
    right by construction: a head was placed on one of its own stem's nodes.
    """
    best, bd = specs[0], None
    for spec, hosts, _i in heads:
        for path, _a, _b in hosts:
            for px, py in path[::4]:
                d = (px - sx) ** 2 + (py - sy) ** 2
                if bd is None or d < bd:
                    best, bd = spec, d
    return best


# --------------------------------------------------------------------------
# 章法 — the sheet
# --------------------------------------------------------------------------

def compose(cv, rng, scene_key, vase_key, ware_key, flower_key, stat=None):
    W, H = cv.w, cv.h
    S = min(W, H)
    light_a = set_light(rng)
    ground = paint_ground(cv, rng)
    sc = SCENES[scene_key]
    vz = VASES[vase_key]
    ware = WARES[ware_key]
    occ = Occupancy(W, H)
    mist = Noise(rng, 4, 5)

    # --- 立器 — where the vessel stands ------------------------------
    #
    # The vase is the only fixed point in the picture: the flowers are
    # arranged *on* it, the 款 is balanced *against* it, and the stand and
    # the fruit are placed *under* it. So it is sited first, and everything
    # else is read off its mouth. Its height is a fraction of the sheet's
    # short side and its foot sits well up from the bottom border, because a
    # vessel standing on the edge of the paper has nothing to stand on.
    vh = H * rng.uniform(*sc["vh"])
    cx = W * rng.uniform(*sc["vx"])
    has_stand = rng.random() < sc["stand"]
    ybot = H * rng.uniform(0.80, 0.87) - (vh * 0.06 if has_stand else 0.0)
    body = Body(cv, rng, vz["prof"], cx, ybot - vh, ybot,
                vh * vz["hw"], vz["squash"], light_a)

    if has_stand:
        stand(cv, rng, body, S)
    else:
        disc(cv, cx, ybot + body.rbot * body.squash * 0.6, body.rbot * 1.3,
             mix(P("danmo"), P("zheshi"), 0.4), 0.15,
             squash=body.squash * 0.7, soft=0.9)

    # --- 器 — the pot itself, in the order a kiln makes one ----------
    body.fill(ware)
    if ware["decor"] == "qinghua":
        qinghua(cv, rng, body, ware)
    elif ware["decor"] == "incise":
        incise(cv, rng, body, ware)
    if ware["crackle"] > 0.0:
        crackle(cv, rng, body, ware, ware["crackle"])
    body.gloss(ware)
    body.outline(ware)
    # 贯耳 last of the vessel: they stand outside the silhouette, so the
    # body's own contour must not be drawn across them, and the neck band
    # must not be drawn over them either.
    if vz["ears"] > 0.0:
        ears(cv, rng, body, ware)
    foot(cv, rng, body, ware)
    mcy = mouth(cv, rng, body, ware)
    occ.add(cx, ybot - vh * 0.5, max(body.rmax, vh * 0.42) * 0.96)

    # --- 花 — the spray --------------------------------------------
    keys = [flower_key]
    if rng.random() < sc["mixed"]:
        second = companion(rng, flower_key)
        if second:
            keys.append(second)
    specs = [SPECIES[k] for k in keys]
    pals = colourways(rng, keys)
    flowers = spray(cv, rng, body, mcy, S, occ, specs, pals, sc, mist)

    # --- 口 — the near lip, re-laid over the stems -------------------
    #
    # This is the whole reason the mouth is a separate function. Under an
    # over-composite the stems are painted after the pot, so they run down
    # the front of the neck and out of the bottom of the vase; drawing the
    # near half of the lip again, afterwards, is what puts them *into* it.
    wall = max(1.2, body.rtop * 0.085)
    rm = body.rtop - wall * 0.5
    rw = max(0.9, wall * 0.62)
    steps = max(40, int(math.pi * rm / max(0.6, rw * 0.45)))
    rim_c, hi_c = rgb(ware["rim"]), rgb(ware["hi"])
    edge = mix(rgb(ware["deep"]), P("mo"), 0.35)
    aL = math.acos(max(-1.0, min(1.0, math.sin(body.phiL))))
    for i in range(steps + 1):
        a = math.pi * i / steps
        x = body.cx + math.cos(a) * rm
        y = mcy + math.sin(a) * rm * body.squash
        # shaded round the lip like everything else, or the near rim comes
        # out a flat white saucer resting on the neck
        lit = clamp01(0.5 + 0.5 * math.cos(a - aL))
        disc(cv, x, y, rw,
             mix(mix(rim_c, edge, 0.45), mix(rim_c, hi_c, 0.30), lit), 0.95)
        disc(cv, x, y + rw * 0.66, max(0.7, rw * 0.52), edge,
             0.22 + 0.14 * (1.0 - lit))
        # 灯草口 — on a 霁红 the copper glaze draws back off the lip in the
        # kiln and leaves a thin white edge, "like a rush wick". It is the
        # signature of the ware, and a sacrificial red without it looks
        # like a painted gourd.
        if ware["rimline"] > 0.0:
            disc(cv, x, y - rw * 0.74, max(0.7, rw * 0.48),
                 mix(rgb(ware["rim"]), P("baifen"), 0.55),
                 (0.55 + 0.35 * lit) * ware["rimline"])

    # --- 果 — the companion, if the scene wants one -------------------
    if rng.random() < sc["fruit"] and flowers:
        side = -1.0 if cx > W * 0.5 else 1.0
        fr = S * rng.uniform(0.038, 0.055)
        fx = cx + side * body.rmax * rng.uniform(1.25, 1.85)
        fy = ybot - fr * 0.25
        if fr * 2.2 < fx < W - fr * 2.2:
            for k in range(rng.choice((2, 2, 3))):
                dx = (k - 0.5) * fr * rng.uniform(1.3, 1.7) * side
                dy = -abs(k - 1) * fr * rng.uniform(0.0, 0.35)
                rr = fr * rng.uniform(0.82, 1.10)
                disc(cv, fx + dx, fy + rr * 0.9, rr * 1.25,
                     mix(P("danmo"), P("zheshi"), 0.4), 0.14,
                     squash=0.26, soft=0.9)
                fruit(cv, rng, fx + dx, fy + dy, rr, S)
                occ.add(fx + dx, fy + dy, rr * 1.2)

    # --- 草虫 — the companion insect ---------------------------------
    if flowers and rng.random() < 0.34:
        bs = S * rng.uniform(0.050, 0.078)
        spot = occ.clear_spot(rng, flowers, bs * 1.4, bs * 1.05)
        if spot:
            paint_butterfly(cv, rng, spot[0], spot[1], bs,
                            rng.uniform(0, TAU))
            occ.add(spot[0], spot[1], bs * 1.05)
    elif flowers and rng.random() < 0.30:
        host = flowers[rng.randrange(len(flowers))]
        bs = S * rng.uniform(0.026, 0.038)
        reach_ = bs * 0.78
        spot = occ.clear_spot(rng, [host], (0.14 * host[2] + 1.25 * reach_)
                              / 0.6, reach_, tries=60)
        if spot:
            paint_bee(cv, rng, spot[0], spot[1], bs,
                      math.atan2(host[1] - spot[1], host[0] - spot[0])
                      + rng.gauss(0.0, 0.42))
            occ.add(spot[0], spot[1], reach_ * 1.5)

    if stat is not None:
        stat["flowers"] = len(flowers)
        stat["vase"] = vase_key

    title = inscribe(cv, rng, occ, S, sc, vz, specs)
    return title, ground


def inscribe(cv, rng, occ, S, sc, vz, specs):
    """题款 — title, signature and seals, written into the 留白."""
    W, H = cv.w, cv.h
    title = (TITLE_A[rng.randrange(len(TITLE_A))]
             + TITLE_B[rng.randrange(len(TITLE_B))])
    sign = (TIANGAN[rng.randrange(10)] + DIZHI[rng.randrange(12)]
            + specs[0]["season"] + rng.choice(("日", "月"))
            + ACTION[rng.randrange(len(ACTION))])
    if FONT is None:
        return title
    gs = S * rng.uniform(0.034, 0.044)
    ss = gs * 0.62
    lead, lead_s = 1.22, 1.18
    gap = gs * 0.42
    sign_dy = gs * rng.uniform(1.2, 2.2)
    seal_s = gs * rng.uniform(0.82, 1.0)
    bw = gs + gap + ss
    bh = max(len(title) * gs * lead,
             sign_dy + len(sign) * ss * lead_s + gs * 0.25 + seal_s)
    margin = S * 0.035
    block = None
    if bw < W - 2 * margin and bh < H - 2 * margin:
        bx, by = occ.place(bw, bh, margin)
        paint_column(cv, rng, FONT, title, bx + gap + ss, by, gs, lead=lead)
        end_s = paint_column(cv, rng, FONT, sign, bx, by + sign_dy, ss,
                             lead=lead_s, alpha=(0.66, 0.86))
        paint_seal(cv, rng, FONT, VASE_SEALS[rng.randrange(len(VASE_SEALS))],
                   bx + (ss - seal_s) * 0.5, end_s + gs * 0.2, seal_s)
        occ.add(bx + bw * 0.5, by + bh * 0.5, max(bw, bh) * 0.55)
        block = (bx, by, bw, bh)
    if rng.random() < 0.5:
        corner_seal(cv, rng, FONT, VASE_SEALS, occ, S, block,
                    lo=0.038, hi=0.052)
    return title


# --------------------------------------------------------------------------
# 字 — the baked inscription font
# --------------------------------------------------------------------------

def vocabulary():
    """Every character this painter can write — tools/extract_glyphs.py bakes
    exactly these into vase_font.py."""
    words = (TITLE_A + TITLE_B + ACTION + VASE_SEALS
             + [s["cn"] for s in SPECIES.values()]
             + [s["season"] for s in SPECIES.values()]
             + [v["cn"] for v in VASES.values()]
             + [w["cn"] for w in WARES.values()]
             + [s["cn"] for s in SCENES.values()]
             + [TIANGAN, DIZHI, "日月年写于图"])
    return sorted(set("".join(words)))


def _load_font():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import vase_font as vf
    except ImportError:
        return None
    try:
        return BrushFont(vf.DATA, vf.EM)
    except Exception as exc:
        print("  (font blob unreadable: %s)" % exc, file=sys.stderr)
        return None


FONT = _load_font()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def paint(path, size=(900, 1300), seed=None, scene=None, vase=None,
          ware=None, flower=None, quiet=False, stat=None):
    if seed is None:
        seed = random.randrange(1 << 30)
    rng = random.Random(seed)
    sk = scene or sorted(SCENES)[rng.randrange(len(SCENES))]
    vk = vase or sorted(VASES)[rng.randrange(len(VASES))]
    wk = ware or sorted(WARES)[rng.randrange(len(WARES))]
    fk = flower or sorted(SPECIES)[rng.randrange(len(SPECIES))]
    for key, tbl, what in ((sk, SCENES, "scene"), (vk, VASES, "vase"),
                           (wk, WARES, "ware"), (fk, SPECIES, "flower")):
        if key not in tbl:
            raise SystemExit("unknown %s %r; try --list" % (what, key))

    cv = Canvas(size[0], size[1])
    t0 = time.time()
    title, ground = compose(cv, rng, sk, vk, wk, fk, stat=stat)
    age(cv, rng)
    cv.write_png(path)
    if not quiet:
        print("  %-24s %s  %s·%s·%s·%s(%s)  seed=%d  %dx%d  %.1fs"
              % (os.path.basename(path), title, SCENES[sk]["cn"],
                 VASES[vk]["cn"], WARES[wk]["cn"], SPECIES[fk]["cn"],
                 ground, seed, size[0], size[1], time.time() - t0))
    return path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate 工笔 paintings of a porcelain vase of flowers.")
    ap.add_argument("-o", "--out", default="vase.png", help="output PNG")
    ap.add_argument("-n", "--count", type=int, default=1, help="how many")
    ap.add_argument("-s", "--seed", type=int, help="random seed")
    ap.add_argument("--scene", help="|".join(sorted(SCENES)))
    ap.add_argument("--vase", help="|".join(sorted(VASES)))
    ap.add_argument("--ware", help="|".join(sorted(WARES)))
    ap.add_argument("--flower", help="|".join(sorted(SPECIES)))
    ap.add_argument("--size", default="900x1300", help="WIDTHxHEIGHT")
    ap.add_argument("--list", action="store_true", help="list the repertoire")
    a = ap.parse_args(argv)

    if a.list:
        print("章法 (composition):")
        for k in sorted(SCENES):
            print("  %-8s %s" % (k, SCENES[k]["en"]))
        print("器型 (vessel):")
        for k in sorted(VASES):
            print("  %-8s %s" % (k, VASES[k]["en"]))
        print("窑口 (ware):")
        for k in sorted(WARES):
            print("  %-8s %s" % (k, WARES[k]["en"]))
        print("花 (flower):")
        for k in sorted(SPECIES):
            print("  %-16s %s" % (k, SPECIES[k]["cn"]))
        return

    try:
        w, h = (int(v) for v in a.size.lower().split("x"))
    except ValueError:
        raise SystemExit("--size wants WIDTHxHEIGHT, e.g. 900x1300")

    print("瓶花 — painting %d ..." % a.count)
    for i in range(a.count):
        out = a.out if a.count == 1 else "%s-%d%s" % (
            os.path.splitext(a.out)[0], i + 1, os.path.splitext(a.out)[1])
        paint(out, (w, h), None if a.seed is None else a.seed + i,
              a.scene, a.vase, a.ware, a.flower)


if __name__ == "__main__":
    sys.exit(main())
