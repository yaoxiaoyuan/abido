#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""随笔 — draw one random painting from whichever of the four painters comes up.

The four modules already do all the work and they already agree on an
interface:

    paint(path, size=(w, h), seed=None, quiet=False) -> (path, title, seed)

and each one picks its own subject, weather, palette and ground when it is not
told.  So the only thing left to choose is *which brush to pick up*, and this
script does nothing else.  No new drawing code lives here.

    python3 random_paint.py                     # one painting, random painter
    python3 random_paint.py -n 4                 # four of them
    python3 random_paint.py -p bamboo -s 909     # a named painter and seed
    python3 random_paint.py --size 600x800       # override the sheet
    python3 random_paint.py --list
"""

import argparse
import os
import random

import gongbi
import shanshui
import bamboo
import animal
import pavilion

# name -> (module, 画科, the module's own default sheet)
PAINTERS = {
    "gongbi":   (gongbi,   "工笔花卉", (1000, 1360)),
    "shanshui": (shanshui, "山水",     (1000, 1400)),
    "bamboo":   (bamboo,   "墨竹",     (900, 1300)),
    "animal":   (animal,   "游鱼",     (900, 1200)),
    "pavilion": (pavilion, "亭景",     (900, 1200)),
}


def paint_random(out="out.png", seed=None, painter=None, size=None,
                 quiet=False):
    """Draw one painting.  Returns (painter_name, path, title, seed)."""
    if seed is None:
        seed = random.randrange(1 << 30)
    # Choose the painter from the seed too, so one number reproduces the whole
    # painting and not just its contents.
    name = painter or sorted(PAINTERS)[random.Random(seed).randrange(
        len(PAINTERS))]
    if name not in PAINTERS:
        raise SystemExit("unknown painter %r; try --list" % name)
    mod, cn, default_size = PAINTERS[name]

    if not quiet:
        print("%s  %s" % (name, cn))
    path, title, seed = mod.paint(out, size=size or default_size, seed=seed,
                                 quiet=quiet)
    return name, path, title, seed


def main(argv=None):
    ap = argparse.ArgumentParser(description="draw a random Chinese painting")
    ap.add_argument("-o", "--out", default="random.png")
    ap.add_argument("-n", "--count", type=int, default=1)
    ap.add_argument("-s", "--seed", type=int, default=None)
    ap.add_argument("-p", "--painter", default=None,
                    help="gongbi | shanshui | bamboo | animal | pavilion")
    ap.add_argument("--size", default=None, help="WxH, e.g. 900x1200")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)

    if a.list:
        for k in sorted(PAINTERS):
            mod, cn, sz = PAINTERS[k]
            print("  %-9s %-9s %dx%d  (%s.py)" % (k, cn, sz[0], sz[1], k))
        return 0

    size = None
    if a.size:
        w, _, h = a.size.lower().partition("x")
        size = (int(w), int(h))

    stem, ext = os.path.splitext(a.out)
    ext = ext or ".png"
    for i in range(max(1, a.count)):
        out = a.out if a.count == 1 else "%s-%d%s" % (stem, i + 1, ext)
        seed = a.seed if a.count == 1 else (
            None if a.seed is None else a.seed + i)
        paint_random(out, seed=seed, painter=a.painter, size=size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
