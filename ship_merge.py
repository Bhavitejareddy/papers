"""
Ship detection merge: global strip pass + hard-cut tiles + seam-aware merge.
Per camera strip: 7680x1080, 4 tiles of 1920x1080, output 2000x270 per camera.
Boxes are [x1, y1, x2, y2] in pixels. Run: python ship_merge.py
"""
import numpy as np

# ---------------- config ----------------
CAM_W, CAM_H = 7680, 1080
TILE_W, N_TILES = 1920, 4
OUT_W, OUT_CAM_H = 2000, 270
SX, SY = OUT_W / CAM_W, OUT_CAM_H / CAM_H     # 0.2604, 0.25

SEAM_TOL = 20          # px: box edge this close to a tile border counts as "touching the seam"
CONTAIN_MIN = 0.7      # tile box counts as inside a global box if >=70% of its area is inside
Y_OVERLAP_MIN = 0.5    # vertical overlap / smaller height
WATERLINE_TOL = 0.15   # |bottom_a - bottom_b| <= 15% of the taller box height
MAX_MERGED_W = 6000    # px: refuse merges wider than this (full-res)


class Box:
    def __init__(self, x1, y1, x2, y2, conf, cls, tile=-1):
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.conf, self.cls, self.tile = conf, cls, tile   # tile=-1 means global-pass box
        self.touch_left = False
        self.touch_right = False

    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1
    @property
    def area(self): return max(0.0, self.w) * max(0.0, self.h)

    def xyxy(self): return [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)]
    def __repr__(self):
        src = "G" if self.tile < 0 else f"T{self.tile}"
        return f"{src}:{self.cls}{self.xyxy()}@{self.conf:.2f}"


# ---------------- step 3: lift to global coords + seam flags ----------------
def lift_tile_boxes(dets_per_tile):
    """dets_per_tile[i] = list of (x1,y1,x2,y2,conf,cls) in tile-local coords."""
    out = []
    for t, dets in enumerate(dets_per_tile):
        for x1, y1, x2, y2, c, k in dets:
            b = Box(x1 + t * TILE_W, y1, x2 + t * TILE_W, y2, c, k, tile=t)
            # only INTERNAL borders are seams. strip edges (x=0, x=7680) are not.
            b.touch_left = t > 0 and b.x1 <= t * TILE_W + SEAM_TOL
            b.touch_right = t < N_TILES - 1 and b.x2 >= (t + 1) * TILE_W - SEAM_TOL
            out.append(b)
    return out


def scale_global_boxes(global_dets, in_w, in_h):
    """Global pass ran on a resized strip (e.g. 3840x544). Map back to 7680x1080."""
    fx, fy = CAM_W / in_w, CAM_H / in_h
    return [Box(x1 * fx, y1 * fy, x2 * fx, y2 * fy, c, k) for x1, y1, x2, y2, c, k in global_dets]


# ---------------- step 4: reconcile with global pass ----------------
def containment(inner, outer):
    """fraction of inner's area that lies inside outer."""
    iw = min(inner.x2, outer.x2) - max(inner.x1, outer.x1)
    ih = min(inner.y2, outer.y2) - max(inner.y1, outer.y1)
    if iw <= 0 or ih <= 0 or inner.area == 0:
        return 0.0
    return iw * ih / inner.area


def reconcile(global_boxes, tile_boxes, log):
    consumed = set()        # indices of tile boxes absorbed by a global box
    keep_global = []
    for g in global_boxes:
        inside = [i for i, t in enumerate(tile_boxes)
                  if i not in consumed and t.cls == g.cls and containment(t, g) >= CONTAIN_MIN]
        any_trunc = any(tile_boxes[i].touch_left or tile_boxes[i].touch_right for i in inside)
        if len(inside) >= 2 or any_trunc:
            keep_global.append(g)                     # global box wins, fragments dropped
            consumed.update(inside)
            log.append(f"  reconcile: {g} replaces {[tile_boxes[i] for i in inside]}")
        elif len(inside) == 1:
            log.append(f"  reconcile: {g} duplicates whole tile box {tile_boxes[inside[0]]}, global dropped")
        else:
            keep_global.append(g)                     # tile pass missed it
            log.append(f"  reconcile: {g} has no tile support, kept")
    leftovers = [t for i, t in enumerate(tile_boxes) if i not in consumed]
    return keep_global, leftovers


# ---------------- step 5: seam linking ----------------
def y_overlap(a, b):
    inter = min(a.y2, b.y2) - max(a.y1, b.y1)
    return inter / max(1e-6, min(a.h, b.h))


def can_link(a, b, log):
    """a on tile t, b on tile t+1."""
    if b.tile != a.tile + 1 or a.cls != b.cls:
        return False
    if not (a.touch_right and b.touch_left):
        return False
    yo = y_overlap(a, b)
    if yo < Y_OVERLAP_MIN:
        log.append(f"  no link {a} | {b}: y-overlap {yo:.2f} < {Y_OVERLAP_MIN}")
        return False
    wl = abs(a.y2 - b.y2)
    if wl > WATERLINE_TOL * max(a.h, b.h):
        log.append(f"  no link {a} | {b}: waterline diff {wl:.0f}px > {WATERLINE_TOL * max(a.h, b.h):.0f}px")
        return False
    return True


def seam_link(boxes, log):
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(n):
            if can_link(boxes[i], boxes[j], log):
                parent[find(i)] = find(j)
                log.append(f"  link {boxes[i]} + {boxes[j]}")

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        x1 = min(m.x1 for m in members); x2 = max(m.x2 for m in members)
        if x2 - x1 > MAX_MERGED_W:
            log.append(f"  group too wide ({x2 - x1:.0f}px), left unmerged")
            merged.extend(members)
            continue
        y1 = min(m.y1 for m in members); y2 = max(m.y2 for m in members)
        m = Box(x1, y1, x2, y2, max(b.conf for b in members), members[0].cls, tile=-2)
        merged.append(m)
        log.append(f"  merged {len(members)} fragments -> {m.xyxy()}")
    return merged


# ---------------- step 7: scale to display ----------------
def to_display(box, camera):
    y_off = 0 if camera == "front" else OUT_CAM_H
    return [round(box.x1 * SX, 1), round(box.y1 * SY + y_off, 1),
            round(box.x2 * SX, 1), round(box.y2 * SY + y_off, 1)]


# ---------------- full per-camera pipeline ----------------
def process_camera(camera, tile_dets, global_dets, g_in_size=(3840, 544), use_global=True):
    log = [f"[{camera}]"]
    tile_boxes = lift_tile_boxes(tile_dets)
    log.append(f"  lifted tile boxes: {tile_boxes}")
    g_boxes = scale_global_boxes(global_dets, *g_in_size) if use_global else []
    if use_global:
        log.append(f"  global boxes (full-res): {g_boxes}")
    kept_global, leftovers = reconcile(g_boxes, tile_boxes, log)
    merged = seam_link(leftovers, log)
    final = kept_global + merged
    result = [{"box_full": b.xyxy(), "box_display": to_display(b, camera), "conf": round(b.conf, 2), "cls": b.cls}
              for b in sorted(final, key=lambda b: b.x1)]
    return result, log, dict(tile_boxes=tile_boxes, g_boxes=g_boxes, kept_global=kept_global,
                             leftovers=leftovers, merged=merged, final=final)


# ---------------- synthetic input ----------------
# Tile-local coords. Ground truth (full-res x): see comments. Detector shrinks boxes ~5px off seams.
FRONT_TILES = [
    # tile 0 (x 0..1920): ship B left part (B = 1400..2200, y 500..680) cut by seam at 1920
    [(1400, 500, 1915, 680, 0.88, "ship")],
    # tile 1 (x 1920..3840): ship B right part, ship A left part (A = 2500..6300, y 300..700)
    [(5, 500, 280, 680, 0.71, "ship"),
     (580, 300, 1915, 700, 0.64, "ship")],
    # tile 2 (x 3840..5760): ship A middle, touches both seams
    [(5, 300, 1915, 700, 0.58, "ship")],
    # tile 3 (x 5760..7680): ship A right part, small ship C (6900..7100, y 560..640) whole
    [(5, 300, 540, 700, 0.69, "ship"),
     (1140, 560, 1340, 640, 0.83, "ship")],
]
# global pass ran on strip resized to 3840x544 (factor 2 in x, ~1.985 in y). Sees only the big ship.
FRONT_GLOBAL = [(1255, 152, 3145, 353, 0.91, "ship")]

REAR_TILES = [
    # tile 0: ship E (1300..1915, y 560..620) ends exactly at seam, small boat
    [(1300, 560, 1915, 620, 0.80, "ship")],
    # tile 1: a DIFFERENT, bigger ship F starting right after the seam (1925..3000, y 400..700)
    [(5, 400, 1080, 700, 0.86, "ship")],
    # tile 2: ship G left part (G = 5400..6400, y 450..650)
    [(1560, 450, 1915, 650, 0.77, "ship")],
    # tile 3: ship G right part
    [(5, 450, 640, 650, 0.74, "ship")],
]
REAR_GLOBAL = []   # global pass found nothing (ships too small at that scale)

if __name__ == "__main__":
    for cam, td, gd in [("front", FRONT_TILES, FRONT_GLOBAL), ("rear", REAR_TILES, REAR_GLOBAL)]:
        res, log, _ = process_camera(cam, td, gd)
        print("\n".join(log))
        print("  FINAL:")
        for r in res:
            print("   ", r)
        print()

    print("=== front camera WITHOUT global pass (seam-link must chain 3 tiles) ===")
    res, log, _ = process_camera("front", FRONT_TILES, FRONT_GLOBAL, use_global=False)
    print("\n".join(log))
    for r in res:
        print("   ", r)
