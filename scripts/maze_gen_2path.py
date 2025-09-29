#!/usr/bin/env python3
"""
maze_gen.py — fixed geometry with explicit anchor + axes; optional two-path variant
via opening exactly one internal wall and checking overlap of the new solution.

Image:
  # classic (unchanged)
  python maze_gen.py image --rows 10 --cols 10 --seed 42 --out mazes/maze.png

  # two-path (open one wall; accept if overlap small)
  python maze_gen.py image --rows 10 --cols 10 --seed 7 \
    --two-path --max-overlap 0.35 --open-tries 200 --avoid-2x2 \
    --out mazes/two_path.png

Dataset:
  # sprinkle two-path mazes with probability p
  python maze_gen.py dataset --count 1000 --rows 10 --cols 10 \
    --two-path-rate 0.5 --max-overlap 0.35 --open-tries 120 --avoid-2x2 \
    --png-dir mazes_out --out info_labels.jsonl
"""

import argparse
import hashlib
import json
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------
# Defaults
# -------------------------
DEF_CANVAS_W = 720
DEF_CANVAS_H = 480

DEF_WALL_PX   = 4     # wall thickness (px)
DEF_KNOB_PX   = 16    # start/goal circle diameter (px)
DEF_GRID_GRAY = 220   # light grid color channel (None to disable)
DEF_GRID_PX   = 1     # light grid thickness (px)

# Axis pads reserve space for tick numbers + "row"/"col" captions
AXIS_PAD_LEFT   = 34  # px
AXIS_PAD_TOP    = 28  # px
AXIS_PAD_RIGHT  = 8   # px
AXIS_PAD_BOTTOM = 8   # px

# Row increases downward, col increases rightward.
DIRS = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
OPP  = {'N':'S', 'S':'N', 'E':'W', 'W':'E'}
DIR_ORDER = ('N','E','S','W')  # deterministic hashing order


# -------------------------
# Maze generation
# -------------------------

def init_grid(rows: int, cols: int):
    return [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False}
             for _ in range(cols)] for __ in range(rows)]

def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

def carve_maze(grid, rows: int, cols: int, rng: random.Random,
               start_r: Optional[int]=None, start_c: Optional[int]=None):
    if start_r is None:
        start_r = rng.randrange(rows)
    if start_c is None:
        start_c = rng.randrange(cols)
    stack = [(start_r, start_c)]
    grid[start_r][start_c]['visited'] = True
    while stack:
        r, c = stack[-1]
        dirs = list(DIRS.items())
        rng.shuffle(dirs)
        progressed = False
        for d, (dr, dc) in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and not grid[nr][nc]['visited']:
                grid[r][c][d] = False
                grid[nr][nc][OPP[d]] = False
                grid[nr][nc]['visited'] = True
                stack.append((nr, nc))
                progressed = True
                break
        if not progressed:
            stack.pop()
    for r in range(rows):
        for c in range(cols):
            grid[r][c]['visited'] = False

def grid_to_graph(grid, rows: int, cols: int) -> Dict[Tuple[int,int], List[Tuple[int,int]]]:
    G = {}
    for r in range(rows):
        for c in range(cols):
            nbrs = []
            for d, (dr, dc) in DIRS.items():
                if not grid[r][c][d]:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc, rows, cols):
                        nbrs.append((nr, nc))
            G[(r, c)] = nbrs
    return G

def shortest_path(graph, start: Tuple[int,int], goal: Tuple[int,int]):
    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in graph[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))

def encode_maze(grid, rows: int, cols: int) -> str:
    bits = []
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            for d in DIR_ORDER:
                bits.append('1' if cell[d] else '0')
    return hashlib.sha256(''.join(bits).encode('ascii')).hexdigest()

def aggregate_segments_labels(true_path, rows: int, cols: int):
    """
    Returns (K, labels) where labels is [[dir_code, len_norm], ...]
    dir_code: Right=0.00, Up=0.25, Left=0.50, Down=0.75
    len_norm: segment length normalized by cols (horizontal) or rows (vertical)
    """
    if len(true_path) <= 1:
        return 0, []
    segs = []
    pr, pc = true_path[0]
    cur_step = (true_path[1][0] - pr, true_path[1][1] - pc)  # (dr, dc)
    acc_dr, acc_dc = cur_step
    pr, pc = true_path[1]
    for (r, c) in true_path[2:]:
        step = (r - pr, c - pc)  # (dr, dc)
        if step == cur_step:
            acc_dr += step[0]
            acc_dc += step[1]
        else:
            segs.append((acc_dr, acc_dc))
            cur_step = step
            acc_dr, acc_dc = step
        pr, pc = r, c
    segs.append((acc_dr, acc_dc))

    def to_dir_and_len(seg):
        dr, dc = seg
        if dc > 0 and dr == 0:      # right
            return 0.00, abs(dc) / cols
        if dc < 0 and dr == 0:      # left
            return 0.50, abs(dc) / cols
        if dr < 0 and dc == 0:      # up
            return 0.25, abs(dr) / rows
        if dr > 0 and dc == 0:      # down
            return 0.75, abs(dr) / rows
        raise ValueError(f"Non-axis-aligned segment: {seg}")

    labels = [[round(d,4), round(l,4)] for (d,l) in map(to_dir_and_len, segs)]
    return len(segs), labels


# -------------------------
# One-open two-path helpers
# -------------------------

def edgeset_of_path(path: List[Tuple[int,int]]) -> Set[frozenset]:
    es = set()
    for a, b in zip(path, path[1:]):
        es.add(frozenset((a, b)))
    return es

def list_internal_closed_walls(grid, rows, cols) -> List[Tuple[int,int,str]]:
    """(r,c,d) for closed walls where neighbor exists (i.e., not outer border)."""
    cands = []
    for r in range(rows):
        for c in range(cols):
            for d,(dr,dc) in DIRS.items():
                if grid[r][c][d]:
                    nr, nc = r + dr, c + dc
                    if in_bounds(nr, nc, rows, cols):
                        cands.append((r,c,d))
    return cands

def open_wall(grid, r, c, d):
    if not grid[r][c][d]:
        return
    grid[r][c][d] = False
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if in_bounds(nr, nc, len(grid), len(grid[0])):
        grid[nr][nc][OPP[d]] = False

def close_wall(grid, r, c, d):
    if grid[r][c][d]:
        return
    grid[r][c][d] = True
    dr, dc = DIRS[d]
    nr, nc = r + dr, c + dc
    if in_bounds(nr, nc, len(grid), len(grid[0])):
        grid[nr][nc][OPP[d]] = True

def would_make_2x2_room(grid, r, c, d) -> bool:
    """Lightweight guard: opening (r,c,d) would complete a fully open 2×2 block?"""
    rows, cols = len(grid), len(grid[0])
    dr, dc = DIRS[d]
    r2, c2 = r + dr, c + dc
    if not in_bounds(r2, c2, rows, cols):
        return False

    def is_open(a, b):
        (ra,ca),(rb,cb) = a,b
        if ra == rb and cb == ca+1:
            return (not grid[ra][ca]['E']) and (not grid[rb][cb]['W'])
        if ra == rb and cb == ca-1:
            return (not grid[ra][ca]['W']) and (not grid[rb][cb]['E'])
        if ca == cb and rb == ra+1:
            return (not grid[ra][ca]['S']) and (not grid[rb][cb]['N'])
        if ca == cb and rb == ra-1:
            return (not grid[ra][ca]['N']) and (not grid[rb][cb]['S'])
        return False

    # Treat (r,c,d) as opened for the check
    def open_or(a, b):
        (ra,ca),(rb,cb) = a,b
        if ra == rb and cb == ca+1:
            return is_open(a,b) or ((ra,ca)==(r,c) and d=='E') or ((rb,cb)==(r2,c2) and OPP[d]=='W')
        if ra == rb and cb == ca-1:
            return is_open(a,b) or ((ra,ca)==(r,c) and d=='W') or ((rb,cb)==(r2,c2) and OPP[d]=='E')
        if ca == cb and rb == ra+1:
            return is_open(a,b) or ((ra,ca)==(r,c) and d=='S') or ((rb,cb)==(r2,c2) and OPP[d]=='N')
        if ca == cb and rb == ra-1:
            return is_open(a,b) or ((ra,ca)==(r,c) and d=='N') or ((rb,cb)==(r2,c2) and OPP[d]=='S')
        return False

    quads = []
    if d in ('E','W'):
        rr = r-1
        if in_bounds(rr, c, rows, cols) and in_bounds(rr, c2, rows, cols):
            quads.append([(r,c),(r,c2),(rr,c),(rr,c2)])
        rr = r
        if in_bounds(rr+1, c, rows, cols) and in_bounds(rr+1, c2, rows, cols):
            quads.append([(r,c),(r,c2),(r+1,c),(r+1,c2)])
    else:
        cc = c-1
        if in_bounds(r, cc, rows, cols) and in_bounds(r2, cc, rows, cols):
            quads.append([(r,c),(r2,c),(r,cc),(r2,cc)])
        cc = c
        if in_bounds(r, cc+1, rows, cols) and in_bounds(r2, cc+1, rows, cols):
            quads.append([(r,c),(r2,c),(r,cc+1),(r2,cc+1)])

    for quad in quads:
        a,b,c3,d3 = quad
        edges = [(a,b),(a,c3),(d3,b),(d3,c3)]
        if all(open_or(u,v) for (u,v) in edges):
            return True
    return False

def find_two_paths_by_opening_one_wall(
    grid, rows, cols, rng: random.Random,
    s: Tuple[int,int], g: Tuple[int,int],
    max_overlap: float = 0.35,
    open_tries: int = 200,
    avoid_2x2: bool = True,
) -> Tuple[List[Tuple[int,int]], Optional[List[Tuple[int,int]]]]:
    """
    Compute P1 on the perfect maze. Try up to open_tries random internal walls:
      - open it, compute P2 (shortest path after opening)
      - accept if edge-overlap(P1,P2) / min(|P1|-1, |P2|-1) <= max_overlap
      - else revert and keep trying
    Returns (P1, P2 or None).
    """
    G0 = grid_to_graph(grid, rows, cols)
    P1 = shortest_path(G0, s, g)
    if not P1:
        return [], None

    es1 = edgeset_of_path(P1)
    best = (1.0, None, None)  # (overlap_ratio, (r,c,d), P2)

    # Pre-shuffle candidates for variety
    cands = list_internal_closed_walls(grid, rows, cols)
    rng.shuffle(cands)

    tries = 0
    i = 0
    while tries < open_tries and i < len(cands):
        r,c,d = cands[i]
        i += 1
        tries += 1

        if avoid_2x2 and would_make_2x2_room(grid, r, c, d):
            continue

        open_wall(grid, r, c, d)

        G = grid_to_graph(grid, rows, cols)
        P2 = shortest_path(G, s, g)

        if P2:
            es2 = edgeset_of_path(P2)
            denom = max(1, min(len(P1)-1, len(P2)-1))
            overlap = len(es1 & es2) / denom
            if overlap < best[0]:
                best = (overlap, (r,c,d), P2)
            if overlap <= max_overlap and es2 != es1:
                return P1, P2

        # revert and continue
        close_wall(grid, r, c, d)

    # If no candidate met the threshold, keep the best we saw (if any)
    if best[1] is not None:
        # Re-open that best one to reflect the image/JSON
        r,c,d = best[1]
        open_wall(grid, r, c, d)
        return P1, best[2]

    return P1, None


# -------------------------
# Geometry helpers (anchor + axis pads)
# -------------------------

def parse_pair(s: Optional[str]) -> Optional[Tuple[int,int]]:
    if s is None:
        return None
    a, b = s.split(",")
    return int(a), int(b)

def compute_geometry(rows: int, cols: int,
                     canvas: Tuple[int,int],
                     cell_px: Optional[int],
                     anchor: Optional[Tuple[int,int]],
                     pads: Tuple[int,int,int,int]) -> Tuple[int,int,int,int,int]:
    """
    Returns (canvas_w, canvas_h, cell_px, x0, y0)
    """
    W, H = canvas
    padL, padT, padR, padB = pads
    inner_w = W - padL - padR
    inner_h = H - padT - padB
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Canvas too small for given axis pads.")

    if cell_px is None:
        cell_px = min(inner_w // cols, inner_h // rows)
        if cell_px <= 0:
            raise ValueError(f"Grid {rows}x{cols} too large for inner region {inner_w}x{inner_h}.")

    maze_w = cols * cell_px
    maze_h = rows * cell_px
    if maze_w > inner_w or maze_h > inner_h:
        raise ValueError("cell_px too large for inner region with axes pads.")

    if anchor is None:
        x0 = padL + (inner_w - maze_w) // 2
        y0 = padT + (inner_h - maze_h) // 2
    else:
        x0, y0 = anchor  # absolute pixels from (0,0) top-left of canvas
    return W, H, cell_px, x0, y0


def cell_center_xy(r: int, c: int, cell_px: int, x0: int, y0: int) -> Tuple[float,float]:
    x = x0 + (c + 0.5) * cell_px
    y = y0 + (r + 0.5) * cell_px
    return x, y


# -------------------------
# Drawing (pixel-perfect with axes)
# -------------------------

def _line(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, w, color=(0,0,0)):
    draw.line([(int(x0),int(y0)), (int(x1),int(y1))], fill=color, width=int(w))

def _circle(draw: ImageDraw.ImageDraw, cx, cy, r, color):
    x0, y0 = int(cx - r), int(cy - r)
    x1, y1 = int(cx + r), int(cy + r)
    draw.ellipse([x0, y0, x1, y1], fill=color, outline=None)

def _load_font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def draw_maze_image_fixed(
    grid, rows: int, cols: int,
    *,
    canvas_w: int, canvas_h: int,
    cell_px: int, x0: int, y0: int,
    wall_px: int,
    knob_px: int,
    draw_grid: bool,
    draw_axes: bool,
    pads: Tuple[int,int,int,int],
    start: Tuple[int,int],
    goal: Tuple[int,int],
    out_png: str,
):
    padL, padT, padR, padB = pads
    maze_w = cols * cell_px
    maze_h = rows * cell_px

    im = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    dr = ImageDraw.Draw(im)

    # Optional light grid at cell borders (inside maze rect)
    if draw_grid and DEF_GRID_GRAY is not None:
        for cc in range(cols + 1):
            x = x0 + cc * cell_px
            _line(dr, x, y0, x, y0 + maze_h, DEF_GRID_PX, color=(DEF_GRID_GRAY,)*3)
        for rr in range(rows + 1):
            y = y0 + rr * cell_px
            _line(dr, x0, y, x0 + maze_w, y, DEF_GRID_PX, color=(DEF_GRID_GRAY,)*3)

    # Outer border
    _line(dr, x0,         y0,          x0 + maze_w, y0,          wall_px)  # top
    _line(dr, x0,         y0+maze_h,   x0 + maze_w, y0+maze_h,   wall_px)  # bottom
    _line(dr, x0,         y0,          x0,          y0 + maze_h, wall_px)  # left
    _line(dr, x0+maze_w,  y0,          x0+maze_w,   y0 + maze_h, wall_px)  # right

    # Inner walls per cell
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            xL = x0 + c * cell_px
            xR = xL + cell_px
            yT = y0 + r * cell_px
            yB = yT + cell_px
            if cell['N']:
                _line(dr, xL, yT, xR, yT, wall_px)
            if cell['S']:
                _line(dr, xL, yB, xR, yB, wall_px)
            if cell['E']:
                _line(dr, xR, yT, xR, yB, wall_px)
            if cell['W']:
                _line(dr, xL, yT, xL, yB, wall_px)

    # Start/Goal knobs at cell centers
    sr, sc = start
    gr, gc = goal
    cx_s, cy_s = cell_center_xy(sr, sc, cell_px, x0, y0)
    cx_g, cy_g = cell_center_xy(gr, gc, cell_px, x0, y0)
    _circle(dr, cx_s, cy_s, knob_px // 2, (255, 0, 0))
    _circle(dr, cx_g, cy_g, knob_px // 2, (0, 200, 0))

    # Axes: numeric ticks and captions
    if draw_axes:
        font = _load_font()
        # Column indices above top border
        for c in range(cols):
            cx = x0 + c * cell_px + cell_px // 2
            dr.text((cx - 3, max(0, y0 - 14)), str(c), fill=(0, 0, 0), font=font)
        dr.text((x0 + maze_w // 2 - 10, max(0, y0 - 28)), "col", fill=(0, 0, 0), font=font)

        # Row indices left of left border
        for r in range(rows):
            cy = y0 + r * cell_px + cell_px // 2
            dr.text((max(0, x0 - 18), cy - 6), str(r), fill=(0, 0, 0), font=font)
        dr.text((max(0, x0 - 35), y0 + maze_h // 2 - 6), "row", fill=(0, 0, 0), font=font)

    im.save(out_png)


# -------------------------
# Dataset writer
# -------------------------

def pick_distinct_cells(rng: random.Random, rows: int, cols: int):
    s = (rng.randrange(rows), rng.randrange(cols))
    g = (rng.randrange(rows), rng.randrange(cols))
    while g == s:
        g = (rng.randrange(rows), rng.randrange(cols))
    return s, g

def generate_dataset(
    count: int, rows: int, cols: int, out_jsonl: str,
    base_seed: int = 20250924,
    png_dir: Optional[str] = None,
    canvas: Tuple[int,int] = (DEF_CANVAS_W, DEF_CANVAS_H),
    cell_px: Optional[int] = None,
    anchor: Optional[Tuple[int,int]] = None,
    wall_px: int = DEF_WALL_PX,
    knob_px: int = DEF_KNOB_PX,
    draw_grid: bool = True,
    draw_axes: bool = True,
    pads: Tuple[int,int,int,int] = (AXIS_PAD_LEFT, AXIS_PAD_TOP, AXIS_PAD_RIGHT, AXIS_PAD_BOTTOM),
    two_path_rate: float = 0.0,
    max_overlap: float = 0.35,
    open_tries: int = 200,
    avoid_2x2: bool = True,
):
    """
    Writes JSONL where each record includes exact geometry.
    When two-path is chosen (Bernoulli two_path_rate), record also includes true_path2 / labels2.
    """
    seen = set()
    made = 0
    idx = 0
    if png_dir:
        import os
        os.makedirs(png_dir, exist_ok=True)

    with open(out_jsonl, "w") as f:
        while made < count:
            rng = random.Random(base_seed + idx)
            idx += 1
            grid = init_grid(rows, cols)
            carve_maze(grid, rows, cols, rng)
            sig = encode_maze(grid, rows, cols)
            if sig in seen:
                continue
            seen.add(sig)

            s, g = pick_distinct_cells(rng, rows, cols)

            # Either classic one-path, or try two-path via one-open
            do_two = (two_path_rate > 0.0) and (rng.random() < two_path_rate)
            if do_two:
                P1, P2 = find_two_paths_by_opening_one_wall(
                    grid, rows, cols, rng, s, g,
                    max_overlap=max_overlap, open_tries=open_tries, avoid_2x2=avoid_2x2
                )
            else:
                G = grid_to_graph(grid, rows, cols)
                P1 = shortest_path(G, s, g)
                P2 = None

            if not P1:
                continue

            # Exact geometry for this image
            canvas_w, canvas_h, cell_px_eff, x0, y0 = compute_geometry(rows, cols, canvas, cell_px, anchor, pads)
            maze_w = cols * cell_px_eff
            maze_h = rows * cell_px_eff
            maze_bbox = [y0, y0 + maze_h, x0, x0 + maze_w]

            K1, labels1 = aggregate_segments_labels(P1, rows, cols)
            rec = {
                "index": made,
                "rows": rows,
                "cols": cols,
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
                "cell_px": cell_px_eff,
                "x0": x0,
                "y0": y0,
                "axis_pads": [pads[0], pads[1], pads[2], pads[3]],
                "maze_bbox": maze_bbox,
                "wall_px": wall_px,
                "knob_px": knob_px,
                "start_pos": [s[0], s[1]],
                "end_pos": [g[0], g[1]],
                "true_path": [[r, c] for (r, c) in P1],
                "labels": [K1, labels1],
            }

            if P2:
                K2, labels2 = aggregate_segments_labels(P2, rows, cols)
                rec["true_path2"] = [[r, c] for (r, c) in P2]
                rec["labels2"] = [K2, labels2]

            f.write(json.dumps(rec) + "\n")

            if png_dir:
                out_png = f"{png_dir}/maze_{made:05d}.png"
                draw_maze_image_fixed(
                    grid, rows, cols,
                    canvas_w=canvas_w, canvas_h=canvas_h,
                    cell_px=cell_px_eff, x0=x0, y0=y0,
                    wall_px=wall_px, knob_px=knob_px,
                    draw_grid=draw_grid, draw_axes=draw_axes, pads=pads,
                    start=s, goal=g, out_png=out_png,
                )

            made += 1
    return made, len(seen)


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Maze image and dataset generator (fixed geometry + axes) with optional two-paths via opening one wall.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("image", help="Generate a single maze PNG with axes.")
    p_img.add_argument("--rows", type=int, default=10)
    p_img.add_argument("--cols", type=int, default=10)
    p_img.add_argument("--seed", type=int, default=42)
    p_img.add_argument("--start", type=str, default=None, help="start 'r,c' (optional)")
    p_img.add_argument("--goal", type=str, default=None, help="goal 'r,c' (optional)")
    p_img.add_argument("--canvas", type=str, default=None, help="W,H (default 720,480)")
    p_img.add_argument("--cell_px", type=int, default=None, help="Fixed cell size in px (optional)")
    p_img.add_argument("--anchor", type=str, default=None, help="x0,y0 (optional absolute top-left of maze rect)")
    p_img.add_argument("--wall_px", type=int, default=DEF_WALL_PX)
    p_img.add_argument("--knob_px", type=int, default=DEF_KNOB_PX)
    p_img.add_argument("--no-grid", action="store_true")
    p_img.add_argument("--no-axes", action="store_true")
    p_img.add_argument("--two-path", action="store_true", help="Open one internal wall to create a second path if overlap is small enough.")
    p_img.add_argument("--max-overlap", type=float, default=0.35, help="Edge overlap threshold between P1 and P2 (ratio).")
    p_img.add_argument("--open-tries", type=int, default=200, help="How many random closed walls to try to open.")
    p_img.add_argument("--avoid-2x2", action="store_true", help="Avoid creating fully open 2x2 rooms with the single opening.")
    p_img.add_argument("--out", type=str, required=True)

    p_ds = sub.add_parser("dataset", help="Generate JSONL (+PNGs) with axes and precise geometry.")
    p_ds.add_argument("--count", type=int, default=5000)
    p_ds.add_argument("--rows", type=int, default=10)
    p_ds.add_argument("--cols", type=int, default=10)
    p_ds.add_argument("--out", type=str, default="info_labels.jsonl")
    p_ds.add_argument("--base-seed", type=int, default=20250935)
    p_ds.add_argument("--png-dir", type=str, default=None, help="Directory to write per-maze PNGs")
    p_ds.add_argument("--canvas", type=str, default=None, help="W,H (default 720,480)")
    p_ds.add_argument("--cell_px", type=int, default=None, help="Fixed cell size in px (optional)")
    p_ds.add_argument("--anchor", type=str, default=None, help="x0,y0 (optional absolute top-left of maze rect)")
    p_ds.add_argument("--wall_px", type=int, default=DEF_WALL_PX)
    p_ds.add_argument("--knob_px", type=int, default=DEF_KNOB_PX)
    p_ds.add_argument("--no-grid", action="store_true")
    p_ds.add_argument("--no-axes", action="store_true")
    p_ds.add_argument("--two-path-rate", type=float, default=1, help="Probability to attempt two-path creation by opening one wall.")
    p_ds.add_argument("--max-overlap", type=float, default=0.3)
    p_ds.add_argument("--open-tries", type=int, default=200)
    p_ds.add_argument("--avoid-2x2", action="store_true")

    args = parser.parse_args()

    def _parse_pair(s, default_pair):
        if s is None:
            return default_pair
        a, b = s.split(",")
        return (int(a), int(b))

    if args.cmd == "image":
        rng = random.Random(args.seed)
        rows, cols = args.rows, args.cols
        grid = init_grid(rows, cols)
        carve_maze(grid, rows, cols, rng)

        def parse_rc(s):
            if s is None:
                return None
            r, c = s.split(",")
            return int(r), int(c)

        start = parse_rc(args.start)
        goal  = parse_rc(args.goal)
        if start is None or goal is None:
            s, g = pick_distinct_cells(rng, rows, cols)
        else:
            s, g = start, goal

        if args.two_path:
            # try to create second path by opening one wall
            _P1, _P2 = find_two_paths_by_opening_one_wall(
                grid, rows, cols, rng, s, g,
                max_overlap=args.max_overlap, open_tries=args.open_tries, avoid_2x2=args.avoid_2x2
            )
            # we don't need the paths here to draw—grid already reflects the chosen opening

        canvas = _parse_pair(args.canvas, (DEF_CANVAS_W, DEF_CANVAS_H))
        anchor = _parse_pair(args.anchor, None)
        pads = (AXIS_PAD_LEFT, AXIS_PAD_TOP, AXIS_PAD_RIGHT, AXIS_PAD_BOTTOM)

        W, H, cell_px, x0, y0 = compute_geometry(
            rows, cols, canvas, args.cell_px, anchor, pads
        )
        draw_maze_image_fixed(
            grid, rows, cols,
            canvas_w=W, canvas_h=H,
            cell_px=cell_px, x0=x0, y0=y0,
            wall_px=args.wall_px, knob_px=args.knob_px,
            draw_grid=not args.no_grid, draw_axes=not args.no_axes, pads=pads,
            start=s, goal=g, out_png=args.out
        )

    elif args.cmd == "dataset":
        canvas = _parse_pair(args.canvas, (DEF_CANVAS_W, DEF_CANVAS_H))
        anchor = _parse_pair(args.anchor, None)
        pads = (AXIS_PAD_LEFT, AXIS_PAD_TOP, AXIS_PAD_RIGHT, AXIS_PAD_BOTTOM)

        made, uniq = generate_dataset(
            count=args.count,
            rows=args.rows,
            cols=args.cols,
            out_jsonl=args.out,
            base_seed=args.base_seed,
            png_dir=args.png_dir,
            canvas=canvas,
            cell_px=args.cell_px,
            anchor=anchor,
            wall_px=args.wall_px,
            knob_px=args.knob_px,
            draw_grid=not args.no_grid,
            draw_axes=not args.no_axes,
            pads=pads,
            two_path_rate=max(0.0, min(1.0, args.two_path_rate)),
            max_overlap=args.max_overlap,
            open_tries=args.open_tries,
            avoid_2x2=args.avoid_2x2,
        )
        print(f"Wrote {made} records to {args.out} (unique mazes: {uniq}).")
        if args.png_dir:
            print(f"PNGs saved to {args.png_dir}")

if __name__ == "__main__":
    main()
