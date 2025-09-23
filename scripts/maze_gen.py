#!/usr/bin/env python3
"""
maze_gen.py

Image:
  python maze_gen.py image --rows 5 --cols 5 --seed 42 --out maze.png
  # Optional: --start "r,c" --goal "r,c"

Dataset (JSONL + optional PNGs):
  python maze_gen.py dataset --count 5000 --rows 10 --cols 10 --out info_labels.jsonl --png-dir mazes_out
"""

import argparse
import hashlib
import json
import random
from collections import deque
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

DIRS = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
OPP = {'N':'S', 'S':'N', 'E':'W', 'W':'E'}
DIR_ORDER = ('N','E','S','W')  # for deterministic hashing

def init_grid(rows: int, cols: int):
    return [[{'N': True, 'S': True, 'E': True, 'W': True, 'visited': False}
             for _ in range(cols)] for __ in range(rows)]

def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols

def carve_maze(grid, rows: int, cols: int, rng: random.Random, start_r: Optional[int]=None, start_c: Optional[int]=None):
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
    if len(true_path) <= 1:
        return 0, []
    segs = []
    pr, pc = true_path[0]
    cur_step = (true_path[1][0] - pr, true_path[1][1] - pc)
    acc_dr, acc_dc = cur_step
    pr, pc = true_path[1]
    for (r, c) in true_path[2:]:
        step = (r - pr, c - pc)
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
        if dr > 0 and dc == 0:      # right
            return 0.0, abs(dr) / rows
        if dr < 0 and dc == 0:      # left
            return 0.5, abs(dr) / rows
        if dc > 0 and dr == 0:      # up
            return 0.25, abs(dc) / cols
        if dc < 0 and dr == 0:      # down
            return 0.75, abs(dc) / cols
        raise ValueError(f"Non-axis-aligned segment: {seg}")

    labels = [[round(d,4), round(l,4)] for (d,l) in map(to_dir_and_len, segs)]
    return len(segs), labels

def draw_maze_image(grid, rows: int, cols: int, start, goal, out_png: str,
                    wall_width: float = 3.5, grid_alpha: float = 0.25, grid_lw: float = 0.8):
    inches = (7.2, 4.8)  # 720x480 @ 100 dpi
    dpi = 100
    fig = plt.figure(figsize=inches, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, rows - 0.5)
    ax.set_ylim(-0.5, cols - 0.5)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Axis ticks at cell centers
    ax.set_xticks(range(0, rows)); ax.set_xlabel("row")
    ax.set_yticks(range(0, cols)); ax.set_ylabel("col")

    # Alignment grid at cell borders (half-integers)
    ax.set_axisbelow(True)
    for xb in np.arange(-0.5, rows - 0.5 + 1, 1.0):
        ax.plot([xb, xb], [-0.5, cols - 0.5], color='0.85', linewidth=grid_lw, alpha=grid_alpha, zorder=0)
    for yb in np.arange(-0.5, cols - 0.5 + 1, 1.0):
        ax.plot([-0.5, rows - 0.5], [yb, yb], color='0.85', linewidth=grid_lw, alpha=grid_alpha, zorder=0)

    # Outer border
    ax.plot([-0.5, rows-0.5], [-0.5, -0.5], 'k-', linewidth=wall_width, zorder=1)
    ax.plot([-0.5, rows-0.5], [cols-0.5, cols-0.5], 'k-', linewidth=wall_width, zorder=1)
    ax.plot([-0.5, -0.5], [-0.5, cols-0.5], 'k-', linewidth=wall_width, zorder=1)
    ax.plot([rows-0.5, rows-0.5], [-0.5, cols-0.5], 'k-', linewidth=wall_width, zorder=1)

    # Inner walls
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            cx, cy = r, c
            if cell['N']:
                ax.plot([cx-0.5, cx+0.5], [cy+0.5, cy+0.5], 'k-', linewidth=wall_width, zorder=2)
            if cell['S']:
                ax.plot([cx-0.5, cx+0.5], [cy-0.5, cy-0.5], 'k-', linewidth=wall_width, zorder=2)
            if cell['E']:
                ax.plot([cx+0.5, cx+0.5], [cy-0.5, cy+0.5], 'k-', linewidth=wall_width, zorder=2)
            if cell['W']:
                ax.plot([cx-0.5, cx-0.5], [cy-0.5, cy+0.5], 'k-', linewidth=wall_width, zorder=2)

    # Start/Goal markers at integer centers
    sr, sc = start
    gr, gc = goal
    ax.scatter(sr, sc, s=300, c='red', edgecolors='none', zorder=3)
    ax.scatter(gr, gc, s=300, c='green', edgecolors='none', zorder=3)

    plt.tight_layout(pad=0.6)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def pick_distinct_cells(rng: random.Random, rows: int, cols: int):
    s = (rng.randrange(rows), rng.randrange(cols))
    g = (rng.randrange(rows), rng.randrange(cols))
    while g == s:
        g = (rng.randrange(rows), rng.randrange(cols))
    return s, g

def generate_dataset(count: int, rows: int, cols: int, out_jsonl: str, base_seed: int = 20250924,
                     png_dir: Optional[str] = None):
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
            G = grid_to_graph(grid, rows, cols)
            path = shortest_path(G, s, g)
            if not path:
                continue

            K, labels = aggregate_segments_labels(path, rows, cols)
            rec = {
                "index": made,
                "start_pos": [s[0], s[1]],
                "end_pos": [g[0], g[1]],
                "true_path": [[r, c] for (r, c) in path],
                "labels": [K, labels],
            }
            f.write(json.dumps(rec) + "\n")

            if png_dir:
                out_png = f"{png_dir}/maze_{made:05d}.png"
                draw_maze_image(grid, rows, cols, s, g, out_png)

            made += 1
    return made, len(seen)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Maze image and dataset generator.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("image", help="Generate a single maze PNG.")
    p_img.add_argument("--rows", type=int, default=5)
    p_img.add_argument("--cols", type=int, default=5)
    p_img.add_argument("--seed", type=int, default=42)
    p_img.add_argument("--start", type=str, default=None, help="start 'r,c' (optional)")
    p_img.add_argument("--goal", type=str, default=None, help="goal 'r,c' (optional)")
    p_img.add_argument("--out", type=str, required=True)

    p_ds = sub.add_parser("dataset", help="Generate a JSONL dataset of unique mazes.")
    p_ds.add_argument("--count", type=int, default=5000)
    p_ds.add_argument("--rows", type=int, default=10)
    p_ds.add_argument("--cols", type=int, default=10)
    p_ds.add_argument("--out", type=str, default="info_labels.jsonl")
    p_ds.add_argument("--base-seed", type=int, default=20250924)
    p_ds.add_argument("--png-dir", type=str, default=None, help="Directory to write per-maze PNGs")

    args = parser.parse_args()

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
        goal = parse_rc(args.goal)
        if start is None or goal is None:
            s, g = pick_distinct_cells(rng, rows, cols)
        else:
            s, g = start, goal

        draw_maze_image(grid, rows, cols, s, g, args.out)

    elif args.cmd == "dataset":
        made, uniq = generate_dataset(
            count=args.count,
            rows=args.rows,
            cols=args.cols,
            out_jsonl=args.out,
            base_seed=args.base_seed,
            png_dir=args.png_dir
        )
        print(f"Wrote {made} records to {args.out} (unique mazes: {uniq}).")
        if args.png_dir:
            print(f"PNGs saved to {args.png_dir}")

if __name__ == "__main__":
    main()
