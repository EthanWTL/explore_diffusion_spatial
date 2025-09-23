#!/usr/bin/env python3
import os, json

IN_JSONL = "datasets/10by10/info_labels.jsonl"
IMG_DIR  = "datasets/10by10/images"
OUT_TXT  = "datasets/10by10/prompts.txt"

os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)

TEMPLATE = (
    "Solve the maze. You are given a maze represented on a grid. "
    "The starting point is located at row {sr} column {sc} marked with a red knob "
    "and the end point is located at row {gr} column {gc} marked with a green knob. "
    "Your task is to find a valid path through the maze from the starting point to the end point "
    "while following the maze's walls and constraints. "
)

count = 0
with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_TXT, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        idx = rec["index"]
        sr, sc = rec["start_pos"]
        gr, gc = rec["end_pos"]
        img_path = f"{IMG_DIR}/maze_{idx:05d}.png"
        prompt = TEMPLATE.format(sr=sr, sc=sc, gr=gr, gc=gc, img=img_path)
        fout.write(prompt + "\n")
        count += 1

print(f"Wrote {count} prompts to {OUT_TXT}")
