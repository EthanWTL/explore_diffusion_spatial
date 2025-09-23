import os
from glob import glob

img_dir = "datasets/10by10/images"
out_txt = "datasets/10by10/images.txt"

# Ensure parent folder for images.txt exists
os.makedirs(os.path.dirname(out_txt), exist_ok=True)

files = sorted(glob(os.path.join(img_dir, "*.png")))
with open(out_txt, "w", encoding="utf-8") as f:
    for p in files:
        f.write("images/" + os.path.basename(p) + "\n")   # write just the filename
        # f.write(p + "\n")                   # <- use this to write full path
print(f"Wrote {len(files)} names to {out_txt}")
