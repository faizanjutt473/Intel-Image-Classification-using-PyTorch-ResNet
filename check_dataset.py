import os

BASE = "data"
for split in ["seg_train", "seg_test"]:
    root = os.path.join(BASE, split, split)
    print(f"\n{split}  ({root})")
    total = 0
    for c in sorted(os.listdir(root)):
        p = os.path.join(root, c)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            total += n
            print(f"  {c:<10} {n}")
    print(f"  TOTAL      {total}")

import torch
print("\nPyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())