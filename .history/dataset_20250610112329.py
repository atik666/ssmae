import os
import random
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR100

def prepare_cifar100_semi_supervised(
    root_dir: str,
    labeled_frac: float = 0.10,
    seed: int = 42
):
    """
    1) Downloads CIFAR-100 train+test.
    2) Under cifar100_ss/train/, creates:
       - labeled/    ← 10% of train per class
       - unlabeled/  ← 100% of train per class (including those in 'labeled/')
    3) Under cifar100_ss/test/, saves all 10k test images by class (unchanged).
    """
    random.seed(seed)

    # ─── STEP 1: DOWNLOAD TRAINING SET ───────────────────────────────────────────
    train_ds = CIFAR100(root=root_dir, train=True, download=True)
    train_data = train_ds.data       # (50000, 32, 32, 3)
    train_labels = train_ds.targets  # list of 50000 ints ∈ [0..99]
    class_names = train_ds.classes   # list of 100 class strings

    num_train = len(train_labels)    # 50 000
    num_classes = len(class_names)   # 100

    # Build: class_idx → [all train‐indices in that class]
    indices_per_class = {c: [] for c in range(num_classes)}
    for idx, cls in enumerate(train_labels):
        indices_per_class[cls].append(idx)

    # Randomly pick 10% of each class to be “labeled”
    labeled_indices = set()
    for cls, idx_list in indices_per_class.items():
        n_in_class = len(idx_list)            # should be 500
        n_labeled = int(round(n_in_class * labeled_frac))
        chosen = random.sample(idx_list, n_labeled)
        labeled_indices.update(chosen)

    # ─── STEP 2: DOWNLOAD TEST SET ───────────────────────────────────────────────
    test_ds = CIFAR100(root=root_dir, train=False, download=True)
    test_data = test_ds.data       # (10000, 32, 32, 3)
    test_labels = test_ds.targets  # list of 10000 ints
    # class_names is the same list of 100 names

    # ─── STEP 3: CREATE OUTPUT FOLDERS ────────────────────────────────────────────
    out_root = Path(root_dir) / "cifar100_ss"

    # TRAIN→ labeled/ and unlabeled/
    train_root = out_root / "train"
    labeled_root = train_root / "labeled"
    unlabeled_root = train_root / "unlabeled"

    for base in (labeled_root, unlabeled_root):
        (base).mkdir(parents=True, exist_ok=True)
        for cls_name in class_names:
            (base / cls_name).mkdir(exist_ok=True)

    # TEST → keep “as is” under test/<class_name>/
    test_root = out_root / "test"
    for cls_name in class_names:
        (test_root / cls_name).mkdir(parents=True, exist_ok=True)

    # ─── STEP 4: SAVE TRAIN IMAGES ─────────────────────────────────────────────────
    # For each train idx:
    #   - always save into unlabeled/<class_name>/
    #   - if idx in labeled_indices → also save into labeled/<class_name>/
    for idx in range(num_train):
        img_arr = train_data[idx]             # np.uint8 array (32×32×3)
        cls = train_labels[idx]
        cls_name = class_names[cls]
        img = Image.fromarray(img_arr)

        fname = f"{idx:05d}.png"

        # Save to unlabeled
        img.save(unlabeled_root / cls_name / fname)

        # If chosen as labeled, also save to labeled/
        if idx in labeled_indices:
            img.save(labeled_root / cls_name / fname)

    # ─── STEP 5: SAVE TEST IMAGES “AS IS” ───────────────────────────────────────────
    # Simply iterate over test_data and dump each into test/<class_name>/ using new filenames
    for idx in range(len(test_labels)):  # 10000 test images
        img_arr = test_data[idx]
        cls = test_labels[idx]
        cls_name = class_names[cls]
        img = Image.fromarray(img_arr)

        fname = f"{idx:05d}.png"
        img.save(test_root / cls_name / fname)

    # ─── FINISHED ─────────────────────────────────────────────────────────────────
    print("Done.")
    print(f"  → Train/Labeled   : {labeled_root}   (≈ {labeled_frac*100:.0f}% of 50 000 train images)")
    print(f"  → Train/Unlabeled : {unlabeled_root} (all 50 000 train images)")
    print(f"  → Test (unchanged): {test_root}      (all 10 000 test images)")

if __name__ == "__main__":
    # Example: download into ./data/, 10% labeled
    prepare_cifar100_semi_supervised(root_dir="./data", labeled_frac=0.10, seed=123)


