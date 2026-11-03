from __future__ import annotations

"""Image-wise segmentation evaluation (mDice / mIoU / HD95 / ASD).

Evaluates saved prediction masks against the ground truth of this
repository's benchmarks. The metric implementation follows the project
reference evaluation:

- Region metrics (mDice, mIoU, ...) are computed pixel-wise from the
  confusion matrix of binarized masks (eps = 1e-8) and averaged over images.
- HD95 is the 95th percentile of the bidirectional surface distances
  (surfaces extracted by 3x3 binary erosion; distances via the Euclidean
  distance transform).
- ASD is the symmetric average surface distance: the mean of the two
  directed means.
- An empty prediction is penalized with the image diagonal
  sqrt(H^2 + W^2); images whose ground truth is empty are skipped.
- Predictions are resized to the ground-truth size (nearest neighbour)
  when the shapes differ.

Supported evaluation targets
----------------------------
- Pooled within-dataset benchmarks (ISIC2017, ISIC2018, COVID-19,
  BRISC2025): the ground truth is the held-out 7:1:2 test split. The split
  recorded in outputs_lesion/<dataset>/split.json (written by
  train_lesion.py) is preferred; otherwise the deterministic split is
  recomputed with --split-seed.
- polypImg: the five official test subsets. Prediction folders are
  expected per subset, e.g. results_lesion/polypImg/<subset>/region_masks
  (as produced by eval_lesion.py). The summary additionally reports the
  sample-count-weighted average over subsets.
- Cross-dataset targets (busi, datasetb, stu, montgomery, jsrt, ...):
  flat image/mask pools under data/<Name>/.

Examples
--------
    python evaluate_metrics.py --dataset ISIC2017 --pred-dir results_lesion/ISIC2017/test/region_masks
    python evaluate_metrics.py --dataset polypImg --pred-dir results_lesion/polypImg
    python evaluate_metrics.py --dataset montgomery --pred-dir results_cross/shenzhen_to_montgomery/Montgomery/region_masks
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

from binary_dataloader import (
    SUPPORTED_DATASETS,
    collect_image_mask_pairs,
    default_data_root,
    get_dataset_spec,
    split_train_validation_test,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Flat cross-dataset pools (images/ + masks/), mirroring train_cross.py.
CROSS_TARGETS = {
    "isic2016": "ISIC2016",
    "ph2": "PH2",
    "fuseg": "FUSeg",
    "busi": "BUSI",
    "datasetb": "DatasetB",
    "stu": "STU",
    "shenzhen": "Shenzhen",
    "montgomery": "Montgomery",
    "jsrt": "JSRT",
}

METRIC_KEYS = (
    "mDice",
    "mIoU",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "HD95",
    "ASD",
    "Directed_ASD",
    "nHD95",
)


# ---------------------------------------------------------------------------
# Metric implementation (reference protocol)
# ---------------------------------------------------------------------------
def get_binary_surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask.astype(bool) ^ eroded


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not np.any(gt_mask):
        return np.nan
    if not np.any(pred_mask):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_surface = get_binary_surface(pred_mask)
    gt_surface = get_binary_surface(gt_mask)

    if not np.any(pred_surface) or not np.any(gt_surface):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_to_gt_dist = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred_dist = distance_transform_edt(~pred_surface)[gt_surface]
    surface_distances = np.concatenate([pred_to_gt_dist, gt_to_pred_dist])

    if surface_distances.size == 0:
        return 0.0
    return float(np.percentile(surface_distances, 95))


def compute_directed_asd(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not np.any(gt_mask):
        return np.nan
    if not np.any(pred_mask):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_surface = get_binary_surface(pred_mask)
    gt_surface = get_binary_surface(gt_mask)

    if not np.any(pred_surface) or not np.any(gt_surface):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_to_gt_dist = distance_transform_edt(~gt_surface)[pred_surface]
    if pred_to_gt_dist.size == 0:
        return 0.0
    return float(pred_to_gt_dist.mean())


def compute_symmetric_asd(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not np.any(gt_mask):
        return np.nan
    if not np.any(pred_mask):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_surface = get_binary_surface(pred_mask)
    gt_surface = get_binary_surface(gt_mask)

    if not np.any(pred_surface) or not np.any(gt_surface):
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))

    pred_to_gt_dist = distance_transform_edt(~gt_surface)[pred_surface]
    gt_to_pred_dist = distance_transform_edt(~pred_surface)[gt_surface]

    if pred_to_gt_dist.size == 0 and gt_to_pred_dist.size == 0:
        return 0.0
    if pred_to_gt_dist.size == 0 or gt_to_pred_dist.size == 0:
        h, w = gt_mask.shape
        return float(np.sqrt(h * h + w * w))
    return float((pred_to_gt_dist.mean() + gt_to_pred_dist.mean()) / 2.0)


def compute_normalized_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    hd95 = compute_hd95(pred_mask, gt_mask)
    if np.isnan(hd95):
        return np.nan
    h, w = gt_mask.shape
    diagonal = np.sqrt(h * h + w * w)
    if diagonal < 1e-8:
        return 0.0
    return float(hd95 / diagonal * 100.0)


def compute_region_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray) -> Dict[str, float]:
    """Pixel-level confusion-matrix metrics (identical to sklearn-based
    confusion_matrix(y_true, y_pred, labels=[0, 1]) with eps = 1e-8)."""
    eps = 1e-8
    tp = float(np.logical_and(pred_binary, gt_binary).sum())
    fp = float(np.logical_and(pred_binary, ~gt_binary).sum())
    fn = float(np.logical_and(~pred_binary, gt_binary).sum())
    tn = float(np.logical_and(~pred_binary, ~gt_binary).sum())

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    miou = tp / (tp + fp + fn + eps)
    return {
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "mDice": f1,  # Dice == F1 for binary masks
        "mIoU": miou,
    }


def evaluate_image(pred_path: Path, gt_path: Path, threshold: float) -> Tuple[Dict[str, float], bool]:
    """Return (metrics, resized). Raises if either mask cannot be read."""
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise OSError(f"Could not read prediction: {pred_path}")
    if gt is None:
        raise OSError(f"Could not read ground truth: {gt_path}")

    resized = pred.shape != gt.shape
    if resized:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred_binary = pred.astype(np.float32) / 255.0 >= threshold
    gt_binary = gt.astype(np.float32) / 255.0 >= 0.5
    if not np.any(gt_binary):
        raise ValueError("empty ground truth")

    metrics = compute_region_metrics(pred_binary, gt_binary)
    metrics["HD95"] = compute_hd95(pred_binary, gt_binary)
    metrics["Directed_ASD"] = compute_directed_asd(pred_binary, gt_binary)
    metrics["ASD"] = compute_symmetric_asd(pred_binary, gt_binary)
    metrics["nHD95"] = compute_normalized_hd95(pred_binary, gt_binary)
    return metrics, resized


def find_prediction(pred_dir: Path, image_stem: str) -> Path | None:
    for extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = pred_dir / f"{image_stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def evaluate_pairs(
    pred_dir: Path,
    pairs: List[Tuple[Path, Path]],
    threshold: float,
    subset_name: str = "",
) -> Tuple[List[List[object]], Dict[str, float]]:
    """Evaluate every (image, ground-truth) pair against pred_dir/<stem>.*"""
    rows: List[List[object]] = []
    totals = {key: 0.0 for key in METRIC_KEYS}
    valid = 0
    missing: List[str] = []
    resized_count = 0
    label = subset_name or str(pred_dir)

    for image_path, gt_path in tqdm(pairs, desc=f"Evaluating {label}"):
        pred_path = find_prediction(pred_dir, image_path.stem)
        if pred_path is None:
            missing.append(image_path.name)
            continue
        try:
            metrics, resized = evaluate_image(pred_path, gt_path, threshold)
        except ValueError:
            continue  # empty ground truth: skipped, as in the reference protocol
        if resized:
            resized_count += 1
        for key in METRIC_KEYS:
            totals[key] += metrics[key]
        rows.append([image_path.stem] + [f"{metrics[key]:.4f}" for key in METRIC_KEYS])
        valid += 1

    if missing:
        examples = ", ".join(missing[:5])
        print(
            f"  warning: {len(missing)} prediction(s) not found in {pred_dir} "
            f"(examples: {examples})"
        )
    if resized_count:
        print(f"  Resized {resized_count} prediction mask(s) to match GT size.")
    if valid == 0:
        raise RuntimeError(f"No evaluated images under {pred_dir} (GT root mismatch?)")

    mean_metrics = {key: totals[key] / valid for key in METRIC_KEYS}
    return rows, mean_metrics


def save_rows_csv(csv_path: Path, rows: List[List[object]], mean_metrics: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Image", *METRIC_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerow(["MEAN"] + [f"{mean_metrics[key]:.4f}" for key in METRIC_KEYS])
    print(f"Per-image results saved to: {csv_path}")


def print_summary(title: str, mean_metrics: Dict[str, float]) -> None:
    print(f"==== {title} (image-wise mean) ====")
    print(
        f"  mDice={mean_metrics['mDice']:.4f}  mIoU={mean_metrics['mIoU']:.4f}  "
        f"HD95={mean_metrics['HD95']:.4f}  ASD={mean_metrics['ASD']:.4f}"
    )


# ---------------------------------------------------------------------------
# Ground-truth resolution
# ---------------------------------------------------------------------------
def _probe_pool_suffix(images: Path, masks: Path) -> Tuple[List[Tuple[Path, Path]], str]:
    for suffix in ("", "_Segmentation", "_segmentation", "_mask"):
        try:
            pairs = collect_image_mask_pairs(images, masks, suffix)
        except (FileNotFoundError, RuntimeError):
            continue
        if pairs:
            return pairs, suffix
    raise FileNotFoundError(
        f"No image/mask pairs under {images} / {masks}. Check the layout under ./data."
    )


def resolve_ground_truth(args: argparse.Namespace):
    """Return {subset_name: (gt_pairs, default_pred_dir_hint)}."""
    dataset_key = args.dataset.casefold()

    if dataset_key in {name.casefold() for name in SUPPORTED_DATASETS}:
        spec = get_dataset_spec(args.dataset, args.data_root)
        if spec.split_mode == "pool":
            pool_pairs = collect_image_mask_pairs(
                spec.train_image_dir, spec.train_mask_dir, spec.train_mask_suffix
            )
            split_path = (args.checkpoint_root / args.dataset / "split.json").resolve()
            if split_path.is_file():
                with split_path.open("r", encoding="utf-8") as handle:
                    split_info = json.load(handle)
                by_image = {str(image): (image, mask) for image, mask in pool_pairs}
                test_names = set(split_info.get("test", []))
                pairs = [by_image[name] for name in sorted(test_names) if name in by_image]
                if pairs:
                    print(f"  [test split] {len(pairs)} samples from {split_path}")
                else:
                    _, _, pairs = split_train_validation_test(pool_pairs, seed=args.split_seed)
                    print(f"  [test split] split.json matched no images; recomputed ({len(pairs)} samples)")
            else:
                _, _, pairs = split_train_validation_test(pool_pairs, seed=args.split_seed)
                print(
                    f"  [test split] split.json not found; recomputed 7:1:2 with "
                    f"seed {args.split_seed} ({len(pairs)} samples)"
                )
            return {"test": pairs}

        subsets = {}
        for subset_name, test_spec in spec.test_sets.items():
            subsets[subset_name] = collect_image_mask_pairs(
                test_spec.image_dir, test_spec.mask_dir, test_spec.mask_suffix
            )
        return subsets

    if dataset_key in CROSS_TARGETS:
        folder = CROSS_TARGETS[dataset_key]
        images = args.data_root / folder / "images"
        masks = args.data_root / folder / "masks"
        pairs, suffix = _probe_pool_suffix(images, masks)
        print(f"  [{folder}] pool={len(pairs)} (mask suffix '{suffix}')")
        return {"test": pairs}

    raise ValueError(
        f"Unsupported dataset '{args.dataset}'. Within-dataset: "
        f"{', '.join(SUPPORTED_DATASETS)}; cross-dataset: {', '.join(sorted(CROSS_TARGETS))}."
    )


def find_subset_pred_dir(pred_root: Path, subset_name: str) -> Path:
    """Locate a subset's prediction folder (eval_lesion layout aware).

    Resolution order: <root>/<subset>/region_masks, <root>/<subset>, then
    <root> itself when it is a non-empty mask folder (an explicitly passed
    --pred-dir is always respected).
    """
    for candidate in (
        pred_root / subset_name / "region_masks",
        pred_root / subset_name,
    ):
        if candidate.is_dir():
            return candidate
    if pred_root.is_dir() and any(pred_root.iterdir()):
        return pred_root  # flat folder already pointing at masks
    return pred_root / subset_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved prediction masks with the manuscript metrics "
            "(mDice, mIoU, HD95, ASD)."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Within-dataset: polypImg, ISIC2017, ISIC2018, COVID-19, BRISC2025; "
            "cross-dataset target: busi, datasetb, stu, montgomery, jsrt, shenzhen, "
            "isic2016, ph2, fuseg."
        ),
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing predicted masks (or, for polypImg, the parent of the "
            "per-subset folders). Defaults to results_lesion/<dataset>."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs_lesion"),
        help="Used to locate the split.json written by train_lesion.py (pooled datasets).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Seed for recomputing the 7:1:2 split when split.json is unavailable.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Per-image CSV path. Defaults to <pred-dir>/evaluation_results.csv "
             "(polypImg: per-subset files plus a summary CSV).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.checkpoint_root = args.checkpoint_root.expanduser().resolve()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")

    pred_root = (
        args.pred_dir
        if args.pred_dir is not None
        else Path("results_lesion") / args.dataset
    ).expanduser().resolve()

    subsets = resolve_ground_truth(args)
    is_multi_subset = len(subsets) > 1
    subset_means = []
    sample_counts = []

    for subset_name, pairs in subsets.items():
        pred_dir = find_subset_pred_dir(pred_root, subset_name)
        print(f"\n[{args.dataset}/{subset_name}] predictions: {pred_dir}  GT pairs: {len(pairs)}")
        rows, mean_metrics = evaluate_pairs(pred_dir, pairs, args.threshold, subset_name)
        csv_path = (
            args.output_csv
            if args.output_csv is not None and not is_multi_subset
            else pred_dir / "evaluation_results.csv"
        )
        save_rows_csv(csv_path, rows, mean_metrics)
        print_summary(f"{args.dataset}/{subset_name}", mean_metrics)
        subset_means.append((subset_name, mean_metrics))
        sample_counts.append(len(rows))

    if is_multi_subset:
        # Sample-count-weighted average over subsets (the manuscript protocol
        # for the polyp benchmark), plus the plain subset mean for reference.
        total = sum(sample_counts)
        weighted = {
            key: sum(m[key] * n for (_, m), n in zip(subset_means, sample_counts)) / total
            for key in METRIC_KEYS
        }
        plain = {
            key: float(np.mean([m[key] for _, m in subset_means])) for key in METRIC_KEYS
        }
        summary_path = pred_root / f"{args.dataset}_evaluation_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Subset", "Samples", *METRIC_KEYS])
            for (name, m), n in zip(subset_means, sample_counts):
                writer.writerow([name, n] + [f"{m[key]:.4f}" for key in METRIC_KEYS])
            writer.writerow(
                ["WEIGHTED-AVERAGE", total] + [f"{weighted[key]:.4f}" for key in METRIC_KEYS]
            )
            writer.writerow(
                ["AVERAGE", total] + [f"{plain[key]:.4f}" for key in METRIC_KEYS]
            )
        print(f"\nSubset summary saved to: {summary_path}")
        print_summary(f"{args.dataset} weighted average (n={total})", weighted)


if __name__ == "__main__":
    main()
