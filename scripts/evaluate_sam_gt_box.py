import argparse
import csv
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from src.datasets.kvasir import build_dataloader
from src.datasets.transforms import denormalize_image
from src.utils.config import load_config
from src.utils.metrics import MetricTracker, dice_iou_per_sample
from src.utils.vis import save_prediction_visualization


CSV_COLUMNS = (
    "run_name", "id", "dice", "iou", "size_category", "polyp_ratio", "num_polyps"
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM ViT-B with GT boxes.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-vis", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.save_vis < 0:
        parser.error("--save-vis must be nonnegative")

    from segment_anything import SamPredictor, sam_model_registry

    cfg = load_config(args.config)
    data = cfg["data"]
    dataset = build_dataloader(args.split, cfg).dataset
    samples = (dataset[index] for index in range(len(dataset)))

    metadata = {}
    meta_path = Path(data["split_dir"]) / "metadata.csv"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8-sig", newline="") as file:
            metadata = {row["id"]: row for row in csv.DictReader(file)}

    model = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    predictor = SamPredictor(model.to(args.device).eval())

    run_name = f"sam_vit_b_gt_box_{data['image_size']}"
    if args.limit is not None:
        run_name += f"_n{args.limit}"

    output_dir = Path(cfg["eval"]["pred_dir"]) / f"{run_name}_{args.split}"
    tracker = MetricTracker()
    rows = []
    skipped = 0

    for sample in islice(samples, args.limit):
        sample_id = sample["id"]
        gt = sample["mask"]
        if not gt.any():
            skipped += 1
            continue

        image = np.ascontiguousarray(
            denormalize_image(sample["image"], mean=data["mean"], std=data["std"])
        )
        box = sample["bbox"].numpy()

        predictor.set_image(image, image_format="RGB")
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        pred = masks[0]

        # Logit +1/-1 bảo toàn mask nhị phân ở ngưỡng 0.5 của metric chung.
        pred_mask = torch.from_numpy(pred).float()[None, None]
        dice, iou = dice_iou_per_sample(2 * pred_mask - 1, gt[None], threshold=0.5)
        tracker.update(dice, iou, [sample_id])

        meta = metadata.get(sample_id, {})
        rows.append({
            "run_name": run_name,
            "id": sample_id,
            "dice": round(dice.item(), 6),
            "iou": round(iou.item(), 6),
            "size_category": meta.get("size_category", ""),
            "polyp_ratio": meta.get("polyp_ratio", ""),
            "num_polyps": meta.get("num_polyps", ""),
        })

        if len(rows) <= args.save_vis:
            save_prediction_visualization(
                image, gt, pred, output_dir / "vis" / f"{sample_id}.png", box=box
            )

    if not rows:
        raise SystemExit(f"No valid GT boxes. Empty masks skipped: {skipped}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_{args.split}_per_image.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    scores = tracker.compute()
    print(f"Split={args.split}, evaluated={len(rows)}, skipped_empty_gt={skipped}")
    print(f"Dice: {scores['dice_mean']:.4f} ± {scores['dice_std']:.4f}")
    print(f"IoU:  {scores['iou_mean']:.4f} ± {scores['iou_std']:.4f}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()