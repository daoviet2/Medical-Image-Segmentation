"""Script tự động đọc và tổng hợp kết quả thí nghiệm qua các seed.

Chức năng:
- Đọc `outputs/results.csv` (danh sách các lần chạy với 20 cột chuẩn).
- Nhóm theo kiến trúc mô hình và hàm loss `(model, loss, encoder_pretrained)`.
- Tính trung bình và độ lệch chuẩn across-seeds:
  + Val Macro Dice (Mean ± Std)
  + Val Macro IoU (Mean ± Std)
  + Thời gian huấn luyện (phút) và Tốc độ suy luận (ms/ảnh, FPS)
  + Epoch hội tụ tối ưu (best_epoch)
- Đọc các file `outputs/predictions/*_val_per_image.csv` để tổng hợp hiệu năng phân tầng
  theo kích thước polyp (small: <5%, medium: 5%-20%, large: >=20%).
- Xuất bảng định dạng Markdown (chuẩn bị đưa vào báo cáo và slide) và file CSV tổng hợp.
"""

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
    """Tính trung bình (mean) và độ lệch chuẩn mẫu (sample std, ddof=1)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n <= 1:
        return mean_val, 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_val = math.sqrt(variance)
    return mean_val, std_val


def load_results_csv(path: str) -> List[Dict[str, str]]:
    """Đọc toàn bộ các dòng từ file results.csv."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("run_name"):
                rows.append(row)
    return rows


def group_results_by_model(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """Nhóm các lần chạy theo (model, loss, encoder_pretrained, image_size)."""
    grouped = defaultdict(lambda: {
        "seeds": [],
        "run_names": [],
        "dice_scores": [],
        "iou_scores": [],
        "train_times": [],
        "infer_times": [],
        "best_epochs": [],
        "epochs_run": [],
        "n_params": None,
        "model": None,
        "loss": None,
        "encoder_pretrained": None,
        "image_size": None,
        "wandb_urls": [],
    })

    for r in rows:
        model = r.get("model", "unknown")
        loss = r.get("loss", "bce_dice")
        pretrained = r.get("encoder_pretrained", "False")
        img_size = r.get("image_size", "256")
        group_key = f"{model}_{loss}_pre{pretrained}_{img_size}"

        g = grouped[group_key]
        g["model"] = model
        g["loss"] = loss
        g["encoder_pretrained"] = pretrained
        g["image_size"] = img_size
        if g["n_params"] is None and r.get("n_params"):
            try:
                g["n_params"] = int(r["n_params"])
            except ValueError:
                g["n_params"] = r["n_params"]

        g["run_names"].append(r.get("run_name", ""))
        if r.get("seed"):
            g["seeds"].append(r["seed"])
        if r.get("dice_mean"):
            g["dice_scores"].append(float(r["dice_mean"]))
        if r.get("iou_mean"):
            g["iou_scores"].append(float(r["iou_mean"]))
        if r.get("train_time_min"):
            g["train_times"].append(float(r["train_time_min"]))
        if r.get("infer_ms_per_img"):
            g["infer_times"].append(float(r["infer_ms_per_img"]))
        if r.get("best_epoch"):
            g["best_epochs"].append(float(r["best_epoch"]))
        if r.get("epochs_run"):
            g["epochs_run"].append(float(r["epochs_run"]))
        if r.get("wandb_url"):
            g["wandb_urls"].append(r["wandb_url"])

    return dict(grouped)


def load_stratified_per_image(pred_dir: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Đọc tất cả các file per-image CSV và phân nhóm theo mô hình và kích thước polyp.
    
    Trả về cấu trúc:
    {
        model_name: {
            'small': {'dice': [...], 'iou': [...]},
            'medium': {'dice': [...], 'iou': [...]},
            'large': {'dice': [...], 'iou': [...]}
        }
    }
    """
    if not os.path.isdir(pred_dir):
        return {}

    stratified = defaultdict(lambda: {
        "small": {"dice": [], "iou": []},
        "medium": {"dice": [], "iou": []},
        "large": {"dice": [], "iou": []},
    })

    csv_files = glob.glob(os.path.join(pred_dir, "*_val_per_image.csv"))
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Suy ra loại mô hình từ tiền tố file name
        if "unet_scratch" in filename:
            m_key = "unet_scratch"
        elif "unet_resnet34" in filename:
            m_key = "unet_resnet34"
        elif "segformer" in filename:
            m_key = "segformer"
        elif "sam" in filename:
            m_key = "sam"
        else:
            m_key = filename.split("_val_per_image")[0]

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cat = row.get("size_category", "").lower().strip()
                if cat in ("small", "medium", "large"):
                    try:
                        d = float(row.get("dice", 0.0))
                        iou = float(row.get("iou", 0.0))
                        stratified[m_key][cat]["dice"].append(d)
                        stratified[m_key][cat]["iou"].append(iou)
                    except ValueError:
                        continue

    return dict(stratified)


def format_table_1(grouped: Dict[str, Dict[str, Any]]) -> str:
    """Tạo Bảng 1: Kết quả chính (Mean ± Std qua các seed)."""
    headers = [
        "Mô hình",
        "Pretrained",
        "Loss",
        "Số Seed",
        "Tham số",
        "Val Macro Dice (Mean ± Std)",
        "Val Macro IoU (Mean ± Std)",
        "Huấn luyện (phút)",
        "Suy luận (ms/ảnh | FPS)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---", ":---:", ":---:", ":---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]

    for key, data in grouped.items():
        n_seeds = len(data["seeds"])
        dice_m, dice_s = calculate_mean_std(data["dice_scores"])
        iou_m, iou_s = calculate_mean_std(data["iou_scores"])
        time_m, _ = calculate_mean_std(data["train_times"])
        infer_m, _ = calculate_mean_std(data["infer_times"])
        fps = round(1000.0 / infer_m, 1) if infer_m > 0 else 0.0

        n_params_str = f"{data['n_params']:,}" if isinstance(data["n_params"], int) else str(data["n_params"])
        pretrained_badge = "✅ Có (ImageNet)" if str(data["encoder_pretrained"]).lower() == "true" else "❌ Không (Scratch)"

        if n_seeds > 1:
            dice_str = f"**{dice_m:.4f} ± {dice_s:.4f}**"
            iou_str = f"**{iou_m:.4f} ± {iou_s:.4f}**"
        else:
            dice_str = f"**{dice_m:.4f}** (s{data['seeds'][0]})" if n_seeds == 1 else "Chưa có"
            iou_str = f"**{iou_m:.4f}** (s{data['seeds'][0]})" if n_seeds == 1 else "Chưa có"

        row = [
            f"`{data['model']}`",
            pretrained_badge,
            f"`{data['loss']}`",
            f"{n_seeds} ({', '.join(map(str, data['seeds']))})",
            n_params_str,
            dice_str,
            iou_str,
            f"{time_m:.1f} min",
            f"{infer_m:.1f} ms (~{fps} FPS)",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_table_2(stratified: Dict[str, Dict[str, Dict[str, List[float]]]]) -> str:
    """Tạo Bảng 2: Phân tầng theo kích thước polyp (Small, Medium, Large)."""
    headers = [
        "Mô hình",
        "Small Dice (<5%)",
        "Medium Dice (5-20%)",
        "Large Dice (≥20%)",
        "Small IoU (<5%)",
        "Medium IoU (5-20%)",
        "Large IoU (≥20%)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---", "---:", "---:", "---:", "---:", "---:", "---:"]) + " |",
    ]

    for model_name, cats in stratified.items():
        s_d_m, s_d_s = calculate_mean_std(cats["small"]["dice"])
        m_d_m, m_d_s = calculate_mean_std(cats["medium"]["dice"])
        l_d_m, l_d_s = calculate_mean_std(cats["large"]["dice"])

        s_i_m, s_i_s = calculate_mean_std(cats["small"]["iou"])
        m_i_m, m_i_s = calculate_mean_std(cats["medium"]["iou"])
        l_i_m, l_i_s = calculate_mean_std(cats["large"]["iou"])

        row = [
            f"`{model_name}`",
            f"{s_d_m:.4f} ± {s_d_s:.4f} (n={len(cats['small']['dice'])})",
            f"{m_d_m:.4f} ± {m_d_s:.4f} (n={len(cats['medium']['dice'])})",
            f"{l_d_m:.4f} ± {l_d_s:.4f} (n={len(cats['large']['dice'])})",
            f"{s_i_m:.4f} ± {s_i_s:.4f}",
            f"{m_i_m:.4f} ± {m_i_s:.4f}",
            f"{l_i_m:.4f} ± {l_i_s:.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_table_3(grouped: Dict[str, Dict[str, Any]]) -> str:
    """Tạo Bảng 3: Chi tiết tốc độ hội tụ (Best Epoch vs Epochs Run)."""
    headers = [
        "Mô hình",
        "Pretrained",
        "Best Epoch (Mean ± Std)",
        "Tổng Epoch Huấn Luyện",
        "Link Weights & Biases",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---", ":---:", "---:", "---:", ":---"]) + " |",
    ]

    for key, data in grouped.items():
        best_m, best_s = calculate_mean_std(data["best_epochs"])
        epochs_m, _ = calculate_mean_std(data["epochs_run"])
        pretrained_badge = "Có" if str(data["encoder_pretrained"]).lower() == "true" else "Không"
        wb_links = ", ".join([f"[Run]({url})" for url in data["wandb_urls"] if url]) or "N/A"

        best_str = f"Epoch {best_m:.1f} ± {best_s:.1f}" if len(data["best_epochs"]) > 1 else f"Epoch {int(best_m)}"
        row = [
            f"`{data['model']}`",
            pretrained_badge,
            best_str,
            f"{epochs_m:.0f} epochs",
            wb_links,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def export_aggregate_csv(grouped: Dict[str, Dict[str, Any]], output_path: str):
    """Xuất bảng tổng hợp ra file CSV phẳng."""
    fieldnames = [
        "model",
        "loss",
        "encoder_pretrained",
        "image_size",
        "n_seeds",
        "seeds",
        "n_params",
        "dice_mean",
        "dice_std",
        "iou_mean",
        "iou_std",
        "train_time_min_mean",
        "infer_ms_per_img_mean",
        "best_epoch_mean",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, data in grouped.items():
            dice_m, dice_s = calculate_mean_std(data["dice_scores"])
            iou_m, iou_s = calculate_mean_std(data["iou_scores"])
            time_m, _ = calculate_mean_std(data["train_times"])
            infer_m, _ = calculate_mean_std(data["infer_times"])
            best_m, _ = calculate_mean_std(data["best_epochs"])

            writer.writerow({
                "model": data["model"],
                "loss": data["loss"],
                "encoder_pretrained": data["encoder_pretrained"],
                "image_size": data["image_size"],
                "n_seeds": len(data["seeds"]),
                "seeds": ";".join(map(str, data["seeds"])),
                "n_params": data["n_params"],
                "dice_mean": round(dice_m, 4),
                "dice_std": round(dice_s, 4),
                "iou_mean": round(iou_m, 4),
                "iou_std": round(iou_s, 4),
                "train_time_min_mean": round(time_m, 2),
                "infer_ms_per_img_mean": round(infer_m, 2),
                "best_epoch_mean": round(best_m, 1),
            })
    print(f"[Xuất file] Đã lưu CSV tổng hợp vào: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp kết quả thí nghiệm qua các seed.")
    parser.add_argument("--results-csv", type=str, default="outputs/results.csv", help="Đường dẫn outputs/results.csv")
    parser.add_argument("--pred-dir", type=str, default="outputs/predictions", help="Thư mục chứa per-image CSV")
    parser.add_argument("--output-md", type=str, default="outputs/aggregate_report.md", help="Đường dẫn xuất file Markdown")
    parser.add_argument("--output-csv", type=str, default="outputs/aggregate_summary.csv", help="Đường dẫn xuất file CSV")

    args = parser.parse_args()

    # 1. Đọc dữ liệu
    rows = load_results_csv(args.results_csv)
    if not rows:
        print(f"[Cảnh báo] File {args.results_csv} rỗng hoặc không tồn tại.")
        return

    grouped = group_results_by_model(rows)
    stratified = load_stratified_per_image(args.pred_dir)

    # 2. Xây dựng các bảng
    table1 = format_table_1(grouped)
    table2 = format_table_2(stratified) if stratified else "_(Chưa có file per-image CSV nào trong outputs/predictions)_"
    table3 = format_table_3(grouped)

    report_content = f"""# Báo Cáo Tổng Hợp Kết Quả Thí Nghiệm (Across Seeds)

### 1. Bảng Hiệu Năng Chính (Val Macro Dice & IoU - Mean ± Std)
{table1}

### 2. Phân Tầng Hiệu Năng Theo Kích Thước Polyp (Validation Set)
{table2}

### 3. Tốc Độ Hội Tụ & Đường Link Weights & Biases
{table3}
"""

    print("=" * 80)
    print(report_content)
    print("=" * 80)

    # 3. Ghi file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Xuất file] Đã lưu Báo cáo Markdown vào: {os.path.abspath(args.output_md)}")

    export_aggregate_csv(grouped, args.output_csv)


if __name__ == "__main__":
    main()
