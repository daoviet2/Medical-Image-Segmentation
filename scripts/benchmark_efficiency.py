"""Script đo lường và so sánh chi phí tính toán (Parameters, Latency, FPS, GPU Memory).

Phục vụ nhiệm vụ của TV2:
- So sánh U-Net (scratch), U-Net (ResNet34), SegFormer-B0, SegFormer-B2.
- Đo thời gian suy luận (latency ms/ảnh) và thông lượng (FPS).
- Xuất bảng Markdown / CSV tổng hợp để đưa vào báo cáo và slide thuyết trình.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import torch
import torch.nn as nn

from src.models import build_model, count_parameters
from src.utils.seed import seed_everything


def benchmark_model(
    model: nn.Module,
    device: torch.device,
    image_size: int = 256,
    batch_size: int = 1,
    warmup_runs: int = 10,
    test_runs: int = 50,
) -> dict:
    model.eval()
    dummy_input = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Đo số tham số
    n_params = count_parameters(model)
    n_params_m = round(n_params / 1e6, 2)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Reset max memory tracker
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Benchmark latency
    times = []
    with torch.no_grad():
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms cho cả batch

    avg_latency_batch_ms = sum(times) / len(times)
    latency_per_img_ms = avg_latency_batch_ms / batch_size
    fps = 1000.0 / latency_per_img_ms if latency_per_img_ms > 0 else 0.0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    return {
        "params_m": n_params_m,
        "n_params": n_params,
        "latency_ms": round(latency_per_img_ms, 2),
        "fps": round(fps, 1),
        "peak_mem_mb": round(peak_mem_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark hiệu năng tính toán CNN vs Transformer.")
    parser.add_argument("--device", type=str, default=None, help="'cuda' hoặc 'cpu'")
    parser.add_argument("--image_size", type=int, default=256, help="Kích thước ảnh vuông")
    parser.add_argument("--output_csv", type=str, default="outputs/efficiency_benchmark.csv")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Benchmark trên thiết bị: {device} | Kích thước ảnh: {args.image_size}x{args.image_size}")

    models_to_test = [
        {"name": "unet_scratch", "type": "CNN", "cfg": {"model": {"name": "unet_scratch"}}},
        {"name": "segformer_b0", "type": "Transformer", "cfg": {"model": {"name": "segformer_b0", "encoder_pretrained": False}}},
        {"name": "segformer_b2", "type": "Transformer", "cfg": {"model": {"name": "segformer_b2", "encoder_pretrained": False}}},
    ]

    results = []
    for item in models_to_test:
        model_name = item["name"]
        print(f"Đang benchmark mô hình: {model_name} ...")
        try:
            model = build_model(item["cfg"]).to(device)
            res = benchmark_model(model, device=device, image_size=args.image_size)
            results.append({
                "Mô hình": model_name,
                "Kiến trúc": item["type"],
                "Tham số (M)": res["params_m"],
                "Độ trễ (ms/ảnh)": res["latency_ms"],
                "Thông lượng (FPS)": res["fps"],
                "VRAM đỉnh (MB)": res["peak_mem_mb"],
            })
            print(f"  -> {model_name}: {res['params_m']}M params | {res['latency_ms']} ms | {res['fps']} FPS")
        except Exception as e:
            print(f"  -> Lỗi khi benchmark {model_name}: {e}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP CHI PHÍ TÍNH TOÁN (BENCHMARK EFFICIENCY)")
    print("=" * 70)
    print(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\n[OK] Đã lưu bảng kết quả vào: {args.output_csv}")


if __name__ == "__main__":
    main()
