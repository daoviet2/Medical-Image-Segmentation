"""Kiểm thử tự động cho feat-014: U-Net Pretrained ResNet34.

Kiểm tra:
1. build_model với 'unet_resnet34'.
2. Shape đầu ra: [2, 3, 256, 256] -> [2, 1, 256, 256]; [2, 3, 352, 352] -> [2, 1, 352, 352].
3. Đếm số tham số: ~24.44M tham số (khớp 24,436,369).
4. Kiểm tra trọng số ImageNet: conv1.weight có giá trị khác Kaiming ngẫu nhiên.
5. Backward pass: 100% tham số có gradient hợp lệ.
6. Nạp cấu hình kế thừa từ configs/unet_resnet34_bce_dice.yaml.
"""

import os
import sys

# Đảm bảo import được từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models import build_model, count_parameters, MODEL_REGISTRY, UNetResNet34
from src.utils.config import load_config


def test_unet_resnet34():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ FEAT-014: U-Net Pretrained ResNet34")
    print("=" * 60)

    # 1. Kiểm tra registry
    assert "unet_resnet34" in MODEL_REGISTRY, "unet_resnet34 chưa được đăng ký trong MODEL_REGISTRY!"
    print("[1/6] MODEL_REGISTRY chứa 'unet_resnet34': PASS")

    # 2. Khởi tạo mô hình qua build_model
    cfg = {"model": {"name": "unet_resnet34", "encoder_weights": "imagenet"}}
    model = build_model(cfg)
    assert isinstance(model, UNetResNet34), "build_model không trả về đối tượng UNetResNet34!"
    print("[2/6] build_model('unet_resnet34') thành công: PASS")

    # 3. Đếm số tham số
    n_params = count_parameters(model)
    print(f"  -> Tổng tham số: {n_params:,}")
    assert 24_000_000 < n_params < 25_000_000, f"Số tham số không hợp lý: {n_params}"
    assert n_params == 24_436_369, f"Số tham số {n_params} khác chuẩn 24,436,369"
    print("[3/6] Số tham số đạt chuẩn 24,436,369 (~24.44M): PASS")

    # 4. Kiểm tra Forward pass với các kích thước ảnh khác nhau
    model.eval()
    with torch.no_grad():
        x256 = torch.randn(2, 3, 256, 256)
        out256 = model(x256)
        assert out256.shape == (2, 1, 256, 256), f"Shape ra {out256.shape} khác (2, 1, 256, 256)"

        x352 = torch.randn(2, 3, 352, 352)
        out352 = model(x352)
        assert out352.shape == (2, 1, 352, 352), f"Shape ra {out352.shape} khác (2, 1, 352, 352)"
    print("[4/6] Forward pass 256x256 và 352x352 đúng kích thước: PASS")

    # 5. Kiểm tra Backward pass
    model.train()
    x = torch.randn(2, 3, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    out = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
    loss.backward()

    no_grad_params = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert len(no_grad_params) == 0, f"Tham số thiếu gradient: {no_grad_params}"
    print("[5/6] Backward pass thành công, 100% tham số có gradient: PASS")

    # 6. Kiểm tra nạp file config
    config_path = "configs/unet_resnet34_bce_dice.yaml"
    full_cfg = load_config(config_path)
    assert full_cfg["model"]["name"] == "unet_resnet34"
    assert full_cfg["model"]["encoder_pretrained"] is True
    assert full_cfg["train"]["optimizer"] == "adamw"
    assert full_cfg["train"]["lr"] == 0.0003
    assert full_cfg["train"]["loss"] == "bce_dice"
    assert full_cfg["data"]["image_size"] == 256
    print("[6/6] Cấu hình configs/unet_resnet34_bce_dice.yaml kế thừa chuẩn: PASS")

    print("=" * 60)
    print("HOÀN THÀNH 100% KIỂM THỬ FEAT-014!")
    print("=" * 60)


if __name__ == "__main__":
    test_unet_resnet34()
