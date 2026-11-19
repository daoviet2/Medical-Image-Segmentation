"""SegFormer (Xie et al., NeurIPS 2021) wrapper cho phân đoạn polyp Kvasir-SEG.

Đặc điểm kiến trúc:
- Hierarchical Transformer Encoder (MiT-B0 hoặc MiT-B2) tạo đặc trưng đa tỷ lệ.
- Lightweight all-MLP Decoder tổng hợp đặc trưng từ 4 tầng encoder.
- Do MLP Decoder trả về logits tại kích thước (H/4, W/4), wrapper tự động
  nội suy song tuyến (bilinear interpolation) đưa logits về (H, W) của ảnh gốc.
- Đầu ra là raw logits [B, num_classes, H, W] (chưa qua sigmoid) để tương thích
  hoàn hảo với hàm mất mát BCEWithLogits và SoftDice trong pipeline chung của nhóm.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from . import register_model


class SegFormerSegmentation(nn.Module):
    """SegFormer wrapper tuân thủ giao diện chung của dự án:
        model(image: Tensor[B, 3, H, W]) -> logits: Tensor[B, num_classes, H, W]
    """

    def __init__(
        self,
        model_name: str = "nvidia/mit-b0",
        num_classes: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        if pretrained:
            # ignore_mismatched_sizes=True cho phép khởi tạo decode head với số lớp num_classes=1
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            try:
                config = SegformerConfig.from_pretrained(model_name, num_labels=num_classes)
            except Exception:
                config = SegformerConfig(num_labels=num_classes)
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass qua SegFormer và upsample logits về kích thước gốc.

        Args:
            x (torch.Tensor): Tensor ảnh đầu vào [B, 3, H, W] đã chuẩn hóa ImageNet.

        Returns:
            torch.Tensor: Raw logits [B, num_classes, H, W] (chưa qua sigmoid).
        """
        h, w = x.shape[2], x.shape[3]
        # SegFormer nhận pixel_values [B, 3, H, W]
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # [B, num_classes, H/4, W/4]

        # Upsample bilinear về đúng kích thước ảnh đầu vào [B, num_classes, H, W]
        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        return logits


@register_model("segformer_b0")
def _build_segformer_b0(model_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo kiến trúc SegFormer MiT-B0 (~3.7M tham số)."""
    pretrained = model_cfg.get("encoder_pretrained", True)
    pretrained_name = model_cfg.get("pretrained_name", "nvidia/mit-b0")
    num_classes = model_cfg.get("num_classes", 1)
    return SegFormerSegmentation(
        model_name=pretrained_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


@register_model("segformer_b2")
def _build_segformer_b2(model_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo kiến trúc SegFormer MiT-B2 (~24.7M tham số)."""
    pretrained = model_cfg.get("encoder_pretrained", True)
    pretrained_name = model_cfg.get("pretrained_name", "nvidia/mit-b2")
    num_classes = model_cfg.get("num_classes", 1)
    return SegFormerSegmentation(
        model_name=pretrained_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


__all__ = ["SegFormerSegmentation"]
