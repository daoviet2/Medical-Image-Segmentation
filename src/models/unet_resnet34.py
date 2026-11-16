"""U-Net với Pretrained Encoder ResNet34 cho phân đoạn tổn thương polyp Kvasir-SEG.

Sử dụng thư viện segmentation_models_pytorch (SMP):
- Encoder: ResNet34 nạp sẵn trọng số tiền huấn luyện ImageNet (Transfer Learning).
  Các bộ lọc tầng nông (edge, texture, color) được bảo toàn từ 1.2M ảnh ImageNet,
  giúp mô hình hội tụ nhanh hơn và giảm overfit trên tập dữ liệu nhỏ Kvasir-SEG (800 ảnh train).
- Decoder: U-Net decoder 5 bậc (decoder_channels=[256, 128, 64, 32, 16]) với skip connections
  từ các tầng tương ứng của ResNet34.
- Tầng phân đoạn (Segmentation Head): Conv2d 1x1 đưa về 1 kênh raw logit [B, 1, H, W] (chưa qua sigmoid).
- Tổng số tham số: ~24,436,369 (~24.44M tham số), nhẹ hơn và hiệu quả hơn so với U-Net scratch (31.04M tham số).

Tuân thủ hợp đồng giao diện của nhóm:
    model(image: Tensor[B, 3, H, W]) -> logits: Tensor[B, 1, H, W] (raw logits cùng kích thước ảnh vào)
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from . import register_model


class UNetResNet34(nn.Module):
    """Mô hình U-Net sử dụng encoder ResNet34 pretrained từ ImageNet.

    Args:
        encoder_name (str): Tên encoder ResNet (mặc định: 'resnet34').
        encoder_weights (Optional[str]): Trọng số tiền huấn luyện ('imagenet' hoặc None).
        in_channels (int): Số kênh ảnh đầu vào (mặc định: 3 cho RGB).
        num_classes (int): Số lớp phân đoạn (mặc định: 1 cho nhị phân polyp/nền).
        encoder_depth (int): Độ sâu encoder (mặc định: 5 bậc downsampling).
        decoder_channels (Sequence[int]): Số kênh đặc trưng của các tầng decoder (mặc định: [256, 128, 64, 32, 16]).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_depth: int = 5,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Xử lý trường hợp chuỗi "none" hoặc "null" từ config YAML
        if isinstance(encoder_weights, str) and encoder_weights.lower() in ("none", "null"):
            encoder_weights = None

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=list(decoder_channels),
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # Xuất raw logits (chưa qua sigmoid) theo đúng hợp đồng nhóm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass).

        Args:
            x (torch.Tensor): Tensor ảnh đầu vào [B, 3, H, W].

        Returns:
            torch.Tensor: Tensor raw logits [B, 1, H, W] cùng kích thước không gian (H, W).
        """
        return self.model(x)

    def get_encoder_params(self) -> List[nn.Parameter]:
        """Lấy danh sách tham số của encoder (dùng khi muốn chia differential learning rate)."""
        return list(self.model.encoder.parameters())

    def get_decoder_params(self) -> List[nn.Parameter]:
        """Lấy danh sách tham số của decoder và segmentation head."""
        params = list(self.model.decoder.parameters())
        params.extend(list(self.model.segmentation_head.parameters()))
        return params


@register_model("unet_resnet34")
def build_unet_resnet34(model_cfg: Dict[str, Any]) -> nn.Module:
    """Khởi tạo mô hình UNetResNet34 từ cấu hình config['model'].

    Args:
        model_cfg (dict): Cấu hình con 'model' từ file cấu hình YAML.

    Returns:
        UNetResNet34: Mô hình PyTorch đã sẵn sàng huấn luyện hoặc đánh giá.
    """
    encoder_name = model_cfg.get("encoder_name", "resnet34")
    encoder_weights = model_cfg.get("encoder_weights", "imagenet")
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("num_classes", 1)
    encoder_depth = model_cfg.get("encoder_depth", 5)
    decoder_channels = model_cfg.get("decoder_channels", (256, 128, 64, 32, 16))

    # Nếu cờ encoder_pretrained là False thì bỏ weights
    if not model_cfg.get("encoder_pretrained", True):
        encoder_weights = None

    return UNetResNet34(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
    )
