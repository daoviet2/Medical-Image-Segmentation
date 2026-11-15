import math

import torch

from src.losses import build_loss


def main():
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    expected = {
        "dice": 0.5,
        "focal": 5 * math.log(2) / 32,
        "tversky": 1 - 1.5 / 2.3,
        "bce_dice": math.log(2) + 0.5,
    }

    for name, value in expected.items():
        criterion = build_loss({"train": {"loss": name}})
        logits = torch.zeros_like(target, requires_grad=True)

        # Logits [B, 1, H, W], target [B, H, W].
        loss = criterion(logits, target[:, 0])
        torch.testing.assert_close(loss, torch.tensor(value))
        loss.backward()
        assert torch.isfinite(logits.grad).all(), name
        assert logits.grad.abs().sum() > 0, name

        good_logits = (2 * target - 1) * 20
        assert criterion(good_logits, target) < criterion(-good_logits, target), name

        # Kiểm tra logit cực trị với cả GT có polyp và GT rỗng.
        for mask in (target, torch.zeros_like(target)):
            extreme = torch.full_like(mask, -100, requires_grad=True)
            extreme_loss = criterion(extreme, mask)
            assert torch.isfinite(extreme_loss), name
            extreme_loss.backward()
            assert torch.isfinite(extreme.grad).all(), name

        print(f"{name}: {loss.item():.6f} OK")


if __name__ == "__main__":
    main()