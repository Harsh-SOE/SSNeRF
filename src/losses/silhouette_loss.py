import torch
import torch.nn.functional as F

def compute_silhouette_loss(acc_f: torch.Tensor,
                            acc_c: torch.Tensor,
                            gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Balanced silhouette BCE loss for NeRF opacity.

    acc_f / acc_c are opacity probabilities in [0, 1],
    so we use BCE on probabilities, but force fp32 because BCE is unsafe
    under autocast/mixed precision.
    """

    def balanced_bce_prob(acc: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device_type = acc.device.type

        # BCE on probabilities must run outside autocast.
        with torch.amp.autocast_mode.autocast(device_type=device_type, enabled=False):
            acc = acc.float().reshape(-1).clamp(1e-4, 1.0 - 1e-4)
            mask = mask.float().reshape(-1)

            loss = F.binary_cross_entropy(
                acc,
                mask,
                reduction="none"
            )

            fg = mask > 0.5
            bg = mask <= 0.5

            zero = acc.new_tensor(0.0)

            fg_loss = loss[fg].mean() if fg.any() else zero
            bg_loss = loss[bg].mean() if bg.any() else zero

            return fg_loss + bg_loss

    return balanced_bce_prob(acc_f, gt_mask) + 0.5 * balanced_bce_prob(acc_c, gt_mask)