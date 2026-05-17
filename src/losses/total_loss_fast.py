from typing import Dict, Tuple
import torch

from src.config.project import Config

from src.losses.silhouette_loss import compute_silhouette_loss
from src.losses.semantic_loss import compute_semantic_loss
from src.losses.distortion_fast import compute_distortion_loss_fast
from src.losses.color_loss import compute_color_loss
from src.losses.bg_supression import compute_bg_suppression_loss

def compute_total_loss_fast(preds, gts, step, config, pose_refiner):
    device = gts["rgb"].device
    bg_mask = gts["mask"].squeeze() < 0.5

    L_color = compute_color_loss(preds["rgb_f"], preds["rgb_c"], gts["rgb"])
    L_sil = compute_silhouette_loss(preds["acc_f"], preds["acc_c"], gts["mask"])
    L_dist = compute_distortion_loss_fast(preds["wf"], preds["z_f"], preds["wc"], preds["z_c"])
    L_bg = compute_bg_suppression_loss(preds["acc_f"], preds["acc_c"], bg_mask)

    if step >= config.training.sem_start_iters:
        L_sem = compute_semantic_loss(preds["sem_f"], preds["sem_c"], gts["sem"], gts["mask"])
    else:
        L_sem = torch.tensor(0.0, device=device)

    L_top_pose = (
        pose_refiner.top_view_regularization()
        if pose_refiner.top_view_exists
        and config.training.top_view_start_iter <= step < config.training.pose_start_iter
        else torch.tensor(0.0, device=device)
    )
    L_pose = pose_refiner.regularization() if step >= config.training.pose_start_iter else torch.tensor(0.0, device=device)

    t_color = min(1.0, step / config.training.color_ramp_iters)
    w_color = 0.05 + 0.95 * t_color

    t_sil = max(
        0.4,
        1.0 - 0.6 * min(1.0, step / (config.training.num_iters * config.training.sil_decay_start_pct)),
    )
    w_sil = config.training.lambda_sil * t_sil

    w_sem = 0.0
    if step >= config.training.sem_start_iters:
        t_sem = max(0.0, min(1.0, (step - config.training.sem_start_iters) / config.training.sem_ramp_iters))
        w_sem = config.training.lambda_sem * t_sem

    w_top_pose = 0.0
    if pose_refiner.top_view_exists and config.training.top_view_start_iter <= step < config.training.pose_start_iter:
        t_top = max(
            0.0,
            min(1.0, (step - config.training.top_view_start_iter) / config.training.top_view_pose_ramp_iters),
        )
        w_top_pose = config.training.lambda_pose * t_top

    w_pose = 0.0
    if step >= config.training.pose_start_iter:
        t_pose = max(0.0, min(1.0, (step - config.training.pose_start_iter) / config.training.pose_ramp_iters))
        w_pose = config.training.lambda_pose * t_pose

    total_loss = (
        (w_color * L_color)
        + (w_sil * L_sil)
        + (w_sem * L_sem)
        + (config.training.lambda_dist * L_dist)
        + (config.training.lambda_bg_acc * L_bg)
        + (w_top_pose * L_top_pose)
        + (w_pose * L_pose)
    )

    loss_dict = {
        "total": total_loss.item(),
        "color": L_color.item(),
        "silhouette": L_sil.item(),
        "semantic": L_sem.item(),
        "semantic_w": float(w_sem * L_sem.item()),
        "distortion": L_dist.item(),
        "bg_acc": L_bg.item(),
        "pose": L_pose.item(),
        "pose_w": float(w_pose * L_pose.item()),
        "top_pose": L_top_pose.item(),
        "top_pose_w": float(w_top_pose * L_top_pose.item()),
    }
    return total_loss, loss_dict