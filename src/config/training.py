from dataclasses import dataclass

@dataclass
class TrainingConfig:
    near: float = 1.5
    far: float = 4.0

    n_samples_coarse: int = 48
    n_samples_fine: int = 128

    batch_rays: int = 4096
    num_iters: int = 10_000

    lr: float = 3e-4
    pose_lr: float = 1e-3

    use_amp: bool = True

    lambda_sil: float = 1.0
    lambda_sem: float = 0.1
    lambda_pose: float = 0.01
    lambda_bg_acc: float = 1.0
    lambda_dist: float = 0.02

    color_ramp_iters: int = 3000

    sil_decay_start_pct: float = 0.7

    enable_sem: bool = True
    sem_start_iters: int = 7500
    sem_ramp_iters: int = 1000

    enable_top_view_refinement: bool = True
    top_view_start_iter: int = 3750
    top_view_pose_ramp_iters: int = 500

    enable_pose_refinement: bool = True
    pose_start_iter: int = 3750
    pose_ramp_iters: int = 1000

    log_every: int = 200
    save_every: int = 500