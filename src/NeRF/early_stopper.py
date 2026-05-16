from src.config.early_stopping import EarlyStopConfig

class EarlyStopper:
    """
    Tracks convergence metrics and determines if training should stop.
    Encapsulates the state tracking so the training loop remains clean.
    """
    def __init__(self, config: EarlyStopConfig):
        self.config = config
        self.stop_condition_met_at = None
        self.has_stopped = False

    def step(self, current_step: int, plant_acc: float, bg_acc: float, psnr: float) -> bool:
        """
        Evaluates current metrics against the stopping criteria.
        Returns True if training should be halted.
        """
        if current_step < self.config.min_iters:
            return False

        # Check if all convergence criteria are met
        ok = (plant_acc >= self.config.plant_acc and
              bg_acc <= self.config.bg_acc and
              psnr >= self.config.psnr_min)

        if ok:
            if self.stop_condition_met_at is None:
                # Start the patience timer
                self.stop_condition_met_at = current_step
            elif current_step - self.stop_condition_met_at >= self.config.patience:
                # Patience exceeded while criteria remained met
                self.has_stopped = True
                return True
        else:
            # Criteria broken, reset the patience timer
            self.stop_condition_met_at = None

        return False