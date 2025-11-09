from lightning.pytorch.callbacks import Callback

class EMAWeightAveraging(Callback):
    def __init__(self, decay=0.999, start_step=100):
        super().__init__()
        self.decay = decay
        self.start_step = start_step
        self.ema_weights = None
        self.global_step = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.global_step += 1
        if self.global_step < self.start_step:
            return
                
        # Initialize or update EMA weights
        if self.ema_weights is None:
            self.ema_weights = [p.detach().clone() for p in pl_module.parameters() if p.requires_grad]
        else:
            for ema_w, model_w in zip(self.ema_weights, pl_module.parameters()):
                ema_w.mul_(self.decay)
                ema_w.add_((1.0 - self.decay) * model_w.detach())

    def on_validation_start(self, trainer, pl_module):
        if self.ema_weights is not None:
            # Store current model weights and switch to EMA
            self.model_weights = [p.detach().clone() for p in pl_module.parameters()]
            for model_w, ema_w in zip(pl_module.parameters(), self.ema_weights):
                model_w.data.copy_(ema_w.data)

    def on_validation_end(self, trainer, pl_module):
        print(f"[EMA Validation End] at step={self.global_step}")
        if hasattr(self, "model_weights"):
            # Restore original weights
            for model_w, old_w in zip(pl_module.parameters(), self.model_weights):
                model_w.data.copy_(old_w.data)
            del self.model_weights

    # ✅ Lightning calls these for checkpointing
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {
            "ema_weights": [w.cpu() for w in self.ema_weights] if self.ema_weights else None,
            "global_step": self.global_step,
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_weights" in checkpoint and checkpoint["ema_weights"] is not None:
            self.ema_weights = [w.to(pl_module.device) for w in checkpoint["ema_weights"]]
        self.global_step = checkpoint.get("global_step", 0)
    
    def state_dict(self):
        print(f"[EMA State Dict] at step={self.global_step}")
        return {
            "ema_weights": self.ema_weights,
            "global_step": self.global_step,
        }

    # ✅ REQUIRED for restoring EMA after Resume
    def load_state_dict(self, state_dict):
        print(f"[EMA Load State Dict] at step={self.global_step}")
        self.ema_weights = state_dict.get("ema_weights")
        self.global_step = state_dict.get("global_step", 0)
