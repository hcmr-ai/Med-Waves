from lightning.pytorch.callbacks import BaseFinetuning

class FreezeEncoderCallback(BaseFinetuning):
    def __init__(self, aggressive_freeze=False):
        super().__init__()
        self.aggressive_freeze = aggressive_freeze

    def freeze_before_training(self, pl_module):
        # Freeze first 1-2 encoder blocks (learn low-level features like edges)
        if self.aggressive_freeze:
            for enc in pl_module.model.encoders:
                for param in enc.parameters():
                    param.requires_grad = False
        else:
            self.freeze(pl_module.model.encoders[0])
            self.freeze(pl_module.model.encoders[1])

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == 5:
            # Unfreeze encoder blocks after epoch 5
            self.make_trainable(pl_module.model.encoders[0])
            self.make_trainable(pl_module.model.encoders[1])
