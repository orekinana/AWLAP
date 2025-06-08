import os
import torch
from transformers import Trainer


class AddrFineTuneTrainer(Trainer):


    def compute_loss(self, model, inputs, return_outputs=False) -> torch.Tensor:
        outputs = model(**inputs)
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss


    def _save(self, output_dir: str | None = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
