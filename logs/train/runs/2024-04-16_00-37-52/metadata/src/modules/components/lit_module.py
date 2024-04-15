from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from peft import LoraConfig, get_peft_model, TaskType

def task_type_converted(task):
    if task == 1:
        return TaskType.SEQ_CLS

class BaseLitModule(LightningModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        peft: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """BaseLightningModule.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(*args, **kwargs)
        self.peft_params = peft

        if self.peft_params.get("train_with_peft"):
            # Setup LoRA configuration for PEFT
            peft_config = LoraConfig(
                task_type=task_type_converted(self.peft_params.get("task_type")),#TaskType.SEQ_CLS
                inference_mode=self.peft_params.get("inference_mode"),#False
                r=self.peft_params.get("r"),#8
                lora_alpha=self.peft_params.get("lora_alpha"),#32
                lora_dropout=self.peft_params.get("lora_dropout"),#0.1
            )
        self.model = hydra.utils.instantiate(network.model)
        if self.peft_params.get("train_with_peft"):
            self.model = get_peft_model(self.model, peft_config)
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.logging_params = logging

    def forward(self, x: Any) -> Any:
        return self.model.forward(x)

    def configure_optimizers(self) -> Any:
        optimizer: torch.optim = hydra.utils.instantiate(
            self.opt_params, params=self.parameters(), _convert_="partial"
        )
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}
