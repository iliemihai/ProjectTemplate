from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, random_split

from src.datamodules.components.transform import TransformsWrapper
from src.datamodules.datamodule import SingleDataModule

class MistralDataModule(SingleDataModule):
    def __init__(self,
                 datasets: DictConfig,
                 loaders: DictConfig,
                 transforms: DictConfig) -> None:
        super().__init__(datasets=datasets, loaders=loaders, transforms=transforms)


    def prepare_data(self) -> None:
        """
        Download data if it's needed.
        Probably here are defined the steps of preprocessing steps of raw data 
        """

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.train_set and not self.valid_set and not self.test_set:
            transforms_train = TransformsWrapper(self.transforms.get("train"))
            transforms_test = TransformsWrapper(self.transforms.get("valid_test_predict"))
            train_set = pass
            test_set = pass
            dataset = ConcatDataset(datasets=[train_set, test_set])
            seed = self.cfg_datasets.get("seed")
            self.train_set, self.valid_set, self.test_set = random_slit(dataset=dataset,
                                                                        lengths=self.cfg_datasets.get("train_val_test_split"),
                                                                        generator=torch.Generator().manual_seed(seed))

            if (stage=="predict") and self.test_set:
                self.predict_set = {"PredictDataset" : self.test_set}

    def state_dict(self):
        """
        Extra things to save to checkpoint
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Things to do when loading checkpoint
        """
        pass

