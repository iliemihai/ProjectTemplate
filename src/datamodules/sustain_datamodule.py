from typing import Any, Dict, Optional

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, random_split

from src.datamodules.datamodules import SingleDataModule


class SUSTDataModule(SingleDataModule):
    def __init__(
            self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig) -> None:
        super().__init__(datasets=datasets,
                         loaders=loaders)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load and split the dataset 
        """

    def state_dict(self):
        """
        Extra things to save to checkpoint
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Things to do when loading checkpoints
        """
        pass
