from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from asp_tools.utils.logging import get_module_logger

logger = get_module_logger(__name__)


class BaseTrainer(ABC):
    def __init__(self):
        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None

    @abstractmethod
    def _reset(self, cfg: DictConfig):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def run_one_epoch(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def run_one_epoch_train(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def run_one_epoch_eval(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, epoch: int, model_path: str):
        raise NotImplementedError
