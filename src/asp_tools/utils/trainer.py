import os
import time
from abc import ABC, abstractmethod

import mlflow
import torch
import torch.nn as nn
import torchaudio
from omegaconf import DictConfig
from pydantic import BaseModel

from asp_tools.models.basetrainers import BaseTrainer
from asp_tools.utils.logging import get_module_logger

logger = get_module_logger(__name__)
BITS_PER_SAMPLE_WSJ0 = 16
MIN_PESQ = -0.5


class Artifact(BaseModel):
    best_model_path: str | None
    last_model_path: str | None


# class Trainer(BaseTrainer):
#     def __init__(self, model, loader, criterion, optimizer, cfg):
#         super().__init__(model, loader, criterion, optimizer, cfg)


class Trainer(BaseTrainer):
    def __init__(self, model, loader, criterion, optimizer, cfg: DictConfig):
        self.train_loader = loader["train"]
        self.valid_loader = loader["valid"]
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self._reset(cfg)

    def _reset(self, cfg: DictConfig):
        self.sampling_rate = cfg.jobs.data.sampling_rate
        self.n_sources = cfg.jobs.training.n_sources
        self.n_epoch = cfg.jobs.training.n_epoch
        self.max_norm = cfg.jobs.training.max_norm
        self.use_cuda = cfg.jobs.training.use_cuda
        self.best_model_path = cfg.jobs.training.best_model_path
        self.last_model_path = cfg.jobs.training.last_model_path

        self.train_loss = torch.empty(self.n_epoch)
        self.valid_loss = torch.empty(self.n_epoch)

        if os.path.exists(self.best_model_path):
            if cfg.jobs.training.overwrite:
                print("Overwrite models.")
            else:
                raise ValueError(f"{self.best_model_path} already exists.")

        self.best_loss = float("infinity")
        self.prev_loss = float("infinity")
        self.no_improvement = 0
        self.start_epoch = 0

    def run(self):
        for epoch in range(self.start_epoch, self.n_epoch):
            start = time.time()
            train_loss, valid_loss = self.run_one_epoch(epoch)
            end = time.time()

            logger.info(
                f"[Epoch {epoch+1}/{self.n_epoch}] loss (train): {train_loss}, loss (valid): {valid_loss}, time: {end - start} [sec]"
            )

            self.train_loss[epoch] = train_loss
            self.valid_loss[epoch] = valid_loss

            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("valid_loss", valid_loss)

            artifact = Artifact()
            if self.best_model_path is not None:
                artifact.best_model_path = self.best_model_path
            if self.last_model_path is not None:
                artifact.last_model_path = self.last_model_path

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.no_improvement = 0
                self.save_model(epoch=epoch, model_path=self.best_model_path)
                mlflow.log_artifact(artifact.best_model_path, "best_model")
            else:
                if valid_loss >= self.prev_loss:
                    self.no_improvement += 1
                    if self.no_improvement >= 10:
                        logger.info("Stop training")
                        break
                    if self.no_improvement >= 3:
                        for param_group in self.optimizer.param_groups:
                            prev_lr = param_group["lr"]
                            lr = 0.5 * prev_lr
                            logger.info(f"Learning rate: {prev_lr} -> {lr}")
                            param_group["lr"] = lr
                else:
                    self.no_improvement = 0

            self.prev_loss = valid_loss
            self.save_model(epoch=epoch, model_path=self.last_model_path)
            mlflow.log_artifact(artifact.last_model_path, "last_model")

    def run_one_epoch(self, epoch: int):
        return self.run_one_epoch_train(epoch), self.run_one_epoch_eval(epoch)

    def run_one_epoch_train(self, epoch: int):
        self.model.train()

        train_loss = 0
        n_train_batch = len(self.train_loader)

        for idx, mixture, sources in enumerate(self.train_loader):
            if self.use_cuda:
                mixture = mixture.cuda()
                sources = sources.cuda()

            estimated_sources = self.model(mixture)
            loss = self.criterion(estimated_sources, sources)

            self.optimizer.zero_grad()
            loss.backward()

            if self.max_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.optimizer.step()

            train_loss += loss.item()

            if (idx + 1) % 100 == 0:
                print(
                    f"[Epoch {epoch + 1}/{self.n_epoch}] iter {idx + 1}/{n_train_batch} loss: {loss.item()}"
                )

        train_loss /= n_train_batch

        return train_loss

    def run_one_epoch_eval(self, epoch: int):
        self.model.eval()

        valid_loss = 0
        n_valid = len(self.valid_loader.dataset)

        with torch.no_grad():
            for idx, (mixture, sources, segment_IDs) in enumerate(self.valid_loader):
                if self.use_cuda:
                    mixture = mixture.cuda()
                    sources = sources.cuda()
                output = self.model(mixture)
                loss = self.criterion(output, sources, batch_mean=False)
                loss = loss.sum(dim=0)
                valid_loss += loss.item()

                if idx < 5:
                    mixture = mixture[0].squeeze(dim=0).cpu()
                    estimated_sources = output[0].cpu()

                    save_dir = os.path.join(self.sample_dir, segment_IDs[0])
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, "mixture.wav")
                    norm = torch.abs(mixture).max()
                    mixture = mixture / norm
                    signal = mixture.unsqueeze(dim=0) if mixture.dim() == 1 else mixture
                    torchaudio.save(
                        save_path,
                        signal,
                        sampling_rate=self.sampling_rate,
                        bits_per_sample=BITS_PER_SAMPLE_WSJ0,
                    )

                    for source_idx, estimated_source in enumerate(estimated_sources):
                        save_path = os.path.join(
                            save_dir, "epoch{}-{}.wav".format(epoch + 1, source_idx + 1)
                        )
                        norm = torch.abs(estimated_source).max()
                        estimated_source = estimated_source / norm
                        signal = (
                            estimated_source.unsqueeze(dim=0)
                            if estimated_source.dim() == 1
                            else estimated_source
                        )
                        torchaudio.save(
                            save_path,
                            signal,
                            sampling_rate=self.sampling_rate,
                            bits_per_sample=BITS_PER_SAMPLE_WSJ0,
                        )

        valid_loss /= n_valid

        return valid_loss

    def save_model(self, epoch: int, model_path: str = "./tmp.pth"):
        if isinstance(self.model, nn.DataParallel):
            config = self.model.module.get_config()
            config["state_dict"] = self.model.module.state_dict()
        else:
            config = self.model.get_config()
            config["state_dict"] = self.model.state_dict()

        config["optim_dict"] = self.optimizer.state_dict()

        # config["best_loss"] = self.best_loss
        config["no_improvement"] = self.no_improvement

        config["train_loss"] = self.train_loss
        config["valid_loss"] = self.valid_loss

        config["epoch"] = epoch + 1

        torch.save(config, model_path)
