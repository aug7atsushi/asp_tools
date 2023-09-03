import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from asp_tools.criterions.pit import PIT1d
from asp_tools.criterions.sdr import NegSISDR
from asp_tools.datasets.dataloader import test_collate_fn
from asp_tools.datasets.dataset import WaveTrainDataset, WaveValidDataset
from asp_tools.models.conv_tasnet import ConvTasNet
from asp_tools.utils.logging import get_module_logger
from asp_tools.utils.trainer import Trainer

logger = get_module_logger(__name__)


@hydra.main(
    version_base=None, config_path="../../settings/", config_name="conv_tasnet.yaml"
)
def main(cfg: DictConfig) -> None:
    train_dataset = WaveTrainDataset(
        wav_root_path=cfg.jobs.data.wav_root_path,
        json_path=cfg.jobs.data.json_train_path,
    )
    valid_dataset = WaveValidDataset(
        wav_root_path=cfg.jobs.data.wav_root_path,
        json_path=cfg.jobs.data.json_valid_path,
    )

    logger.info(f"Training dataset includes {len(train_dataset)} samples.")
    logger.info(f"Valid dataset includes {len(valid_dataset)} samples.")

    loader = {}
    loader["train"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.jobs.training.n_batch,
        shuffle=True,
        num_workers=cfg.jobs.training.num_workers,
        pin_memory=cfg.jobs.training.pin_memory,
    )
    loader["valid"] = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.jobs.training.n_batch,
        shuffle=False,
        collate_fn=test_collate_fn,
    )

    print(next(iter(loader["train"]))[0].shape)

    model = ConvTasNet(
        cfg.jobs.model.params.n_basis,
        cfg.jobs.model.params.kernel_size,
        cfg.jobs.model.params.stride,
        cfg.jobs.model.params.enc_basis,
        cfg.jobs.model.params.dec_basis,
        cfg.jobs.model.params.sep_hidden_channels,
        cfg.jobs.model.params.sep_bottleneck_channels,
        cfg.jobs.model.params.sep_skip_channels,
        cfg.jobs.model.params.sep_kernel_size,
        cfg.jobs.model.params.sep_num_blocks,
        cfg.jobs.model.params.sep_num_layers,
        cfg.jobs.model.params.dilated,
        cfg.jobs.model.params.separable,
        cfg.jobs.model.params.sep_nonlinear,
        cfg.jobs.model.params.sep_norm,
        cfg.jobs.model.params.mask_nonlinear,
        cfg.jobs.model.params.causal,
        enc_nonlinear=None,
    )
    logger.info(f"{model}")
    logger.info(f"# Parameters: {model.num_parameters}")

    if cfg.jobs.training.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            logger.info("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        logger.info("Does NOT use CUDA")

    # Optimizer
    if cfg.jobs.optim.name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.jobs.optim.params.lr,
            weight_decay=cfg.jobs.optim.params.weight_decay,
        )
    elif cfg.jobs.optim.name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.jobs.optim.params.lr,
            weight_decay=cfg.jobs.optim.params.weight_decay,
        )
    else:
        raise ValueError(f"Not support optimizer {cfg.jobs.optim.name}")

    # Criterion
    if cfg.jobs.criterion.name == "sisdr":
        criterion = NegSISDR()
    else:
        raise ValueError(f"Not support criterion {cfg.jobs.criterion.name}")

    pit_criterion = PIT1d(criterion, n_sources=cfg.jobs.training.n_sources)
    trainer = Trainer(model, loader, pit_criterion, optimizer, cfg)
    trainer.run()
    os.exit()


if __name__ == "__main__":
    main()
