import torch
import torch.nn as nn

EPS = 1e-12


def sdr(input, target, eps=EPS):
    """
    Source-to-distortion ratio (SDR)
    Args:
        input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
        target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
    Returns:
        loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
    """
    n_dims = input.dim()

    assert n_dims in [
        2,
        3,
        4,
    ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

    loss = (torch.sum(target**2, dim=n_dims - 1) + eps) / (
        torch.sum((target - input) ** 2, dim=n_dims - 1) + eps
    )
    loss = 10 * torch.log10(loss)

    return loss


class SDR(nn.Module):
    def __init__(self, reduction="mean", eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ["mean", "sum", None]:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
        """
        n_dims = input.dim()

        assert n_dims in [
            2,
            3,
            4,
        ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(
            n_dims
        )

        loss = sdr(input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == "mean":
                    loss = loss.mean(dim=1)
                elif self.reduction == "sum":
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == "mean":
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == "sum":
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return True


class NegSDR(nn.Module):
    def __init__(self, reduction="mean", eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ["mean", "sum", None]:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, C, T)
            target (batch_size, T) or (batch_size, C, T)
        Returns:
            loss (batch_size,)
        """
        n_dims = input.dim()

        assert n_dims in [
            2,
            3,
            4,
        ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(
            n_dims
        )

        loss = -sdr(input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == "mean":
                    loss = loss.mean(dim=1)
                elif self.reduction == "sum":
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == "mean":
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == "sum":
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False


"""
    Scale-invariant-SDR (source-to-distortion ratio)
    See "SDR - half-baked or well done?"
    https://arxiv.org/abs/1811.02508
"""


def sisdr(input, target, eps=EPS):
    """
    Scale-invariant-SDR (source-to-distortion ratio)
    Args:
        input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
        target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
    Returns:
        loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
    """
    n_dims = input.dim()

    assert n_dims in [
        2,
        3,
        4,
    ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

    alpha = torch.sum(input * target, dim=n_dims - 1, keepdim=True) / (
        torch.sum(target**2, dim=n_dims - 1, keepdim=True) + eps
    )
    loss = (torch.sum((alpha * target) ** 2, dim=n_dims - 1) + eps) / (
        torch.sum((alpha * target - input) ** 2, dim=n_dims - 1) + eps
    )
    loss = 10 * torch.log10(loss)

    return loss


class SISDR(nn.Module):
    def __init__(self, reduction="mean", eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ["mean", "sum", None]:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
        """
        n_dims = input.dim()

        assert n_dims in [
            2,
            3,
            4,
        ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(
            n_dims
        )

        loss = sisdr(input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == "mean":
                    loss = loss.mean(dim=1)
                elif self.reduction == "sum":
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == "mean":
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == "sum":
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return True


class NegSISDR(nn.Module):
    def __init__(self, reduction="mean", eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ["mean", "sum", None]:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, C, T)
            target (batch_size, T) or (batch_size, C, T)
        Returns:
            loss (batch_size,)
        """
        n_dims = input.dim()

        assert n_dims in [
            2,
            3,
            4,
        ], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(
            n_dims
        )

        loss = -sisdr(input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == "mean":
                    loss = loss.mean(dim=1)
                elif self.reduction == "sum":
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == "mean":
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == "sum":
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False
