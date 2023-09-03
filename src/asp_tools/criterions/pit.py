import itertools

import torch
import torch.nn as nn


def pit(criterion, input, target, n_sources=None, patterns=None, batch_mean=True):
    """
    Args:
        criterion <callable>
        input (batch_size, n_sources, *)
        output (batch_size, n_sources, *)
    Returns:
        loss (batch_size,): minimum loss for each data
        pattern (batch_size,): permutation indices
    """
    if patterns is None:
        if n_sources is None:
            n_sources = input.size(1)
        patterns = list(itertools.permutations(range(n_sources)))
        patterns = torch.Tensor(patterns).long()

    P = len(patterns)
    possible_loss = []

    for idx in range(P):
        pattern = patterns[idx]
        loss = criterion(input, target[:, pattern], batch_mean=False)
        possible_loss.append(loss)

    possible_loss = torch.stack(possible_loss, dim=1)

    # possible_loss (batch_size, P)
    if hasattr(criterion, "maximize") and criterion.maximize:
        loss, indices = torch.max(
            possible_loss, dim=1
        )  # loss (batch_size,), indices (batch_size,)
    else:
        loss, indices = torch.min(
            possible_loss, dim=1
        )  # loss (batch_size,), indices (batch_size,)

    if batch_mean:
        loss = loss.mean(dim=0)

    return loss, patterns[indices]


class PIT(nn.Module):
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        super().__init__()

        self.criterion = criterion
        patterns = list(itertools.permutations(range(n_sources)))
        self.patterns = torch.Tensor(patterns).long()

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, n_sources, *)
            target (batch_size, n_sources, *)
        Returns:
            loss (batch_size,): minimum loss for each data
            pattern (batch_size,): permutation indices
        """
        loss, pattern = pit(
            self.criterion, input, target, patterns=self.patterns, batch_mean=batch_mean
        )

        return loss, pattern


class PIT1d(PIT):
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        super().__init__(criterion, n_sources)
