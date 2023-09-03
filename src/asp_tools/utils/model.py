import torch.nn as nn


def choose_nonlinear(name, **kwargs):
    if name == "relu":
        nonlinear = nn.ReLU()
    elif name == "sigmoid":
        nonlinear = nn.Sigmoid()
    elif name == "softmax":
        assert "dim" in kwargs, "dim is expected for softmax."
        nonlinear = nn.Softmax(**kwargs)
    elif name == "tanh":
        nonlinear = nn.Tanh()
    elif name == "leaky-relu":
        nonlinear = nn.LeakyReLU()
    elif name == "gelu":
        nonlinear = nn.GELU()
    else:
        raise NotImplementedError(
            "Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(
                name
            )
        )

    return nonlinear
