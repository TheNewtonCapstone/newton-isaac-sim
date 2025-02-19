def get_torch_activation_fn(activation_fn: str):
    import torch.nn

    nn_module_name = activation_fn.split(".")[-1]

    return getattr(torch.nn, nn_module_name)
