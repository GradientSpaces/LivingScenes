import torch
import logging


def naninf_hook(self, inputs, outputs):
    if not isinstance(outputs, tuple):
        outputs = [outputs]

    if not isinstance(inputs, tuple):
        inputs = [inputs]

    for i, out in enumerate(outputs):
        if not isinstance(out, torch.Tensor):
            continue
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            logging.error("In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in output {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                out[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            )
        inf_mask = torch.isinf(out)
        if inf_mask.any():
            logging.error("In", self.__class__.__name__)
            raise RuntimeError(
                f"Found Inf in output {i} at indices: ",
                inf_mask.nonzero(),
                "where:",
                out[inf_mask.nonzero()[:, 0].unique(sorted=True)],
            )

    for i, inp in enumerate(inputs):
        if not isinstance(inp, torch.Tensor):
            continue
        nan_mask = torch.isnan(inp)
        if nan_mask.any():
            logging.error("In", self.__class__.__name__)
            raise RuntimeError(
                f"Found NAN in input {i} at indices: ",
                nan_mask.nonzero(),
                "where:",
                inp[nan_mask.nonzero()[:, 0].unique(sorted=True)],
            )
        inf_mask = torch.isinf(inp)
        if inf_mask.any():
            logging.error("In", self.__class__.__name__)
            raise RuntimeError(
                f"Found Inf in input {i} at indices: ",
                inf_mask.nonzero(),
                "where:",
                inp[inf_mask.nonzero()[:, 0].unique(sorted=True)],
            )
