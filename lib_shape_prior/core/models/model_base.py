import torch.nn as nn
import torch
import copy
import logging
import numpy as np


class ModelBase(object):
    def __init__(self, cfg, network):
        """
        Model Base
        """
        self.cfg = copy.deepcopy(cfg)
        self.__dataparallel_flag__ = False
        self.network = network
        self.optimizer_specs = self.cfg["training"]["optim"]
        self.optimizer_dict = self._register_optimizer()
        # self.to_gpus()
        self.output_specs = {
            "metric": [],
        }
        self.grad_clip = float(cfg["training"]["grad_clip"])
        self.loss_clip = float(cfg["training"]["loss_clip"])
        return

    def _register_optimizer(self):
        optimizer_dict = {}
        parameter_keys = self.optimizer_specs.keys()
        logging.debug("Config defines {} network parameters optimization".format(parameter_keys))
        if len(parameter_keys) != len(self.network.network_dict.keys()):
            logging.warning("Network Components != Optimizer Config")
        if "all" in parameter_keys:
            optimizer = torch.optim.Adam(
                params=self.network.parameters(),
                lr=self.optimizer_specs["all"]["lr"],
            )
            optimizer_dict["all"] = optimizer
        else:
            for key in parameter_keys:
                try:
                    optimizer = torch.optim.Adam(
                        params=self.network.network_dict[key].parameters(),
                        lr=self.optimizer_specs[key]["lr"],
                    )
                    optimizer_dict[key] = optimizer
                except:
                    raise RuntimeError(
                        "Optimizer registration of network component {} fail!".format(key)
                    )
        return optimizer_dict

    def count_parameters(self):
        net = (
            self.network.module.network_dict
            if self.__dataparallel_flag__
            else self.network.network_dict
        )
        for k, v in net.items():
            count = sum(p.numel() for p in v.parameters())
            logging.info("Model-{} has {} parameters".format(k, count))

    def _preprocess(self, batch, viz_flag=False):
        """
        Additional operation if necessary before send batch to network
        """
        data, meta_info = batch
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda().float()
        data["phase"] = meta_info["mode"][0]
        data["viz_flag"] = viz_flag
        batch = {"model_input": data, "meta_info": meta_info}
        return batch

    def _predict(self, batch, viz_flag=False):
        """
        forward through the network
        """
        model_out = self.network(batch["model_input"], viz_flag)
        for k, v in model_out.items():
            batch[k] = v  # directly place all output to batch dict
        return batch

    def _postprocess(self, batch):
        """
        Additional operation process on one gpu or cpu
        :return: a dictionary
        """
        for k in self.output_specs["metric"]:
            try:
                batch[k] = batch[k].mean()
            except:
                # sometime the metric might not be computed, e.g. during training the val metric
                pass
        return batch

    def _postprocess_after_optim(self, batch):
        """
        Additional operation process after optimizer.step
        :return: a dictionary
        """
        return batch

    def _detach_before_return(self, batch):
        for k, v in batch.items():
            if isinstance(v, dict):
                self._detach_before_return(v)
            if isinstance(v, torch.Tensor):
                batch[k] = v.detach()
        return batch

    def train_batch(self, batch, viz_flag=False):
        batch = self._preprocess(batch, viz_flag)
        self.set_train()
        self.zero_grad()
        # ! 2022.12.25 add here use model.zero_grad to also remove some networks that is not in the optimizer list!!
        self.network.zero_grad()
        batch = self._predict(batch, viz_flag)
        batch = self._postprocess(batch)
        if self.loss_clip > 0.0:
            if abs(batch["batch_loss"]) > self.loss_clip:
                logging.warning(f"Loss Clipped from {abs(batch['batch_loss'])} to {self.loss_clip}")
            batch["batch_loss"] = torch.clamp(batch["batch_loss"], -self.loss_clip, self.loss_clip)
        batch["batch_loss"].backward()
        if self.grad_clip > 0:
            # old: update Jul 5th 2022
            # mew: update Dec 12th 2022
            for k in self.network.network_dict.keys():
                clip_grad_norm(k, self.network.network_dict[k], self.grad_clip)
        self.optimizers_step()
        batch = self._postprocess_after_optim(batch)
        batch = self._detach_before_return(batch)
        return batch

    def val_batch(self, batch, viz_flag=False):
        batch = self._preprocess(batch, viz_flag)
        self.set_eval()
        with torch.no_grad():
            batch = self._predict(batch, viz_flag)
        batch = self._postprocess(batch)
        batch = self._dataparallel_postprocess(batch)
        batch = self._postprocess_after_optim(batch)
        batch = self._detach_before_return(batch)
        return batch

    def _dataparallel_postprocess(self, batch):
        if self.__dataparallel_flag__:
            for k in batch.keys():
                if k.endswith("loss") or k in self.output_specs["metric"]:
                    if isinstance(batch[k], list):
                        for idx in len(batch[k]):
                            batch[k][idx] = batch[k][idx].mean()
                    else:
                        batch[k] = batch[k].mean()
        return batch

    def zero_grad(self):
        for k in self.optimizer_dict.keys():
            self.optimizer_dict[k].zero_grad()

    def optimizers_step(self):
        for k in self.optimizer_dict.keys():
            self.optimizer_dict[k].step()

    def model_resume(
        self, checkpoint, is_initialization, network_name=None, loading_ignore_key=[], strict=True
    ):
        # reprocess to fit the old version
        state_dict = {}
        logging.info("Load from ep {}".format(checkpoint["epoch"]))
        for k, v in checkpoint["model_state_dict"].items():
            if k.startswith("module."):
                name = ".".join(k.split(".")[1:])
            else:
                name = k
            ignore_flag = False
            for ignore_key in loading_ignore_key:
                if ignore_key in name:
                    logging.warning(f"ignore checkpoint {name} because set ignore key {ignore_key}")
                    ignore_flag = True
                    break
            if ignore_flag:
                continue
            state_dict[name] = v
        checkpoint["model_state_dict"] = state_dict
        if not is_initialization or network_name == ["all"]:
            self.network.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            for k, v in checkpoint["optimizers_state_dict"]:
                self.optimizer_dict[k].load_state_dict(v)
                # send to cuda
                for state in self.optimizer_dict[k].state.values():
                    for _k, _v in state.items():
                        if torch.is_tensor(_v):
                            state[_k] = _v.cuda()
        else:
            if network_name is not None:
                prefix = ["network_dict." + name for name in network_name]
                restricted_model_state_dict = {}
                for k, v in checkpoint["model_state_dict"].items():
                    for pf in prefix:
                        if k.startswith(pf):
                            restricted_model_state_dict[k] = v
                            break
                checkpoint["model_state_dict"] = restricted_model_state_dict
            self.network.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def save_checkpoint(self, filepath, additional_dict=None):
        save_dict = {
            "model_state_dict": self.network.module.state_dict()
            if self.__dataparallel_flag__
            else self.network.state_dict(),
            "optimizers_state_dict": [
                (k, opti.state_dict()) for k, opti in self.optimizer_dict.items()
            ],
        }
        if additional_dict is not None:
            for k, v in additional_dict.items():
                save_dict[k] = v
        torch.save(save_dict, filepath)

    def to_gpus(self):
        if torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)
            self.__dataparallel_flag__ = True
        else:
            self.__dataparallel_flag__ = False
        self.network.cuda()

    def set_train(self):
        self.network.train()

    def set_eval(self):
        self.network.eval()


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = copy.deepcopy(cfg)
        self.network_dict = None

    def forward(self, *input):
        raise NotImplementedError


from torch._six import inf


def clip_grad_norm(
    name, net, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False
) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:``parameters`` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = net.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

    if total_norm > max_norm:
        # actual clip happens
        logging.info(f"Warning! Clip {name} gradient from {total_norm} to {max_norm}")
        for n, p in net.named_parameters():
            g_norm = p.grad.norm()
            logging.warning(f"{n}: {g_norm}")

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm
