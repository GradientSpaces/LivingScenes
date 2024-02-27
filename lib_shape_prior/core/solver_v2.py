"""
solve the optimization on a finite dataset

2022.9.12: update to iteration oriented
Because on different dataset, sometimes the iteration is the key factor, not the epoch, update the lr decay, the model viz, the model eval, save is controlled by the iteration

"""
from copy import deepcopy
import os
import logging
import torch
from torch.utils.data import DataLoader
import gc
from .solver_utils import naninf_hook
from core.models.utils.misc import cfg_with_default


class Solver_v2(object):
    def __init__(self, cfg, model, datasets_dict, logger):
        self.cfg = deepcopy(cfg)

        self.modes = self.cfg["modes"]
        if "train" in self.modes:
            assert self.modes[0] == "train", "should write train as the first phase"
        self.dataloader_dict = {}
        for mode in cfg["modes"]:  # prepare dataloader
            if mode.lower() == "train" or cfg["evaluation"]["batch_size"] < 0:
                bs = cfg["training"]["batch_size"]
            else:
                bs = cfg["evaluation"]["batch_size"]
            n_workers = cfg["dataset"]["num_workers"]
            # decide shuffle
            shuffle_dataset = True if mode in ["train"] else False
            if "shuffle" in cfg["evaluation"].keys():
                if cfg["evaluation"]["shuffle"]:
                    shuffle_dataset = True
            logging.debug(f"{mode} dataloader use pin_mem={cfg['dataset']['pin_mem']}")
            self.dataloader_dict[mode] = DataLoader(
                datasets_dict[mode],
                batch_size=bs,
                shuffle=shuffle_dataset,
                num_workers=n_workers,
                pin_memory=cfg["dataset"]["pin_mem"],
                drop_last=mode == "train",
                collate_fn=getattr(datasets_dict[mode], "collate_fn", None),
            )
        self.model = model
        self.logger = logger

        self.current_epoch = 1
        self.batch_count = 0
        self.batch_in_epoch_count = 0
        self.total_iter = cfg["training"]["total_iter"]

        self.eval_every_iter = int(cfg["evaluation"]["eval_every_iter"])

        self.clear_phase_cache = cfg["training"]["clear_phase_cache"]

        # save lr decay
        self.lr_config = self.init_lr_schedule()

        # handle resume and initialization
        self.loading_ignore_key = cfg_with_default(cfg, ["logging", "ignore_loading_key"], [])
        if cfg["resume"]:  # resume > initialization
            self.solver_resume()
        elif len(cfg["training"]["initialize_network_file"]) > 0:
            assert isinstance(
                cfg["training"]["initialize_network_file"], list
            ), "Initialization from file config should be a list fo file path"
            self.initialize_from_file(
                cfg["training"]["initialize_network_file"],
                cfg["training"]["initialize_network_name"],
            )
        self.model.to_gpus()

        # control viz in model and logger
        log_config = self.cfg["logging"]
        self.viz_interval_iter = log_config["viz_iter_interval"]
        self.viz_nontrain_interval = log_config["viz_nontrain_interval"]
        self.viz_flag = False

        # ! warning, the model save is controlled by iteration, not epoch
        self.checkpoint_iter = log_config["checkpoint_iter"]

        # register the hooks
        if cfg["enable_anomaly"]:
            logging.warning("Register NaN-Inf hooks")
            for net in self.model.network.network_dict.values():
                for submodule in net.modules():
                    submodule.register_forward_hook(naninf_hook)

        # * set the maximum step for early termination
        self.maximum_iters_per_ep = cfg_with_default(cfg, ["training", "maximum_iters_per_ep"], -1)
        self.early_terminate = self.maximum_iters_per_ep > 0
        if self.early_terminate:
            logging.warning(f"Solver-v2 set early termination at {self.early_terminate}")

        return

    def solver_resume(self):
        resume_key = self.cfg["resume"]
        checkpoint_dir = os.path.join(
            self.cfg["root"], "log", self.cfg["logging"]["log_dir"], "checkpoint"
        )
        if resume_key == "latest":
            checkpoint_founded = os.listdir(checkpoint_dir)
            checkpoint_fn = None
            for fn in checkpoint_founded:
                if fn.endswith("_latest.pt"):
                    checkpoint_fn = os.path.join(checkpoint_dir, fn)
        else:
            checkpoint_fn = os.path.join(checkpoint_dir, resume_key + ".pt")
        try:
            checkpoint = torch.load(checkpoint_fn)
        except:
            logging.warning(f"ckpt file {checkpoint_fn} load fail, skip loading!")
            return
        logging.info("Checkpoint {} Loaded".format(checkpoint_fn))
        self.current_epoch = checkpoint["epoch"]
        self.batch_count = checkpoint["batch"]
        self.model.model_resume(
            checkpoint,
            is_initialization=False,
            loading_ignore_key=self.loading_ignore_key,
            strict=len(self.loading_ignore_key) == 0,  # if ignore during resuming, can't use strict
        )
        self.adjust_lr()
        self.current_epoch += 1
        return

    def initialize_from_file(self, filelist, network_name):
        for fn in filelist:
            checkpoint = torch.load(fn)
            logging.info("Initialization {} Loaded".format(fn))
            self.model.model_resume(
                checkpoint,
                is_initialization=True,
                network_name=network_name,
                loading_ignore_key=self.loading_ignore_key,
                strict=len(self.loading_ignore_key)
                == 0,  # if ignore during resuming, can't use strict
            )
        return

    def run(self):
        logging.info("Start Running...")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}...")
        while self.batch_count <= self.total_iter:
            need_val_flag = "train" not in self.modes  #! the train should always be the first
            for mode in self.modes:
                if self.clear_phase_cache:
                    torch.cuda.empty_cache()
                if mode.lower() != "train" and not need_val_flag:
                    continue  # for val and test, skip if not meets eval epoch interval
                batch_total_num = len(self.dataloader_dict[mode])
                self.batch_in_epoch_count = 0
                for batch in iter(self.dataloader_dict[mode]):
                    if (
                        self.batch_in_epoch_count > self.maximum_iters_per_ep
                        and self.early_terminate
                    ):
                        logging.warning(f"Solver v2 early break at {self.batch_in_epoch_count}")
                        break
                    self.batch_in_epoch_count += 1
                    self.batch_count += 1
                    self.viz_flag = self.viz_state(mode)
                    batch[0]["epoch"] = self.current_epoch
                    if mode == "train":
                        batch = self.model.train_batch(batch, self.viz_flag)
                    else:
                        batch = self.model.val_batch(batch, self.viz_flag)
                    batch = self.wrap_output(
                        batch,
                        min(self.maximum_iters_per_ep, batch_total_num)
                        if self.early_terminate
                        else batch_total_num,
                        mode=mode,
                    )
                    batch['batch_count'] = self.batch_count
                    batch['total_iter'] = self.total_iter
                    self.logger.log_batch(batch)

                    if self.batch_count % self.checkpoint_iter == 0:
                        self.logger.model_logger.set_save_flag(True)

                    if self.batch_count % self.eval_every_iter == 0:
                        need_val_flag = True
                    self.adjust_lr()

                self.logger.log_phase()
                gc.collect()

            self.current_epoch += 1
        self.logger.end_log()
        return

    def wrap_output(self, batch, batch_total, mode="train"):
        assert "meta_info" in batch.keys()
        wrapped = dict()

        wrapped["viz_flag"] = self.viz_flag

        wrapped["batch"] = self.batch_count
        wrapped["batch_in_epoch"] = self.batch_in_epoch_count
        wrapped["batch_total"] = batch_total
        wrapped["epoch"] = self.current_epoch
        wrapped["phase"] = mode.lower()

        wrapped["output_parser"] = self.model.output_specs
        wrapped["save_method"] = self.model.save_checkpoint
        wrapped["meta_info"] = batch["meta_info"]
        wrapped["data"] = batch

        return wrapped

    def init_lr_schedule(self):
        schedule = self.cfg["training"]["optim"]
        schedule_keys = schedule.keys()
        if "all" in schedule_keys:
            schedule_keys = ["all"]
        for k in schedule_keys:
            if isinstance(schedule[k]["decay_schedule"], int):
                assert NotImplementedError()
            elif isinstance(schedule[k]["decay_schedule"], list):
                assert isinstance(schedule[k]["decay_factor"], list)
            else:
                assert RuntimeError("Lr Schedule error!")
        return schedule

    def adjust_lr(self, specific_iter=None):
        iter = self.batch_count if specific_iter is None else specific_iter
        for k in self.lr_config.keys():
            if iter in self.lr_config[k]["decay_schedule"]:
                optimizer = self.model.optimizer_dict[k]
                for param_group in optimizer.param_groups:
                    lr_before = param_group["lr"]
                    factor = self.lr_config[k]["decay_factor"][
                        self.lr_config[k]["decay_schedule"].index(iter)
                    ]
                    param_group["lr"] = param_group["lr"] * factor
                    param_group["lr"] = max(param_group["lr"], self.lr_config[k]["lr_min"])
                    lr_new = param_group["lr"]
                logging.info(
                    "After iters {}, Change {} lr {:.5f} to {:.5f}".format(
                        iter, k, lr_before, lr_new
                    )
                )

    def viz_state(self, mode):
        viz_flag = True
        if mode == "train":
            if self.batch_count % self.viz_interval_iter != 0:
                viz_flag = False
        else:
            if self.batch_in_epoch_count % self.viz_nontrain_interval != 0:
                viz_flag = False
        return viz_flag
