from random import random
from torch.utils.data import Dataset
import logging
import json
import os
import os.path as osp
import numpy as np
from os.path import join
import h5py
from tqdm import tqdm

from core.models.utils.misc import cfg_with_default
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2mat
from transforms3d.axangles import axangle2mat
import torch
import csv
import pickle


class Dataset(Dataset):
    def __init__(self, cfg, mode) -> None:
        super().__init__()

        d_cfg = cfg["dataset"]
        self.dataset_mode = cfg_with_default(d_cfg, ["dataset_mode"], "hybrid")
        assert self.dataset_mode in ["occ", "hybrid"]

        self.input_mode = cfg_with_default(d_cfg, ["input_mode"], "pcl")
        assert self.input_mode in ["pcl", "dep"]

        self.balanced_class = cfg_with_default(d_cfg, ["balanced_class"], True)

        self.mode = mode.lower()
        self.dataset_proportion = d_cfg["dataset_proportion"][cfg["modes"].index(self.mode)]
        self.data_root = join(cfg["root"], d_cfg["data_root"])

        self.partnet_flag = cfg_with_default(d_cfg, ["partnet_flag"], False)
        if self.partnet_flag:
            # only when all tag appears in the part's semantic trace will include this part
            self.partnet_filter_tags = cfg_with_default(d_cfg, ["partnet_filter_tags"], [])

        # build and check meta list
        self.cates = d_cfg["categories"]
        self.n_cates = len(self.cates)
        self.cates_count = []
        self.meta_dict, self.meta_list = {}, []

        self.depth_postfix = cfg_with_default(d_cfg, ["depth_postfix"], "_dep")

        for cls_id, cate in enumerate(self.cates):
            meta_list = []
            internal_idx = len(self.meta_list)

            if self.partnet_flag:  # ! Partnet
                assert "partnet_split_dir" in d_cfg.keys()
                split_fn = f"{cate.capitalize()}.{self.mode.lower()}.json"
                split_fn = osp.join(cfg["root"], d_cfg["partnet_split_dir"], split_fn)
                # the training set is the union of train and non-mob subset, all others are validation
                with open(split_fn, "r") as f:
                    self.partnet_split_data = json.load(f)
                partnet_split = [p["anno_id"] for p in self.partnet_split_data]
                partnet_data_root = osp.join(
                    self.data_root, cate, d_cfg["partnet_level_names"][cate]
                )
                processed_list = [
                    d
                    for d in os.listdir(partnet_data_root)
                    if osp.isdir(osp.join(partnet_data_root, d))
                ]
                split = [i for i in partnet_split if i in processed_list]
                logging.info(
                    f"Partnet official split {self.mode} has {len(split)} intersection with the static processed partnet data"
                )

                for obj_id in split:
                    meta_pkl_fn = osp.join(partnet_data_root, obj_id, "meta.pkl")
                    with open(meta_pkl_fn, "rb") as f:
                        meta_info_list = pickle.load(f)
                    obj_dir = osp.join(partnet_data_root, obj_id, "combined_part_mesh")
                    parts_list = [d[:-4] for d in os.listdir(obj_dir) if d.endswith(".obj")]
                    if self.input_mode == "dep":
                        raise NotImplementedError()
                    else:
                        for part in parts_list:
                            # do meta info check and filtering
                            meta_check_valid = False
                            part_id = int(part.split("_")[0])
                            meta = None
                            for _meta in meta_info_list:
                                if _meta["id"] == part_id:
                                    meta = _meta
                                    semantic_list = meta["semantic_all"][meta["semantic"]]
                                    meta_check_valid = True
                                    for tag in self.partnet_filter_tags:
                                        if tag not in semantic_list:
                                            meta_check_valid = False
                                            break
                                    break

                            part_dir = osp.join(obj_dir, part)
                            if (
                                osp.exists(part_dir)
                                and len(os.listdir(part_dir)) > 0
                                and meta_check_valid
                            ):
                                meta_list.append(
                                    {
                                        "cate": cate,
                                        "part_id": part,  # the part name
                                        "obj_id": obj_id,
                                        "obj_dir": part_dir,
                                        "internal_idx": internal_idx,
                                        "cls": cls_id,
                                    }
                                )
                                internal_idx += 1
                            else:
                                if meta_check_valid:
                                    logging.warning(f"Dataset {cate} {obj_id} broken, skipped")
                                else:
                                    logging.debug(
                                        f"Dataset {cate} {obj_id} filtered out by tags conditions"
                                    )

            else:  # ! ShapeNet
                # support multiple split
                if "shapenet_split_fn" in d_cfg.keys():
                    split = self.read_split(
                        osp.join(cfg["root"], d_cfg["shapenet_split_fn"]), cate, self.mode
                    )
                    
                elif "split_room4cate" in d_cfg.keys():
                    split_file = osp.join(cfg["root"], d_cfg["split_room4cate"], cate, self.mode+'.lst')
                    with open(split_file, "r") as f:
                        split = f.read().split("\n")
                    split = list(filter(lambda x: len(x) > 0, split))
                
                else:
                    split_fn = osp.join(cfg["root"], d_cfg["split"][self.mode])
                    with open(split_fn, "r") as f:
                        split = f.read().split("\n")
                    split = list(filter(lambda x: len(x) > 0, split))

                constrain_flag = False
                if "constrain_txt" in d_cfg.keys():
                    constrain_flag = True
                    txt_fn = osp.join(cfg["root"], d_cfg["constrain_txt"][cate])
                    valid_id_list = []
                    with open(txt_fn) as txt_file:
                        lines = txt_file.readlines()
                    valid_id_list = [l.split(",")[0].split(".")[-1] for l in lines[1:]]
                    logging.info(f"constrain list len={len(valid_id_list)} from {txt_file}")
                exclude_flag = False
                if "exclude_txt" in d_cfg.keys():
                    exclude_flag = True
                    exclude_id_list = []
                    for txt_fn in d_cfg["exclude_txt"][cate]:
                        txt_fn = osp.join(cfg["root"], txt_fn)
                        with open(txt_fn) as txt_file:
                            lines = txt_file.readlines()
                        exclude_id_list += [l.split(",")[0].split(".")[-1] for l in lines[1:]]
                    logging.info(
                        f"exclude list len={len(exclude_id_list)} from {d_cfg['exclude_txt'][cate]}"
                    )

                for obj_id in split:
                    if constrain_flag:
                        if obj_id not in valid_id_list:
                            continue
                    if exclude_flag:
                        if obj_id in exclude_id_list:
                            continue
                    obj_dir = osp.join(self.data_root, cate, obj_id)
                    if self.input_mode == "dep":
                        dep_dir = osp.join(self.data_root, cate + self.depth_postfix, obj_id)
                        dep_check_flag = osp.exists(dep_dir) and len(os.listdir(dep_dir)) > 0
                        if osp.exists(obj_dir) and len(os.listdir(obj_dir)) > 0 and dep_check_flag:
                            meta_list.append(
                                {
                                    "cate": cate,
                                    "obj_id": obj_id,
                                    "obj_dir": obj_dir,
                                    "dep_dir": dep_dir,
                                    "internal_idx": internal_idx,
                                    "cls": cls_id,
                                }
                            )
                            internal_idx += 1
                        else:
                            logging.warning(f"Dataset {cate} {obj_id} broken, skipped")
                    else:
                        if osp.exists(obj_dir) and len(os.listdir(obj_dir)) > 0:
                            meta_list.append(
                                {
                                    "cate": cate,
                                    "obj_id": obj_id,
                                    "obj_dir": obj_dir,
                                    "internal_idx": internal_idx,
                                    "cls": cls_id,
                                }
                            )
                            internal_idx += 1
                        else:
                            logging.warning(f"Dataset {cate} {obj_id} broken, skipped")

            self.meta_dict[cate] = meta_list[: int(len(meta_list) * self.dataset_proportion)]
            self.meta_list += self.meta_dict[cate]
            self.cates_count.append(len(self.meta_dict[cate]))
        logging.info(
            f"Dataset {mode} with {self.dataset_proportion * 100}% data, dataset len is {len(self)}, total len is {len(self.meta_list)}"
        )

        # init sampling configs
        self.n_input = d_cfg["n_pcl"]
        self.n_input_fewer = cfg_with_default(d_cfg, ["n_pcl_fewer"], -1.0)
        self.n_query_uni = d_cfg["n_query_uni"]
        if self.dataset_mode == "hybrid":
            self.n_query_nss = d_cfg["n_query_nss"]
        self.n_query_eval = d_cfg["n_query_eval"]
        self.noise_std = d_cfg["noise_std"]
        self.field_mode = d_cfg["field_mode"].lower()
        assert self.field_mode in ["sdf", "occ"]
        if self.dataset_mode == "occ":
            assert self.field_mode == "occ"
        if self.field_mode == "sdf":
            assert self.dataset_mode == "hybrid", "only hybrid (our sampled dataset) support sdf"

        if self.input_mode == "dep":
            self.dep_total_view = d_cfg["dep_total_view"]
            self.dep_max_use_view = d_cfg["dep_max_use_view"]
            self.dep_min_use_view = cfg_with_default(d_cfg, ["dep_min_use_view"], 1)

        # cache data
        self.cache_flag = d_cfg["ram_cache"]
        self.CACHE = []
        if self.cache_flag:
            logging.info(f"Caching {self.mode} dataset...")
            for idx in tqdm(range(len(self.meta_list))):
                self.CACHE.append(self.__read_from_disk__(idx))

        # augmentation configs
        self.use_aug = cfg_with_default(d_cfg, ["use_augmentation"], False)
        self.use_aug_nontrain = cfg_with_default(d_cfg, ["use_augmentation_nontrain"], False)

        self.aug_version = cfg_with_default(d_cfg, ["aug_version"], "v1")
        self.aug_ratio = cfg_with_default(d_cfg, ["aug_ratio"], 0.4)

        if self.use_aug:
            if self.aug_version == "v1":
                self.aug_v1_config(d_cfg)
            elif self.aug_version == "v2":
                logging.warning(
                    "!Warning! aug version v2, can only used to train implicit reconstruction, not for the BBox!!"
                )
                self.aug_v2_config(d_cfg)
            else:
                raise NotImplementedError()

        # sampling augmentation config, different from v1, v2 etc aug, this aug is independent to them, just to change the sampling density
        self.use_sampling_aug = cfg_with_default(d_cfg, ["use_sampling_augmentation"], False)
        self.sampling_aug_version = cfg_with_default(d_cfg, ["sampling_aug_version"], "s1")
        if self.use_sampling_aug:
            if self.sampling_aug_version == "s1":
                self.sampling_aug_s1_config(d_cfg)
            else:
                raise NotImplementedError()

    def __len__(self):
        if self.balanced_class:
            return int(max([len(i) for i in self.meta_dict.values()]) * len(self.cates))
        else:
            return len(self.meta_list)

    def __read_from_ram__(self, index):
        return self.CACHE[index]

    def __read_from_disk__(self, index):
        meta = self.meta_list[index]
        obj_dir = meta["obj_dir"]
        pointcloud_data = np.load(osp.join(obj_dir, "pointcloud.npz"))
        # occ_loc, occ_scale = pointcloud_data["loc"], pointcloud_data["scale"]
        pointcloud = pointcloud_data["points"]
        dep_list = []
        if self.input_mode == "dep":
            dep_dir = meta["dep_dir"]
            for vid in range(self.dep_total_view):
                dep_list.append(np.load(osp.join(dep_dir, f"dep_pcl_{vid}.npz"))["p_w"])

        if self.dataset_mode == "hybrid":
            uni_query = np.load(osp.join(obj_dir, "points_uni.npz"))["points"]
            nss_query = np.load(osp.join(obj_dir, "points_nss.npz"))["points"]
            return uni_query, nss_query, pointcloud, dep_list
        elif self.dataset_mode == "occ":
            uni_data = np.load(osp.join(obj_dir, "points.npz"))
            query, occ = uni_data["points"], np.unpackbits(uni_data["occupancies"])
            return query, occ, pointcloud, dep_list

    def __getitem__(self, index):
        if self.balanced_class:
            cls_id = index % self.n_cates
            cate = self.cates[cls_id]
            cate_idx = int((index - cls_id) / self.n_cates) % len(self.meta_dict[cate])
            meta_info = self.meta_dict[cate][cate_idx]
        else:
            meta_info = self.meta_list[index]
            cls_id = meta_info["cls"]

        internal_idx = meta_info["internal_idx"]

        # # debug
        # assert internal_idx == index
        viz_id = f"{self.mode}_{meta_info['cate']}_{meta_info['obj_id']}_{internal_idx}"
        meta_info["viz_id"] = viz_id
        meta_info["mode"] = self.mode
        ret = {}

        # Load training samples
        if self.dataset_mode == "hybrid":
            if self.cache_flag:
                uni_query, nss_query, pointcloud, dep_list = self.__read_from_ram__(internal_idx)
            else:
                uni_query, nss_query, pointcloud, dep_list = self.__read_from_disk__(internal_idx)
            # # debug
            # np.savetxt("./debug/pointcloud.txt", pointcloud)
            # np.savetxt("./debug/nss_pos.txt", nss_query[nss_query[:, -1] > 0])
            # np.savetxt("./debug/nss_neg.txt", nss_query[nss_query[:, -1] <= 0])
            # np.savetxt("./debug/uni_pos.txt", uni_query[uni_query[:, -1] > 0])
            # np.savetxt("./debug/uni_neg.txt", uni_query[uni_query[:, -1] <= 0])

            # load training supervision
            if self.n_query_uni > 0:
                choice = np.random.choice(len(uni_query), self.n_query_uni)
                sdf_data = uni_query[choice]
                ret["points.uni"] = sdf_data[:, :3]
                if self.field_mode == "sdf":
                    ret["points.uni.value"] = sdf_data[:, 3]
                else:
                    ret["points.uni.value"] = (sdf_data[:, 3] <= 0).astype(np.float64)
            if self.n_query_nss > 0:
                choice = np.random.choice(len(nss_query), self.n_query_nss)
                sdf_data = nss_query[choice]
                ret["points.nss"] = sdf_data[:, :3]
                if self.field_mode == "sdf":
                    ret["points.nss.value"] = sdf_data[:, 3]
                else:
                    ret["points.nss.value"] = (sdf_data[:, 3] <= 0).astype(np.float64)

            # load evaluation if necessary
            if self.mode != "train":
                ret["eval.points"] = uni_query[: self.n_query_eval, :3]
                ret["eval.points.occ"] = (uni_query[: self.n_query_eval, 3] <= 0).astype(np.float64)
                ret["eval.points.value"] = uni_query[: self.n_query_eval, 3]
                ret["eval.pointcloud"] = pointcloud[: self.n_query_eval]
        elif self.dataset_mode == "occ":  # can only have occ mode, no sdf mode
            if self.cache_flag:
                query, occ, pointcloud, dep_list = self.__read_from_ram__(internal_idx)
            else:
                query, occ, pointcloud, dep_list = self.__read_from_disk__(internal_idx)
            if self.n_query_uni > 0:
                choice = np.random.choice(len(query), self.n_query_uni)
                ret["points.uni"] = query[choice]
                ret["points.uni.value"] = occ[choice].astype(np.float64)
            if self.mode != "train":
                ret["eval.points"] = query[: self.n_query_eval, :3]
                ret["eval.points.occ"] = occ[: self.n_query_eval].astype(np.float64)
                ret["eval.pointcloud"] = pointcloud[: self.n_query_eval]
        else:
            raise NotImplementedError()

        # load input
        bb_min, bb_max = pointcloud.min(axis=0), pointcloud.max(axis=0)
        total_size = (
            bb_max - bb_min
        ).max()  # ! wreid, occ data has unstable total size ~ 1.02, why?? maybe because the normalization weight is compute on CAD mesh but later occ do fusion to get watertight mesh, the watertight recon is a little bit larger than the original mesh!
        bbox_target = (abs(bb_min) + bb_max) / 2.0
        ret["bbox"] = bbox_target

        if self.input_mode == "pcl":
            input_src = pointcloud
        else:
            n_views = np.random.randint(low=self.dep_min_use_view, high=self.dep_max_use_view + 1)
            vid = np.random.choice(self.dep_total_view, n_views)
            input_src = np.concatenate([dep_list[v] for v in vid], 0)

        if self.use_sampling_aug and self.mode == "train":
            if self.sampling_aug_version == "s1":
                input_pcl = self.sampling_with_aug_v1(input_src, self.n_input)
            else:
                raise NotImplementedError()
        else:
            input_pcl = self.uniform_sampling(input_src, self.n_input)

        noise = self.noise_std * np.random.randn(*input_pcl.shape)
        input_pcl = input_pcl + noise
        ret["inputs"] = input_pcl

        # the last step is to augment the input (maybe the output)
        if self.use_aug and (self.mode == "train" or self.use_aug_nontrain):
            if self.aug_version == "v1":
                ret = self.augment_v1(ret, bottom_y=bb_min[1])
            elif self.aug_version == "v2":
                # ! WARNING V2 IS ONLY USED TO TRAIN OCCNET, CAN NOT TRAIN CANONICALIZER
                ret = self.augment_v2(ret)
            else:
                raise NotImplementedError()

        if self.n_input_fewer > 0:
            ret["inputs_fewer"] = self.uniform_sampling(ret["inputs"], self.n_input_fewer)

        ret["class"] = cls_id

        return ret, meta_info

    def read_split(self, path, id, phase):
        id_list = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    line_count += 1
                    if row[1] == id and row[-1] == phase:
                        id_list.append(row[-2])
        return id_list

    #####################################################################################
    # For sampling s1 aug
    #####################################################################################

    def sampling_aug_s1_config(self, d_cfg):
        self.s1_mixing_sampling_prob = d_cfg["s1_mixing_sampling_prob"]
        self.s1_mixing_mode_selection_ratio = np.array(d_cfg["s1_mixing_mode_selection_ratio"])
        self.s1_single_mode_selection_ratio = np.array(d_cfg["s1_single_mode_selection_ratio"])
        self.s1_single_mode_selection_ratio = (
            self.s1_single_mode_selection_ratio / self.s1_single_mode_selection_ratio.sum()
        )
        self.s1_single_mode_selection_ratio = np.cumsum(self.s1_single_mode_selection_ratio)
        self.s1_sampling_range = d_cfg["s1_sampling_range"]

        self.s1_gaussian_num_range = d_cfg["s1_gaussian_num_range"]
        self.s1_gaussian_std_range = d_cfg["s1_gaussian_std_range"]
        self.s1_gaussian_nss_range = d_cfg["s1_gaussian_nss_range"]

        self.s1_halfspace_num_range = d_cfg["s1_halfspace_num_range"]
        self.s1_halfspace_difference_range = d_cfg["s1_halfspace_difference_range"]

        return

    def sampling_with_aug_v1(self, pcl, N):
        if np.random.rand() < self.s1_mixing_sampling_prob:
            # mixing sampling
            random_seed = np.random.rand(3) * self.s1_mixing_mode_selection_ratio
            ratio = random_seed / (random_seed.sum() + 1e-8)
            N_uni = int(N * ratio[0])
            N_gauss = int(N * ratio[1])
            N_half = N - N_uni - N_gauss
            sampled_list = []
            if N_uni > 0:
                sampled_list.append(self.uniform_sampling(pcl, N_uni))
            if N_gauss > 0:
                sampled_list.append(self.random_gaussian_aug_sampling(pcl, N_gauss))
            if N_half > 0:
                sampled_list.append(self.random_half_space_aug_sampling(pcl, N_half))
            for i in range(len(sampled_list)):
                if sampled_list[i].ndim == 1:
                    sampled_list[i] = sampled_list[i][None, :]
            pcl_sampled = np.concatenate(sampled_list, 0)
        else:
            # single mode sampling
            random_seed = np.random.rand()
            if random_seed < self.s1_single_mode_selection_ratio[0]:
                pcl_sampled = self.uniform_sampling(pcl, N)
            elif random_seed > self.s1_single_mode_selection_ratio[1]:
                pcl_sampled = self.random_half_space_aug_sampling(pcl, N)
            else:
                pcl_sampled = self.random_gaussian_aug_sampling(pcl, N)
        # shrink the pointcloud / make duplication
        random_N = np.random.uniform(self.s1_sampling_range[0], self.s1_sampling_range[1])
        random_N = min(int(random_N * N), N)
        pcl_sampled = self.uniform_sampling(pcl_sampled, random_N)
        pcl_sampled = self.uniform_sampling(pcl_sampled, N)
        return pcl_sampled

    ###### Basic sampling method ########

    def uniform_sampling(self, pcl, N):
        choice = np.random.choice(len(pcl), N, replace=True)
        return pcl[choice]

    def weighted_sampling(self, pcl, weight, N):
        if weight.sum() == 0:
            logging.warning("Weight Aug Sampling failed, use uniform to cover the slot")
            return self.uniform_sampling(pcl, N)
        else:
            weight = weight / (weight.sum() + 1e-8)
        selection = torch.multinomial(torch.from_numpy(weight), N)
        return pcl[selection]

    def random_gaussian_aug_sampling(self, pcl, N):
        num_kernel = np.random.randint(
            low=self.s1_gaussian_num_range[0], high=self.s1_gaussian_num_range[1] + 1
        )
        anchor = self.uniform_sampling(pcl, num_kernel)
        random_dir = np.random.randn(num_kernel, 3)
        random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-8)
        random_drift = (
            np.random.uniform(
                low=self.s1_gaussian_nss_range[0],
                high=self.s1_gaussian_nss_range[1],
                size=num_kernel,
            )[:, None]
            * random_dir
        )
        mu = anchor + random_drift
        std = np.random.uniform(
            low=self.s1_gaussian_std_range[0], high=self.s1_gaussian_std_range[1], size=num_kernel
        )

        var = std**2
        denom = (2 * np.pi * var) ** 0.5
        dist = np.linalg.norm(pcl[None, ...] - mu[:, None, :], axis=-1)  # K,N
        num = np.exp(-(dist**2) / (2 * var[:, None]))
        prob = num / denom[:, None]
        decrease = prob.sum(0)

        weight = np.clip(1.0 - decrease, a_min=0.0, a_max=1.0)
        return self.weighted_sampling(pcl, weight, N)

    def random_half_space_aug_sampling(self, pcl, N):
        num_split = np.random.randint(
            low=self.s1_halfspace_num_range[0], high=self.s1_halfspace_num_range[1] + 1
        )
        anchor = self.uniform_sampling(pcl, num_split)
        random_dir = np.random.randn(num_split, 3)
        random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-8)
        d = pcl[None, ...] - anchor[:, None, :]
        inner = (d * random_dir[:, None, :]).sum(-1)  # K,N
        reduce_mask = (inner < 0).astype(np.float64)
        random_reduce = np.random.uniform(
            low=self.s1_halfspace_difference_range[0],
            high=self.s1_halfspace_difference_range[1],
            size=num_split,
        )
        decrease = (reduce_mask * random_reduce[:, None]).sum(0)
        weight = np.clip(1.0 - decrease, a_min=0.0, a_max=1.0)
        return self.weighted_sampling(pcl, weight, N)

    #####################################################################################
    # For standard v1, v2 aug
    #####################################################################################

    def aug_v2_config(self, d_cfg):
        self.random_rot_deg = cfg_with_default(d_cfg, ["random_rot_deg"], 0.0)
        self.random_shift_len = cfg_with_default(d_cfg, ["random_shift_len"], 0.0)
        self.random_scale_range = cfg_with_default(d_cfg, ["random_scale_range"], [1.0, 1.0])
        self.random_noise_range = cfg_with_default(d_cfg, ["random_noise_range"], 0.0)
        self.random_noise_prob = cfg_with_default(d_cfg, ["random_noise_prob"], 0.0)
        return

    def aug_coordinates(self, x, s, R, t):
        # N,3; scalar; 3,3; 3
        y = x @ R.T * s + t[None, :]
        return y

    def augment_v2(self, ret):
        random_rot_deg = (np.random.rand() - 0.5) * 2 * self.random_rot_deg
        random_rot_dir = np.random.normal(size=(3,))
        random_rot_dir = random_rot_dir / (np.linalg.norm(random_rot_dir) + 1e-8)
        R = axangle2mat(random_rot_dir, angle=random_rot_deg / 180.0 * np.pi)
        t = (np.random.rand(3) - 0.5) * 2 * self.random_shift_len
        s = (
            np.random.rand() * (self.random_scale_range[1] - self.random_scale_range[0])
            + self.random_scale_range[0]
        )
        ret["points.uni"] = self.aug_coordinates(ret["points.uni"], s, R, t)
        if "points.nss" in ret.keys():
            ret["points.nss"] = self.aug_coordinates(ret["points.nss"], s, R, t)

        if np.random.rand() < self.random_noise_prob:
            input_pcl = ret["inputs"]
            n_random_removal = int(self.aug_ratio * len(input_pcl))
            input_pcl = input_pcl[
                np.random.choice(len(input_pcl), len(input_pcl) - n_random_removal)
            ]
            outliers = (
                (np.random.rand(n_random_removal * 3).reshape(n_random_removal, 3) - 0.5)
                * 2
                * self.random_noise_range
            )
            ret["inputs"] = np.concatenate([input_pcl, outliers], 0)

        ret["inputs"] = self.aug_coordinates(ret["inputs"], s, R, t)

        if self.field_mode == "sdf":
            ret["points.uni.value"] = ret["points.uni.value"] * s
            if "points.nss.value" in ret.keys():
                ret["points.nss.value"] = ret["points.nss.value"] * s

        # # debug
        # np.savetxt("./debug/inputs.txt",ret["inputs"] )
        # np.savetxt("./debug/query.txt",np.concatenate([ret["points.uni"], ret["points.uni.value"][:,None]],-1) )
        return ret

    #####################################################################################

    def aug_v1_config(self, d_cfg):
        # v1
        self.random_object_prob = cfg_with_default(d_cfg, ["random_object_prob"], 0.5)

        self.random_object_radius = cfg_with_default(d_cfg, ["random_object_radius"], 0.1)
        self.random_object_radius_std = cfg_with_default(d_cfg, ["random_object_radius_std"], 0.05)
        self.random_object_center_near_surface = cfg_with_default(
            d_cfg, ["random_object_center_near_surface"], False
        )
        self.random_object_center_L = cfg_with_default(d_cfg, ["random_object_center_L"], 0.5)
        self.random_object_scale = cfg_with_default(
            d_cfg, ["random_object_center_scale"], [0.5, 1.5]
        )

        self.random_plane_prob = cfg_with_default(d_cfg, ["random_plane_prob"], 0.5)
        self.random_plane_vertical_prob = cfg_with_default(
            d_cfg, ["random_plane_vertical_prob"], 0.5
        )
        self.random_plane_vertical_scale = cfg_with_default(
            d_cfg, ["random_plane_vertical_scale"], [0.05, 0.5]
        )
        self.random_plane_vertical_height_range = cfg_with_default(
            d_cfg, ["random_plane_vertical_height_range"], [0.0, 0.5]
        )
        self.random_plane_vertical_horizon_range = cfg_with_default(
            d_cfg, ["random_plane_vertical_horizon_range"], [0.0, 0.5]
        )
        self.random_plane_ground_scale = cfg_with_default(
            d_cfg, ["random_plane_ground_scale"], [0.05, 0.5]
        )
        self.random_plane_ground_range = cfg_with_default(d_cfg, ["random_plane_ground_range"], 0.2)

        self.random_ball_removal_prob = cfg_with_default(d_cfg, ["random_ball_removal_prob"], 0.5)
        self.random_ball_removal_max_k = cfg_with_default(d_cfg, ["random_ball_removal_max_k"], 30)
        self.random_ball_removal_noise_std = cfg_with_default(
            d_cfg, ["random_ball_removal_noise_std"], 0.02
        )

    def ball_removal(self, pcl, n, noise_std):
        anchor = pcl[np.random.choice(len(pcl), 1)]
        d = ((pcl - anchor) ** 2).sum(-1)
        d_noise = np.random.normal(0.0, noise_std, size=len(d))
        d += d_noise
        idx = d.argsort()[:n]
        return idx

    def ball_crop(self, pcl, radius):
        seed = np.random.choice(len(pcl), 1)
        ball_d = np.linalg.norm(pcl - pcl[seed], axis=-1)
        ball_pts = pcl[ball_d < radius]
        return ball_pts

    def augment_v1(self, ret, bottom_y=None):
        ret["inputs"], N_aug = self.__augment_v1(
            ret["inputs"], ret["points.uni"], ret["points.uni.value"], bottom_y=bottom_y
        )
        aug_mask = np.ones(len(ret["inputs"]))
        aug_mask[N_aug:] = 0.0
        ret["inputs_outlier_mask"] = aug_mask
        return ret

    def __augment_v1(self, pcl, points, points_sdf, bottom_y=None):

        N = pcl.shape[0]
        N_aug_max = int(self.aug_ratio * N)
        N_aug = int(np.random.rand() * N_aug_max)
        if N_aug == 0:
            return pcl, 0

        random_seed = np.random.rand(3)
        # random_seed = np.array([0, 0, 0])
        # ! 2022.8.26 permanently changed here, all previous version has bug, all three prob are wrongly placed!!!!!
        aug_mask = random_seed <= np.array(
            [self.random_object_prob, self.random_plane_prob, self.random_ball_removal_prob]
        )
        if not aug_mask.any():
            return pcl, 0
        # ! old bug
        # ! flag_ball, flag_obj, flag_ground = aug_mask
        flag_obj, flag_ground, flag_ball = aug_mask
        if bottom_y is None:
            bottom_y = pcl[:, 1].min()

        total_remove = N_aug
        N_random_noise_fallback = 0
        if flag_obj and flag_ground:
            N_other_obj = int(np.random.rand() * N_aug)
            N_ground = N_aug - N_other_obj
        elif flag_obj:
            N_other_obj = N_aug
            N_ground = 0
        elif flag_ground:
            N_other_obj = 0
            N_ground = N_aug
        else:
            N_other_obj = 0
            N_ground = 0
            N_random_noise_fallback = N_aug

        aug_main_pcl = pcl

        # * random crop out some point cloud
        if flag_ball:
            # remove some ball from the point cloud
            n_ball_removal = int(np.random.rand() * N_aug)
            cnt_ball_removed = 0
            while cnt_ball_removed < n_ball_removal:
                removal_idx = self.ball_removal(
                    aug_main_pcl,
                    min(self.random_ball_removal_max_k, n_ball_removal - cnt_ball_removed),
                    self.random_ball_removal_noise_std,
                )
                cnt_ball_removed += len(removal_idx)
                aug_main_pcl = np.delete(aug_main_pcl, removal_idx, axis=0)
            total_remove -= cnt_ball_removed
        # remove other points
        removal_idx = np.random.choice(len(aug_main_pcl), total_remove, replace=False)
        aug_main_pcl = np.delete(aug_main_pcl, removal_idx, axis=0)

        # * random add some other object's part near outside the object
        AUG_LIST = []
        if N_other_obj > 0:
            cnt_object_added = 0
            other_obj_aug_list = []
            while cnt_object_added < N_other_obj:
                other_obj_id = int(np.random.choice(len(self.meta_list), 1))
                other_obj_id = min(other_obj_id, len(self.meta_list) - 1)
                if self.cache_flag:
                    _, _, pointcloud, _ = self.__read_from_ram__(other_obj_id)
                else:
                    _, _, pointcloud, _ = self.__read_from_disk__(other_obj_id)

                other_pcl = self.ball_crop(
                    pointcloud[np.random.choice(len(pointcloud), self.n_input)],
                    radius=max(
                        self.random_object_radius
                        + np.random.normal(0.0, self.random_object_radius_std),
                        0.01,  # have a least radius
                    ),
                )
                # ! 2022.8.26 permanently change, for both ways of random obj add
                other_pcl = other_pcl - other_pcl.mean(0)[None, ...]
                random_scale = (
                    np.random.rand() * (self.random_object_scale[1] - self.random_object_scale[0])
                    + self.random_object_scale[0]
                )
                other_pcl = random_scale * other_pcl
                other_pcl_r = np.linalg.norm(other_pcl, axis=-1).max()

                for _i in range(100):
                    if self.random_object_center_near_surface:
                        random_center = aug_main_pcl[
                            np.random.choice(len(aug_main_pcl), 1)
                        ] + np.random.normal(loc=0.0, scale=self.random_object_center_L, size=(3))
                    else:
                        random_center = (np.random.rand(3) - 0.5) * 2 * self.random_object_center_L
                        random_center = random_center[None, ...]
                    random_center_d = np.linalg.norm(points - random_center, axis=-1)
                    random_center_nearest_values = points_sdf[random_center_d.argmin()]
                    if random_center_nearest_values > other_pcl_r:
                        break
                random_center = random_center.squeeze(0)
                random_rotation = Rotation.random().as_matrix()

                other_pcl = other_pcl @ random_rotation + random_center[None, :]
                other_obj_aug_list.append(other_pcl)
                cnt_object_added += len(other_pcl)
            AUG_LIST.append(np.concatenate(other_obj_aug_list, 0)[:N_other_obj])
        # * random add the ground or other planes
        if N_ground > 0:
            n_ground = N_ground
            if np.random.rand() < self.random_plane_vertical_prob:
                # if True:
                n_vertical = int(np.random.rand() * N_ground)
                n_ground = N_ground - n_vertical
                random_uv = (np.random.rand(n_vertical * 2).reshape(n_vertical, 2) - 0.5) * 2
                random_scale = (
                    np.random.rand()
                    * (self.random_plane_vertical_scale[1] - self.random_plane_vertical_scale[0])
                    + self.random_plane_vertical_scale[0]
                )
                random_height = (
                    np.random.rand()
                    * (
                        self.random_plane_vertical_height_range[1]
                        - self.random_plane_vertical_height_range[0]
                    )
                    + self.random_plane_vertical_height_range[0]
                )
                vertical_pcl = np.zeros((n_vertical, 3))
                vertical_pcl[:, :2] = random_uv * random_scale
                vertical_pcl[:, 1] += random_height + bottom_y
                random_vertical_rotation = euler2mat(np.random.rand() * np.pi * 2, 0.0, 0.0, "syzx")
                vertical_pcl = (
                    random_vertical_rotation[None, ...] @ vertical_pcl[..., None]
                ).squeeze(-1)
                random_vertical_center_r = (
                    np.random.rand()
                    * (
                        self.random_plane_vertical_horizon_range[1]
                        - self.random_plane_vertical_scale[0]
                    )
                    + self.random_plane_vertical_horizon_range[0]
                )
                random_vertical_center_angle = np.random.rand() * np.pi * 2
                vertical_pcl[:, 0] += (
                    np.cos(random_vertical_center_angle) * random_vertical_center_r
                )
                vertical_pcl[:, 2] += (
                    np.sin(random_vertical_center_angle) * random_vertical_center_r
                )
                AUG_LIST.append(vertical_pcl)
            if n_ground > 0:
                # add ground
                random_uv = (np.random.rand(n_ground * 2).reshape(n_ground, 2) - 0.5) * 2
                random_scale = (
                    np.random.rand()
                    * (self.random_plane_ground_scale[1] - self.random_plane_ground_scale[0])
                    + self.random_plane_ground_scale[0]
                )
                random_center = (np.random.rand(2) - 0.5) * 2 * self.random_plane_ground_range
                ground_pcl = np.zeros((n_ground, 3))
                ground_pcl[:, 1] += bottom_y
                ground_pcl[:, [0, 2]] = random_uv * random_scale + random_center[None, :]
                AUG_LIST.append(ground_pcl)
        if N_random_noise_fallback > 0:
            AUG_LIST.append(
                np.random.rand(N_random_noise_fallback * 3).reshape(N_random_noise_fallback, 3)
                - 0.5
            )
        AUG_LIST.append(aug_main_pcl)
        aug_pcl = np.concatenate(AUG_LIST, 0)
        # # debug
        # np.savetxt("./debug/aug.txt", aug_pcl)
        assert aug_pcl.shape[0] == N
        return aug_pcl, N_aug
