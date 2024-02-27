# UDF dense surface point cloud extraction
import logging
import torch
import time
import numpy as np


class Generator3D(object):
    def __init__(
        self,
        sample_base_num=200000,
        num_steps=10,
        accept_th=9e-3,
        df_truncated_th=0.1,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        chunk_size=80000,
        max_global_iter=10,
        verbose=True,
        # **kwargs,
    ) -> None:
        self.device = torch.device("cuda")

        self.sample_base_num = sample_base_num
        self.num_steps = num_steps
        self.accept_th = accept_th
        self.df_truncated_th = df_truncated_th
        self.aabb = aabb
        self.chunk_size = chunk_size
        self.max_global_iter = max_global_iter
        
        self.verbose = verbose

        return

    def generate_from_latent(self, c, F, num_points=30000):
        # from  https://raw.githubusercontent.com/jchibane/ndf/master/models/generation.py

        start = time.time()

        sample_base_num = self.sample_base_num
        num_steps = self.num_steps
        accept_th = self.accept_th
        df_truncated_th = self.df_truncated_th
        aabb = self.aabb
        chunk_size = self.chunk_size
        max_global_iter = self.max_global_iter

        old_grad_context = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        aabb = torch.as_tensor(aabb).to(self.device)  # uniform sample inside
        aabb_size = aabb[1] - aabb[0]

        samples_cpu = np.zeros((0, 3))

        samples = (
            torch.rand(1, sample_base_num, 3).float().to(self.device) * aabb_size[None, None, :]
            + aabb[0][None, None, :]
        )  # 1,N,3
        samples.requires_grad = True

        i = 0
        while len(samples_cpu) < num_points and i < max_global_iter:
            # print("iteration", i)

            for j in range(num_steps):
                # print("refinement", j)
                # todo: there should be a safe chuck
                df_pred = []
                cur = 0
                while cur < samples.shape[1]:
                    _df_pred = torch.clamp(
                        abs(F(samples[:, cur : cur + chunk_size], None, c).logits),
                        max=df_truncated_th, # ! warning, always abs
                    )
                    df_pred.append(_df_pred)
                    cur += chunk_size
                df_pred = torch.cat(df_pred, 1)

                df_pred.sum().backward()

                gradient = samples.grad.detach()
                samples = samples.detach()
                df_pred = df_pred.detach()
                samples = samples - torch.nn.functional.normalize(
                    gradient, dim=2
                ) * df_pred.reshape(
                    -1, 1
                )  # better use Tensor.copy method?
                samples = samples.detach()
                samples.requires_grad = True

            # print("finished refinement")

            if not i == 0:
                samples_cpu = np.vstack(
                    (samples_cpu, samples[df_pred < accept_th].detach().cpu().numpy())
                )

            samples = samples[df_pred < df_truncated_th / 3.0].unsqueeze(
                0
            )  # original code hard-coded 0.03
            indices = torch.randint(samples.shape[1], (1, sample_base_num))
            samples = samples[[[0] * sample_base_num], indices]
            samples += (df_truncated_th / 3) * torch.randn(samples.shape).to(
                self.device
            )  # 3 sigma rule
            samples = samples.detach()
            samples.requires_grad = True

            i += 1
            # print(samples_cpu.shape)

        duration = time.time() - start
        if self.verbose:
            logging.info(
                f"Dense PCL extracted {samples_cpu.shape} pts in {i} iters within {duration:.3f}s"
            )

        torch.set_grad_enabled(old_grad_context)

        if len(samples_cpu) == 0:
            return None

        choice = np.random.choice(len(samples_cpu), num_points, replace=True)
        ret = samples_cpu[choice]

        return ret


def get_generator(cfg):
    return Generator3D(**cfg["generation"]["udf_cfg"])
