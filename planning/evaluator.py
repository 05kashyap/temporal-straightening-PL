import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        decode_for_viz=True,
        chunk_size=None,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.decode_for_viz = bool(decode_for_viz)
        self.chunk_size = chunk_size
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Episodes are processed in chunks of self.chunk_size so GPU memory
        (world-model rollouts) and CPU/worker memory (env rollouts) stay bounded
        regardless of the total number of evals. The aggregated metrics are the
        same as a single full-batch call would produce.
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        cs = self.chunk_size or n_evals
        cs = max(1, min(int(cs), n_evals))

        e_obses_list, e_states_list = [], []
        i_final_z_obs_list = []
        e_final_obs_list, e_final_state_list = [], []
        eval_results_list = []
        visual_dists_list, proprio_dists_list = [], []
        div_visual_sq, div_proprio_sq = 0.0, 0.0
        i_z_obses_first = None  # only needed for visualization; bounded to the first chunk

        for start in range(0, n_evals, cs):
            end = min(start + cs, n_evals)
            chunk_obs_0 = {k: v[start:end] for k, v in self.obs_0.items()}
            chunk_obs_g = {k: v[start:end] for k, v in self.obs_g.items()}
            chunk_state_0 = self.state_0[start:end]
            chunk_state_g = self.state_g[start:end]
            chunk_len = action_len[start:end]
            chunk_actions = actions[start:end]

            # rollout in wm (bounded GPU memory per chunk)
            trans_obs_0 = move_to_device(
                self.preprocessor.transform_obs(chunk_obs_0), self.device
            )
            with torch.no_grad():
                i_z_obses, _ = self.wm.rollout(
                    obs_0=trans_obs_0,
                    act=chunk_actions,
                )
            if start == 0:
                i_z_obses_first = i_z_obses
            i_final_z_obs = self._get_trajdict_last(i_z_obses, chunk_len + 1)

            # rollout in env (chunk size matches the vector env's worker count)
            exec_actions = rearrange(
                chunk_actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
            )
            exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
            e_obses, e_states = self.env.rollout(
                self.seed[start:end], chunk_state_0, exec_actions
            )
            e_final_obs = self._get_trajdict_last(
                e_obses, chunk_len * self.frameskip + 1
            )
            e_final_state = self._get_traj_last(
                e_states, chunk_len * self.frameskip + 1
            )[:, 0]  # reduce dim back

            # compute per-episode metrics for this chunk (goals point at the chunk)
            orig_obs_g, orig_state_g = self.obs_g, self.state_g
            self.obs_g, self.state_g = chunk_obs_g, chunk_state_g
            eval_results, visual_dists, proprio_dists, dv, dp = (
                self._compute_rollout_metrics(
                    e_state=e_final_state,
                    e_obs=e_final_obs,
                    i_z_obs=i_final_z_obs,
                )
            )
            self.obs_g, self.state_g = orig_obs_g, orig_state_g

            eval_results_list.append(eval_results)
            visual_dists_list.append(visual_dists)
            proprio_dists_list.append(proprio_dists)
            div_visual_sq += dv * dv
            div_proprio_sq += dp * dp
            e_obses_list.append(e_obses)
            e_states_list.append(e_states)
            i_final_z_obs_list.append(i_final_z_obs)
            e_final_obs_list.append(e_final_obs)
            e_final_state_list.append(e_final_state)

        # aggregate across chunks (identical to a single full-batch evaluation)
        eval_results_all = {
            k: np.concatenate([d[k] for d in eval_results_list], axis=0)
            for k in eval_results_list[0]
        }
        successes = eval_results_all["success"]
        visual_dists = np.concatenate(visual_dists_list)
        proprio_dists = np.concatenate(proprio_dists_list)
        e_obses = {
            k: np.concatenate([d[k] for d in e_obses_list], axis=0)
            for k in e_obses_list[0]
        }
        e_states = np.concatenate(e_states_list, axis=0)
        i_final_z_obs = {
            k: torch.cat([d[k] for d in i_final_z_obs_list], dim=0)
            for k in i_final_z_obs_list[0]
        }
        e_final_obs = {
            k: np.concatenate([d[k] for d in e_final_obs_list], axis=0)
            for k in e_final_obs_list[0]
        }
        e_final_state = np.concatenate(e_final_state_list, axis=0)

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value)
            if key != "success"
            else np.mean(value.astype(float))
            for key, value in eval_results_all.items()
        }
        logs.update({
            "mean_visual_dist": np.mean(visual_dists),
            "mean_proprio_dist": np.mean(proprio_dists),
            "mean_div_visual_emb": np.sqrt(div_visual_sq),
            "mean_div_proprio_emb": np.sqrt(div_proprio_sq),
        })
        print("Success rate: ", logs["success_rate"])
        print(eval_results_all)

        # plot trajs (only the first chunk is decoded, matching n_plot_samples)
        if self.decode_for_viz and self.wm.decoder is not None:
            with torch.no_grad():
                i_visuals = self.wm.decode_obs(i_z_obses_first)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len[: self.n_plot_samples] + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_obses["visual"])
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Per-episode eval metrics for the current (possibly chunked) batch.
        Args
            e_state
            e_obs
            i_z_obs
        Return
            eval_results: dict of per-episode arrays (e.g. success, state_dist)
            visual_dists: (n,) per-episode
            proprio_dists: (n,) per-episode
            div_visual_emb: float, L2 norm over the batch
            div_proprio_emb: float, L2 norm over the batch
        Uses self.state_g / self.obs_g as the goals for the current batch.
        """
        eval_results = self.env.eval_state(self.state_g, e_state)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        with torch.no_grad():
            e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        return (
            eval_results,
            visual_dists,
            proprio_dists,
            div_visual_emb,
            div_proprio_emb,
        )

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
