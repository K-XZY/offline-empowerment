# Measuring Empowerment from Offline Data for Intrinsically Motivated Exploration

Code for the paper. It contains

- **our offline empowerment estimator** (Sec. 4), trained on unlabeled [OGBench](https://github.com/seohongpark/ogbench) datasets,
- the four **offline empowerment baselines** (Sec. 5.1 / A.6): CF, CF-FlowBC, DV and DADS-Empowerment,
- **online exploration** with CRL + RLPD, with our empowerment bonus, an RND bonus, or both (Sec. 5.2 / A.3),
- the four **online goal-reaching tasks** (A.3.1).

Hyperparameters are in the paper (Tables 1, 3–5 and A.6). Below we list which flag each one maps to.

## Installation

We use [uv](https://docs.astral.sh/uv/):

```bash
uv sync                # cpu
uv sync --extra cuda   # gpu (jax[cuda12])
```

Then run everything from the repo root with `uv run python ...` (or activate `.venv`). `.python-version` pins
python 3.11, which is what we tested with. With pip, install the packages listed in
`pyproject.toml` (`ogbench==1.2.1`, `jax`, `flax`, `distrax`, `optax`, `ml_collections`, `absl-py`, `wandb`, ...).

Datasets are downloaded by ogbench to `~/.ogbench/data` on first use (set `OGBENCH_DATASET_DIR` to change that).
Runs log to wandb; set `WANDB_MODE=offline` or `WANDB_MODE=disabled` to turn that off. Metrics are also written to
`train.csv` / `eval.csv` in the run directory. On a headless machine you may need `MUJOCO_GL=egl`.

## Offline empowerment

```bash
python train_offline.py --env_name=antmaze-medium-navigate-v0 --agent=agents/empowerment_skill.py \
    --agent.num_skills=$K --agent.action_noise_std=$SIGMA_A --agent.bc_alpha=$ALPHA_BC
```

Runs go to `exp/<run_group>/<run_name>/` (`flags.json`, `params_<step>.pkl`, `train.csv`). The logged
`training/empowerment/*` and `validation/empowerment/*` are the estimate Ê(s) (eq. 15) on a batch. For any trained
estimator (ours or a baseline), `agent.empowerment(observations, rng)` returns Ê(s) at arbitrary states; the
baselines' logged `mi_estimate` is their bound on dataset samples, not this.

| Paper (Table 1)                    | Flag                                   |
| ---------------------------------- | -------------------------------------- |
| number of skills K                 | `--agent.num_skills`                   |
| action noise std σ_a               | `--agent.action_noise_std`             |
| behavioral constraint α_bc         | `--agent.bc_alpha`                     |
| value latent dim d                 | `--agent.value_latent_dim`             |
| Q/V, policy width/depth            | `--agent.hidden_dims`, `--agent.actor_hidden_dims` |
| future-state samples per update    | `--agent.num_splus_samples`            |
| γ, τ, lr, batch size               | `--agent.discount`, `--agent.tau`, `--agent.lr`, `--agent.batch_size` |
| training steps                     | `--train_steps`                        |

### Baselines

Same script, different `--agent`:

| Paper                  | Agent                                |
| ---------------------- | ------------------------------------ |
| CF-Empowerment         | `agents/empowerment_crl.py`          |
| CF-FlowBC-Empowerment  | `agents/empowerment_crl_flowbc.py`   |
| DV-Empowerment         | `agents/empowerment_dv.py`           |
| DADS-Empowerment       | `agents/empowerment_dads.py` (λ_kl is `--agent.kl_coef`) |

## Online exploration

The online tasks (A.3.1) are registered in `envs/`. Use them with `online` in place of the dataset type (the bare
registered ids, e.g. `antmaze-medium-corner-all-squares-v0`, won't work as `--env_name`):

| Task               | `--env_name`                                      | Horizon | `--offline_dataset` (RLPD) |
| ------------------ | ------------------------------------------------- | ------- | -------------------------- |
| antmaze-medium     | `antmaze-medium-corner-all-squares-online-v0`     | 1000    | `antmaze-medium-{navigate,stitch,explore}-v0` |
| pointmaze-teleport | `pointmaze-teleport-corner-all-squares-online-v0` | 1000    | `pointmaze-teleport-{navigate,stitch}-v0` |
| antsoccer-arena    | `antsoccer-arena-easy-online-v0`                  | 500     | `antsoccer-arena-{navigate,stitch}-v0` |
| cube-single        | `cube-single-multigoal-online-v0`                 | 200     | `cube-single-{play,noisy}-v0` |

First train our estimator on the same dataset used for RLPD (above), then:

```bash
# CRL
python train_online.py --env_name=antmaze-medium-corner-all-squares-online-v0 --offline_dataset=antmaze-medium-navigate-v0

# CRL + Empowerment (ours)
python train_online.py --env_name=... --offline_dataset=... \
    --agent.add_explore=distill-to-rlpd --agent.emp_checkpoint_path=exp/<run_group>/<run_name> --agent.bonus_scale=$BETA

# CRL + RND
python train_online.py --env_name=... --offline_dataset=... --agent.rnd_bonus_scale=$BETA_RND

# CRL + RND + Empowerment: pass both
```

| Paper (A.3, Tables 3–5)                    | Flag                                         |
| ------------------------------------------ | -------------------------------------------- |
| empowerment bonus weight β                 | `--agent.bonus_scale`                        |
| RND bonus weight β_RND                     | `--agent.rnd_bonus_scale`                    |
| future-state samples per skill for Ê       | `--agent.emp_num_splus_samples`              |
| offline fraction of each batch (RLPD)      | `--agent.offline_ratio`                      |
| target entropy multiplier, replay size     | `--agent.target_entropy_multiplier`, `--agent.replay_size` |
| total env steps                            | `--total_env_steps`                          |
| env steps per update round / updates per env step / warm-up steps | `--agent.unroll_length`, `--agent.utd_ratio`, `--agent.min_replay_size` |
| RND nets, predictor lr, Q_RND discount / τ | `--agent.rnd_hidden_dims`, `--agent.rnd_rep_dim`, `--agent.rnd_lr`, `--agent.bonus_discount`, `--agent.bonus_tau` |

`--agent.add_explore=distill-to-rlpd` regresses the bonus network B_ω on both online and offline trajectories (eq. 16),
`distill` uses online trajectories only. `--agent.explore_reward_time_frac` / `--agent.rnd_time_frac` optionally
anneal β / β_RND linearly to 0 over that fraction of training, and `--rlpd_frac_time` stops mixing in offline data
after that fraction of training.

## Code layout

```
train_offline.py            offline estimator training
train_online.py             online CRL (+ RLPD, + bonuses)
agents/
  empowerment_skill.py      our estimator
  empowerment_crl.py        CF-Empowerment
  empowerment_crl_flowbc.py CF-FlowBC-Empowerment
  empowerment_dv.py         DV-Empowerment
  empowerment_dads.py       DADS-Empowerment
  online_crl.py             online CRL with the empowerment / RND bonus
envs/                       online goal-reaching tasks
utils/                      datasets, networks, replay buffer, RLPD, rollouts, logging
```

This code builds on the [OGBench](https://github.com/seohongpark/ogbench) reference implementations
(`envs/maze.py` is a lightly modified copy of OGBench's maze env).
