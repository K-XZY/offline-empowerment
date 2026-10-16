# Measuring Empowerment from Offline Data for Intrinsically Motivated Exploration

Code for the paper. It contains

- **our offline empowerment estimator** (Sec. 4), trained on unlabeled [OGBench](https://github.com/seohongpark/ogbench) datasets,
- the four **offline empowerment baselines** (Sec. 5.1 / A.6): CF, CF-FlowBC, DV and DADS-Empowerment,
- **online exploration** with CRL + RLPD, with our empowerment bonus, an RND bonus, or both (Sec. 5.2 / A.3),
- the **SUPE** baseline (OPAL skill pretraining + online SUPE), with or without our empowerment bonus,
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

To train an existing run for longer (or continue after a crash), resume it from its latest checkpoint:

```bash
python train_offline.py --resume_dir=exp/<run_group>/<run_name> --train_steps=<new total>
```

This reuses the run's `flags.json`, appends to its `train.csv` / wandb run and restores the numpy rng, so the result is
the same as one uninterrupted run.

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
| antsoccer-arena    | `antsoccer-arena-less-easy-online-v0`             | 500     | `antsoccer-arena-{navigate,stitch}-v0` |
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

Online runs (CRL and SUPE) write their full train state to `<run dir>/train_state.pkl` every `--checkpoint_interval`
env steps (default: `--save_interval`) and at the end: agent, replay buffer, both envs (incl. their MuJoCo state), all
rngs and the loop counters. To continue a crashed run, or to train a finished one for longer:

```bash
python train_online.py --resume_dir=exp/<run_group>/<run_name> [--total_env_steps=<new total>]
```

This reuses the run's `flags.json`, trims `train.csv` / `eval.csv` to the checkpoint and appends to them (and to its
wandb run), so the result is the same as one uninterrupted run (bitwise on CPU). The annealing fractions and
`--rlpd_frac_time` are fractions of `--total_env_steps`, so with those, extending a run changes its schedule.

### SUPE

SUPE first pretrains OPAL skills on the offline dataset (the skill horizon H must match the online
`skill_commitment_k`), then trains the online high level on those frozen skills:

```bash
python train_offline.py --env_name=antmaze-medium-navigate-v0 --agent=agents/opal.py \
    --agent.chunk_size=$H --agent.sequence_length=$H --agent.kl_coef=$KL --agent.batch_size=256

python train_online.py --agent=agents/supe.py --env_name=antmaze-medium-corner-all-squares-online-v0 \
    --offline_dataset=antmaze-medium-navigate-v0 --agent.skill_checkpoint_path=exp/<opal run> \
    --agent.goal_conditioned=True --agent.aux_schedule=paper --agent.minibatch_split=interleave \
    --agent.gamma_low=0.99 --agent.discount=$GAMMA --agent.num_min_qs=$NUM_MIN_QS --agent.utd_ratio=4

# SUPE + Empowerment (ours): add
#   --agent.add_explore=distill-to-rlpd --agent.emp_checkpoint_path=exp/<estimator run> --agent.bonus_scale=$BETA
```

`goal_conditioned` feeds the task goal to SUPE's actor, critic and reward model (offline rows get random task goals),
`aux_schedule=paper` trains the reward model and RND predictor once per macro step as in SUPE, and `interleave` makes
every critic minibatch half online / half offline.

| Paper (Table 4, SUPE)                    | Flag                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| skill horizon H                          | `--agent.chunk_size` / `--agent.sequence_length` (OPAL), `--agent.skill_commitment_k` (SUPE, default = OPAL's) |
| skill latent dimension, OPAL KL coef     | `--agent.skill_dim`, `--agent.kl_coef` (OPAL)                |
| number of critics / critics in target min | `--agent.num_qs`, `--agent.num_min_qs`                      |
| discount, RND coefficient, warm-up steps | `--agent.discount`, `--agent.rnd_coeff`, `--agent.warmup_env_steps` |

## Code layout

```
train_offline.py            offline estimator / OPAL training
train_online.py             online CRL or SUPE (+ RLPD, + bonuses)
agents/
  empowerment_skill.py      our estimator
  empowerment_crl.py        CF-Empowerment
  empowerment_crl_flowbc.py CF-FlowBC-Empowerment
  empowerment_dv.py         DV-Empowerment
  empowerment_dads.py       DADS-Empowerment
  online_crl.py             online CRL with the empowerment / RND bonus
  opal.py                   OPAL skills (SUPE's offline stage)
  supe.py                   SUPE (+ empowerment bonus)
envs/                       online goal-reaching tasks
utils/                      datasets, networks, replay buffer, RLPD, rollouts, logging
```

This code builds on the [OGBench](https://github.com/seohongpark/ogbench) reference implementations
(`envs/maze.py` is a lightly modified copy of OGBench's maze env), and the OPAL / SUPE agents port
[the SUPE code](https://github.com/rail-berkeley/supe).
