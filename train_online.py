"""Online goal-reaching with CRL (+ RLPD), optionally with the empowerment and / or RND exploration bonus.

Every `unroll_length` env steps (after `min_replay_size` warm-up steps) we run `unroll_length * utd_ratio`
updates. Logging / eval / checkpoints are keyed on env steps.
"""

import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.env_utils import env_horizon, make_example_batch, make_online_env
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.replay_buffer import TrajectoryReplayBuffer
from utils.rlpd import BufferSource, MixedBatchSampler, make_offline_source
from utils.rollout import Collector, evaluate, example_transition

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-corner-all-squares-online-v0', 'Online env (see envs/__init__.py).')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('total_env_steps', 1000000, 'Total number of env steps.')
flags.DEFINE_integer('episode_length', None, 'Episode horizon override (None -> registered one).')
flags.DEFINE_string('offline_dataset', None, 'RLPD: ogbench dataset mixed into every batch (None -> online only).')
flags.DEFINE_float('rlpd_frac_time', 1.0, 'RLPD is only used for this fraction of total_env_steps.')
flags.DEFINE_integer('log_interval', 25000, 'Logging interval (env steps).')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval (env steps).')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval (env steps).')
flags.DEFINE_integer('eval_episodes', 100, 'Number of random-task eval episodes.')

config_flags.DEFINE_config_file('agent', 'agents/online_crl.py', lock_config=False)


@jax.jit
def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(group=FLAGS.run_group, name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    config = FLAGS.agent
    config['total_env_steps'] = FLAGS.total_env_steps
    env = make_online_env(FLAGS.env_name, episode_length=FLAGS.episode_length)
    eval_env = make_online_env(FLAGS.env_name, episode_length=FLAGS.episode_length)
    horizon = env_horizon(env)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = make_example_batch(env)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config
    )

    buffer = TrajectoryReplayBuffer.create(example_transition(example_batch), int(config['replay_size']))
    assert buffer.capacity > horizon + 1
    collector = Collector(env, buffer, seed=FLAGS.seed)

    unroll_length = int(config['unroll_length'])
    updates_per_round = int(unroll_length * float(config['utd_ratio']))
    batch_size = int(config['batch_size'])

    # RLPD for the first rlpd_frac_time of the run, online-only afterwards
    online_sampler = BufferSource(buffer, discount=float(config['discount']))
    sampler = online_sampler
    rlpd_end_step = 0
    offline_source = None
    if FLAGS.offline_dataset is not None:
        if agent.uses_empowerment:
            with open(os.path.join(config['emp_checkpoint_path'], 'flags.json')) as f:
                emp_env = json.load(f)['env_name']
            if emp_env != FLAGS.offline_dataset:
                print(f'[train_online] WARNING: estimator was trained on {emp_env}, rlpd uses {FLAGS.offline_dataset}')
        offline_source = make_offline_source(
            FLAGS.offline_dataset, agent, example_transition(example_batch), seed=FLAGS.seed
        )
        rlpd_end_step = int(round(FLAGS.rlpd_frac_time * FLAGS.total_env_steps))
        if rlpd_end_step > 0:
            sampler = MixedBatchSampler(online_sampler, offline_source, float(config['offline_ratio']))

    def set_empowerment_mean(agent):
        # constant shift of the bonus target, doesn't change the actor gradient
        if agent.config['emp_mean'] is not None:
            mean = float(agent.config['emp_mean'])
        elif agent.config['emp_mean_metric'] is not None:
            mean = float(agent.config['emp_mean_metric'])
        elif offline_source is not None:
            mean = float(np.mean(offline_source.buffer.valid_field('empowerment')))
        else:
            mean = float(np.mean(buffer.valid_field('empowerment')))
        print(f'[train_online] E_mean = {mean:.4f}')
        return agent.with_empowerment_mean(mean)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = last_time = time.time()
    last_log_step = 0

    env_steps = 0
    num_updates = 0
    rows_since_update = 0
    round_infos, round_num_updates = None, 0  # update metrics summed on device, averaged at log time
    stream_infos = defaultdict(list)
    episode_stats = defaultdict(list)
    next_log, next_eval, next_save = FLAGS.log_interval, 0, FLAGS.save_interval
    last_saved_step = None

    pbar = tqdm.tqdm(total=FLAGS.total_env_steps, smoothing=0.1, dynamic_ncols=True)
    while env_steps < FLAGS.total_env_steps:
        if env_steps >= next_eval:
            eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            eval_metrics = {
                f'evaluation/{k}': v for k, v in evaluate(eval_agent, eval_env, FLAGS.eval_episodes).items()
            }
            wandb.log(eval_metrics, step=env_steps)
            eval_logger.log(eval_metrics, step=env_steps)
            next_eval += FLAGS.eval_interval

        episode = collector.step(agent)
        env_steps += 1
        rows_since_update += 1
        pbar.update(1)
        if episode is not None:
            for k, v in episode.items():
                episode_stats[k].append(v)

        if sampler is not online_sampler and env_steps >= rlpd_end_step:
            sampler = online_sampler
            print(f'[train_online] env step {env_steps}: switching to online-only batches')

        if buffer.num_valid >= int(config['min_replay_size']) and rows_since_update >= unroll_length:
            rows_since_update = 0
            if agent.uses_empowerment:
                collector.flush_empowerment(agent)
                if agent.emp_mean is None:
                    agent = set_empowerment_mean(agent)
            if agent.uses_rnd:
                agent, rnd_info = collector.flush_rnd(agent)
                for k, v in rnd_info.items():
                    stream_infos[f'rnd_stream/{k}'].append(v)
            for _ in range(updates_per_round):
                agent, update_info = agent.update(sampler.sample(batch_size), env_steps=env_steps)
                num_updates += 1
                round_infos = update_info if round_infos is None else _tree_add(round_infos, update_info)
                round_num_updates += 1

        if env_steps >= next_log:
            next_log += FLAGS.log_interval
            if num_updates > 0:  # csv header is fixed by the first row, so wait for the first updates
                sums = jax.device_get(round_infos) if round_infos is not None else {}
                train_metrics = {f'training/{k}': float(np.mean(v) / round_num_updates) for k, v in sums.items()}
                train_metrics.update({f'training/{k}': float(np.mean(v)) for k, v in stream_infos.items()})
                train_metrics.update({f'training/{k}': float(np.mean(v)) for k, v in episode_stats.items()})
                train_metrics['training/num_episodes'] = float(len(episode_stats['episode_length']))
                train_metrics['training/env_steps'] = env_steps
                train_metrics['training/num_updates'] = num_updates
                train_metrics['training/rlpd_active'] = float(sampler is not online_sampler)
                train_metrics['time/sps'] = (env_steps - last_log_step) / max(time.time() - last_time, 1e-6)
                train_metrics['time/total_time'] = time.time() - first_time
                last_time, last_log_step = time.time(), env_steps
                round_infos, round_num_updates = None, 0
                stream_infos, episode_stats = defaultdict(list), defaultdict(list)
                wandb.log(train_metrics, step=env_steps)
                train_logger.log(train_metrics, step=env_steps)

        if env_steps >= next_save:
            save_agent(agent, FLAGS.save_dir, env_steps)
            last_saved_step = env_steps
            next_save += FLAGS.save_interval
    pbar.close()

    eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
    eval_metrics = {f'evaluation/{k}': v for k, v in evaluate(eval_agent, eval_env, FLAGS.eval_episodes).items()}
    wandb.log(eval_metrics, step=env_steps)
    eval_logger.log(eval_metrics, step=env_steps)
    if last_saved_step != env_steps:
        save_agent(agent, FLAGS.save_dir, env_steps)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
