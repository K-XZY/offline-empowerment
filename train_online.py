"""Online goal-reaching with CRL or SUPE (+ RLPD), optionally with the empowerment and / or RND exploration bonus.

Every `unroll_length` replay rows (env steps for CRL, macro steps for SUPE), once `min_replay_size` rows are in the
buffer, we run `unroll_length * utd_ratio` updates. Logging / eval / checkpoints are keyed on env steps.

Every `checkpoint_interval` env steps and at the end of the run, the full train state (agent, replay buffer, collector
and both envs incl. their MuJoCo state, numpy / python / jax rngs, loop counters, csv row counts) goes to
<run dir>/train_state.pkl, overwriting the previous one. `--resume_dir=<run dir>` continues from it, also with a larger
--total_env_steps to extend a finished run; the result is the same as one uninterrupted run (up to GPU
nondeterminism). The run's flags come from its flags.json.
"""

import json
import os
import pickle
import random
import time
from collections import defaultdict

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.env_utils import env_horizon, get_env_state, make_example_batch, make_online_env, set_env_state
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.replay_buffer import TrajectoryReplayBuffer
from utils.rlpd import BufferSource, GoalBankSource, MixedBatchSampler, make_offline_source
from utils.rollout import Collector, MacroCollector, evaluate, example_transition

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
flags.DEFINE_integer('checkpoint_interval', None, 'Train-state checkpoint interval (env steps, None -> save_interval).')
flags.DEFINE_string('resume_dir', None, 'Run dir to continue from its train_state.pkl (optionally to a new total).')

config_flags.DEFINE_config_file('agent', 'agents/online_crl.py', lock_config=False)


TRAIN_STATE = 'train_state.pkl'
# flags a resumed run takes from its flags.json (passing a different value is an error: it wouldn't be the same run)
RESUME_FLAGS = (
    'run_group',
    'seed',
    'env_name',
    'offline_dataset',
    'episode_length',
    'rlpd_frac_time',
    'log_interval',
    'eval_interval',
    'save_interval',
    'eval_episodes',
    'checkpoint_interval',
)


@jax.jit
def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def load_resume_flags(run_dir):
    """Replay the run's flags.json (agent config included); --total_env_steps defaults to the run's own."""
    with open(os.path.join(run_dir, 'flags.json')) as f:
        saved = json.load(f)
    for key in RESUME_FLAGS:
        if key not in saved:  # runs from before the flag existed
            continue
        if not FLAGS[key].using_default_value and FLAGS[key].value != saved[key]:
            raise ValueError(f'--{key}={FLAGS[key].value} differs from the resumed run ({saved[key]})')
        FLAGS[key].value = saved[key]
    if FLAGS['total_env_steps'].using_default_value:
        FLAGS.total_env_steps = saved['total_env_steps']
    elif FLAGS.total_env_steps != saved['total_env_steps']:
        anneal = [k for k in ('explore_reward_time_frac', 'rnd_time_frac') if saved['agent'].get(k) is not None]
        if anneal or FLAGS.rlpd_frac_time < 1.0:
            print(
                f'[train_online] WARNING: {anneal + ["rlpd_frac_time"] * (FLAGS.rlpd_frac_time < 1.0)} are fractions '
                f'of total_env_steps, which changes {saved["total_env_steps"]} -> {FLAGS.total_env_steps}'
            )
    config = FLAGS.agent
    for key in list(config.keys()):
        del config[key]
    config.update({k: tuple(v) if isinstance(v, list) else v for k, v in saved['agent'].items()})


def main(_):
    resume_state = None
    if FLAGS.resume_dir is not None:
        run_dir = FLAGS.resume_dir.rstrip('/')
        load_resume_flags(run_dir)
        with open(os.path.join(run_dir, TRAIN_STATE), 'rb') as f:
            resume_state = pickle.load(f)
        if resume_state['env_steps'] >= FLAGS.total_env_steps:
            print(f'{run_dir} is already at env step {resume_state["env_steps"]}, pass a larger --total_env_steps')
            return
        print(f'[train_online] resuming {run_dir} from env step {resume_state["env_steps"]}')
        exp_name = os.path.basename(run_dir)
        FLAGS.save_dir = run_dir
    else:
        exp_name = get_exp_name(FLAGS.seed)
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    id_path = os.path.join(FLAGS.save_dir, 'wandb_run_id.txt')
    run_id = None
    if resume_state is not None and os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.read().strip()
    setup_wandb(group=FLAGS.run_group, name=exp_name, run_id=run_id)
    with open(id_path, 'w') as f:
        f.write(wandb.run.id)
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)
    checkpoint_interval = FLAGS.checkpoint_interval or FLAGS.save_interval

    config = FLAGS.agent
    config['total_env_steps'] = FLAGS.total_env_steps
    env = make_online_env(FLAGS.env_name, episode_length=FLAGS.episode_length)
    eval_env = make_online_env(FLAGS.env_name, episode_length=FLAGS.episode_length)
    horizon = env_horizon(env)
    assert horizon is not None, 'env has no TimeLimit, pass --episode_length'
    assert 0.0 <= FLAGS.rlpd_frac_time <= 1.0

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = make_example_batch(env)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config
    )

    # schedule keys come from the created agent (supe fills in some defaults from its skill horizon)
    config = agent.config
    macro = config['agent_name'] == 'supe'
    row_layout = example_transition(example_batch, agent)
    buffer = TrajectoryReplayBuffer.create(row_layout, int(config['replay_size']))
    if macro:
        k = int(config['skill_commitment_k'])
        assert horizon % k == 0, f'episode horizon {horizon} must be divisible by skill_commitment_k {k}'
        assert buffer.capacity > horizon // k + 1
        collector = MacroCollector(
            env, buffer, FLAGS.seed, k, config['gamma_low'], config['reward_shift'], agent.stores_task_goals
        )
    else:
        assert buffer.capacity > horizon + 1
        collector = Collector(env, buffer, seed=FLAGS.seed)

    unroll_length = int(config['unroll_length'])
    updates_per_round = int(unroll_length * float(config['utd_ratio']))
    assert updates_per_round >= 1, 'unroll_length * utd_ratio rounds down to 0 updates per round'
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
        offline_source = make_offline_source(FLAGS.offline_dataset, agent, row_layout, seed=FLAGS.seed)
        if getattr(agent, 'stores_task_goals', False):
            # goal-conditioned SUPE: offline rows get goals from the env's own tasks (a few resets each, since a
            # task's goal observation can vary between resets)
            bank_env = make_online_env(FLAGS.env_name, episode_length=FLAGS.episode_length)
            resets = int(config['goal_bank_resets_per_task'])
            goal_bank = np.stack(
                [
                    np.asarray(bank_env.reset(options=dict(task_id=task_id))[1]['goal'], dtype=np.float32)
                    for task_id in range(1, int(bank_env.unwrapped.num_tasks) + 1)
                    for _ in range(resets)
                ]
            )
            bank_env.close()
            offline_source = GoalBankSource(
                offline_source.buffer, offline_source.discount, offline_source.next_offset, goal_bank=goal_bank
            )
        # rlpd_frac_time=1: rlpd for the whole run, however long (no switch at the very end, so --resume_dir can
        # extend a finished run exactly)
        rlpd_end_step = (
            None if FLAGS.rlpd_frac_time >= 1.0 else int(round(FLAGS.rlpd_frac_time * FLAGS.total_env_steps))
        )
        if rlpd_end_step is None or rlpd_end_step > 0:
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

    env_steps = 0
    num_updates = 0
    rows_since_update = 0
    round_infos, round_num_updates = None, 0  # update metrics summed on device, averaged at log time
    stream_infos = defaultdict(list)
    aux_keys = {}  # supe aux_schedule='paper': metric names of update_rm / update_rnd
    episode_stats = defaultdict(list)
    next_log, next_eval, next_save = FLAGS.log_interval, 0, FLAGS.save_interval
    next_checkpoint = checkpoint_interval
    last_saved_step, last_eval_step, last_log_step = None, None, 0
    elapsed = 0.0
    csv_rows = dict(train=None, eval=None)

    if resume_state is not None:
        s = resume_state
        agent = flax.serialization.from_state_dict(agent, s['agent'])
        agent = jax.tree_util.tree_map(jnp.asarray, agent)
        buffer.load_state_dict(s['buffer'])
        collector.load_state_dict(s['collector'])
        set_env_state(eval_env, s['eval_env'])
        if s['goal_bank'] is not None:
            offline_source.goal_bank = s['goal_bank']
        env_steps, num_updates, rows_since_update = s['env_steps'], s['num_updates'], s['rows_since_update']
        round_infos, round_num_updates = s['round_infos'], s['round_num_updates']
        stream_infos = defaultdict(list, s['stream_infos'])
        aux_keys = s['aux_keys']
        episode_stats = defaultdict(list, s['episode_stats'])
        next_log, next_eval = s['next_log'], s['next_eval']
        next_save, next_checkpoint = s['next_save'], s['next_checkpoint']
        last_saved_step, last_eval_step, last_log_step = s['last_saved_step'], s['last_eval_step'], s['last_log_step']
        elapsed = s['elapsed']
        csv_rows = s['csv_rows']
        # derived from the (possibly new) total, so extending a run keeps rlpd on like an uninterrupted run
        if sampler is not online_sampler and rlpd_end_step is not None and env_steps >= rlpd_end_step:
            sampler = online_sampler
        # last, after everything above that draws from them
        np.random.set_state(s['np_random'])
        random.setstate(s['py_random'])
        del resume_state, s

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'), resume_rows=csv_rows['train'])
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'), resume_rows=csv_rows['eval'])
    last_time = time.time()
    first_time = last_time - elapsed

    def save_train_state():
        state = dict(
            env_steps=env_steps,
            num_updates=num_updates,
            rows_since_update=rows_since_update,
            round_infos=jax.device_get(round_infos),
            round_num_updates=round_num_updates,
            stream_infos=dict(stream_infos),
            aux_keys=aux_keys,
            episode_stats=dict(episode_stats),
            next_log=next_log,
            next_eval=next_eval,
            next_save=next_save,
            next_checkpoint=next_checkpoint,
            last_saved_step=last_saved_step,
            last_eval_step=last_eval_step,
            last_log_step=last_log_step,
            elapsed=time.time() - first_time,
            csv_rows=dict(train=train_logger.num_rows, eval=eval_logger.num_rows),
            agent=jax.device_get(flax.serialization.to_state_dict(agent)),
            buffer=buffer.state_dict(),
            collector=collector.state_dict(),
            eval_env=get_env_state(eval_env),
            goal_bank=getattr(offline_source, 'goal_bank', None),
            np_random=np.random.get_state(),
            py_random=random.getstate(),
        )
        path = os.path.join(FLAGS.save_dir, TRAIN_STATE)
        with open(f'{path}.tmp', 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(f'{path}.tmp', path)
        print(f'[train_online] train state at env step {env_steps} -> {path}')

    def run_eval():
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
        eval_metrics = {f'evaluation/{k}': v for k, v in evaluate(eval_agent, eval_env, FLAGS.eval_episodes).items()}
        wandb.log(eval_metrics, step=env_steps)
        eval_logger.log(eval_metrics, step=env_steps)

    pbar = tqdm.tqdm(total=FLAGS.total_env_steps, initial=env_steps, smoothing=0.1, dynamic_ncols=True)
    while env_steps < FLAGS.total_env_steps:
        if env_steps >= next_eval:
            run_eval()
            last_eval_step = env_steps
            next_eval += FLAGS.eval_interval

        steps, episode = collector.step(agent)
        env_steps += steps
        rows_since_update += 1
        pbar.update(steps)
        if episode is not None:
            for k, v in episode.items():
                episode_stats[k].append(v)

        if sampler is not online_sampler and rlpd_end_step is not None and env_steps >= rlpd_end_step:
            sampler = online_sampler
            print(f'[train_online] env step {env_steps}: switching to online-only batches')

        if buffer.num_valid >= int(config['min_replay_size']) and rows_since_update >= unroll_length:
            rows_since_update = 0
            if agent.uses_empowerment:
                collector.flush_empowerment(agent)
                if agent.emp_mean is None:
                    agent = set_empowerment_mean(agent)
            if getattr(agent, 'uses_rnd', False):
                agent, rnd_info = collector.flush_rnd(agent)
                for k, v in rnd_info.items():
                    stream_infos[f'rnd_stream/{k}'].append(v)
            for _ in range(updates_per_round):
                agent, update_info = agent.update(sampler.sample(batch_size), env_steps=env_steps)
                num_updates += 1
                round_infos = update_info if round_infos is None else _tree_add(round_infos, update_info)
                round_num_updates += 1
            if macro and config['aux_schedule'] == 'paper':
                # SUPE: reward model on a batch of online rows and RND on the newest transition, once per macro step
                rm_batch = online_sampler.sample(int(config['rm_updates_per_macro']) * int(config['rm_batch_size']))
                aux_calls = [
                    ('rm', int(config['rm_start_env_steps']), lambda a: a.update_rm(rm_batch)),
                    ('rnd', int(config['rnd_start_env_steps']), lambda a: a.update_rnd(collector.last_transition)),
                ]
                for name, start, call in aux_calls:
                    if env_steps >= start:
                        agent, aux_info = call(agent)
                        aux_keys[name] = list(aux_info)
                        for k, v in aux_info.items():
                            stream_infos[k].append(v)
                    else:
                        # log zeros until it starts so the csv header has the columns
                        if name not in aux_keys:
                            aux_keys[name] = list(jax.eval_shape(call, agent)[1])
                        for k in aux_keys[name]:
                            stream_infos[k].append(0.0)

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

        if env_steps >= next_checkpoint:
            next_checkpoint += checkpoint_interval
            save_train_state()
    pbar.close()

    # the eval an uninterrupted longer run would do next, then the train state (so --resume_dir can extend this run
    # exactly), then a final eval if the last one is older (not part of the train state: a resume drops its eval.csv
    # row)
    if env_steps >= next_eval:
        run_eval()
        last_eval_step = env_steps
        next_eval += FLAGS.eval_interval
    save_train_state()
    if last_eval_step != env_steps:
        run_eval()
    if last_saved_step != env_steps:
        save_agent(agent, FLAGS.save_dir, env_steps)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
