"""Train an empowerment estimator (ours or a baseline), or the OPAL skills for SUPE, on an ogbench dataset.

`--resume_dir=<run dir>` continues a run from its latest checkpoint (e.g. with a larger --train_steps). The run's own
flags.json is replayed, and the numpy rng state is checkpointed too, so a resumed run matches an uninterrupted one.
"""

import glob
import json
import os
import pickle
import random
import re
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset, SequenceDataset
from utils.env_utils import load_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'ogbench dataset name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('resume_dir', None, 'Run dir to resume from its latest checkpoint.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

config_flags.DEFINE_config_file('agent', 'agents/empowerment_skill.py', lock_config=False)

# flags taken from the resumed run's flags.json (unless passed explicitly)
RESUME_FLAGS = ('run_group', 'seed', 'env_name', 'log_interval', 'save_interval')


def main(_):
    config = FLAGS.agent
    resume_step = None
    if FLAGS.resume_dir is not None:
        run_dir = FLAGS.resume_dir.rstrip('/')
        with open(os.path.join(run_dir, 'flags.json')) as f:
            saved = json.load(f)
        config.update({k: tuple(v) if isinstance(v, list) else v for k, v in saved['agent'].items()})
        for key in RESUME_FLAGS:
            if FLAGS[key].using_default_value:
                FLAGS[key].value = saved[key]
        ckpts = glob.glob(os.path.join(run_dir, 'params_*.pkl'))
        resume_step = max(int(re.search(r'params_(\d+)\.pkl', p).group(1)) for p in ckpts)
        if resume_step >= FLAGS.train_steps:
            print(f'{run_dir} is already at step {resume_step}, pass a larger --train_steps')
            return
        exp_name = os.path.basename(run_dir)
        FLAGS.save_dir = run_dir
    else:
        exp_name = get_exp_name(FLAGS.seed)
        FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    if 'log_interval' in config:
        config['log_interval'] = FLAGS.log_interval
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    id_path = os.path.join(FLAGS.save_dir, 'wandb_run_id.txt')
    run_id = open(id_path).read().strip() if resume_step is not None and os.path.exists(id_path) else None
    setup_wandb(group=FLAGS.run_group, name=exp_name, run_id=run_id)
    with open(id_path, 'w') as f:
        f.write(wandb.run.id)
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    train_dataset, val_dataset = load_datasets(FLAGS.env_name)
    dataset_class = SequenceDataset if config.get('dataset_class') == 'SequenceDataset' else GCDataset
    train_dataset = dataset_class(Dataset.create(**train_dataset), config)
    val_dataset = dataset_class(Dataset.create(**val_dataset), config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config
    )
    if resume_step is not None:
        agent = restore_agent(agent, FLAGS.save_dir, resume_step)
        rng_path = os.path.join(FLAGS.save_dir, f'np_rng_{resume_step}.pkl')
        if os.path.exists(rng_path):
            with open(rng_path, 'rb') as f:
                np.random.set_state(pickle.load(f))
        else:
            print(f'no {rng_path}, resuming with a fresh numpy rng (batches will differ from an uninterrupted run)')

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'), resume_step=resume_step)
    first_time = time.time()
    last_time = time.time()
    start_step = 1 if resume_step is None else resume_step + 1
    for i in tqdm.tqdm(range(start_step, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            val_batch = val_dataset.sample(config['batch_size'])
            _, val_info = agent.total_loss(val_batch, grad_params=None)
            train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0 or i == FLAGS.train_steps:
            save_agent(agent, FLAGS.save_dir, i)
            with open(os.path.join(FLAGS.save_dir, f'np_rng_{i}.pkl'), 'wb') as f:
                pickle.dump(np.random.get_state(), f)

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
