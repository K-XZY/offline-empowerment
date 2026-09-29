"""Train an empowerment estimator (ours or a baseline) on an ogbench dataset."""

import json
import os
import random
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from utils.datasets import Dataset, GCDataset
from utils.env_utils import load_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'ogbench dataset name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 100000, 'Saving interval.')

config_flags.DEFINE_config_file('agent', 'agents/empowerment_skill.py', lock_config=False)


def main(_):
    config = FLAGS.agent
    if 'log_interval' in config:
        config['log_interval'] = FLAGS.log_interval

    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(group=FLAGS.run_group, name=exp_name)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(get_flag_dict(), f)

    train_dataset, val_dataset = load_datasets(FLAGS.env_name)
    train_dataset = GCDataset(Dataset.create(**train_dataset), config)
    val_dataset = GCDataset(Dataset.create(**val_dataset), config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent = agents[config['agent_name']].create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config
    )
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
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

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
