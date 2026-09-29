import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import wandb


class CsvLogger:
    """Logs metric dicts to a csv file (header fixed by the first row)."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        if self.file is None:
            self.file = open(self.path, 'w')
            self.header = list(row.keys())
            self.file.write(','.join(self.header) + '\n')
        self.file.write(','.join([str(row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def _wandb_settings():
    # wandb renamed / dropped some settings across versions, only pass the ones it knows about
    wanted = dict(start_method='thread', _disable_stats=False, x_disable_stats=False)
    fields = getattr(wandb.Settings, 'model_fields', None)
    if fields is None:
        wanted.pop('x_disable_stats')
    else:
        wanted = {k: v for k, v in wanted.items() if k in fields}
    return wandb.Settings(**wanted)


def setup_wandb(project='offline-empowerment', group=None, name=None, entity=None):
    """Set up wandb. Respects WANDB_MODE (e.g. WANDB_MODE=disabled to turn it off)."""
    return wandb.init(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=[group] if group is not None else None,
        group=group,
        dir=tempfile.mkdtemp(),
        name=name,
        settings=_wandb_settings(),
        mode=os.environ.get('WANDB_MODE', 'online'),
        save_code=True,
    )
