"""Online goal-reaching tasks from the paper (fixed start, a fixed set of goals, no reset noise).

Use them through the usual ogbench naming with `online` in place of the dataset type, e.g.
`--env_name=antmaze-medium-corner-all-squares-online-v0`. They share the observation/action spaces of the regular
ogbench envs, so the regular datasets (antmaze-medium-navigate-v0, ...) can be used for RLPD.
"""

import functools

import ogbench  # noqa: F401  (registers the regular ogbench envs)
from gymnasium.envs.registration import register

from envs.maze import center_to_free_cells_tasks

online_dict = dict(
    add_noise_to_init=False,
    add_noise_to_goal=False,
)

# antmaze-medium: ant at the bottom-left corner (1, 1), goal = center of any other free cell (25 goals).
register(
    id='antmaze-medium-corner-all-squares-v0',
    entry_point='envs.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='maze',
        maze_type='medium',
        tasks=functools.partial(center_to_free_cells_tasks, start_ij=(1, 1)),
        **online_dict,
    ),
)

# pointmaze-teleport: point at the bottom-left corner (1, 1), goal = center of any other free cell except the two
# teleporter entrances (4, 6), (5, 1) and the walled-off (1, 7) pocket (41 goals).
register(
    id='pointmaze-teleport-corner-all-squares-v0',
    entry_point='envs.maze:make_maze_env',
    max_episode_steps=1000,
    kwargs=dict(
        loco_env_type='point',
        maze_env_type='maze',
        maze_type='teleport',
        tasks=functools.partial(center_to_free_cells_tasks, start_ij=(1, 1), exclude_ijs=((4, 6), (5, 1), (1, 7))),
        **online_dict,
    ),
)

# antsoccer-arena: ant at (10, 10), ball at (11, 11), eight ball goals up and to the right of the ball.
antsoccer_arena_easy_goal_xys = [
    (11.0, 12.0),
    (12.0, 11.0),
    (12.0, 12.0),
    (13.0, 12.0),
    (12.0, 13.0),
    (13.0, 13.0),
    (11.0, 13.0),
    (13.0, 11.0),
]
register(
    id='antsoccer-arena-easy-v0',
    entry_point='envs.maze:make_maze_env',
    max_episode_steps=500,
    kwargs=dict(
        loco_env_type='ant',
        maze_env_type='ball',
        maze_type='arena',
        tasks=[
            dict(agent_init_xy=(10.0, 10.0), ball_init_xy=(11.0, 11.0), goal_xy=goal_xy)
            for goal_xy in antsoccer_arena_easy_goal_xys
        ],
        **online_dict,
    ),
)

# cube-single: cube starts at (0.425, 0.1), goal = one of 30 positions on a 5 x 6 grid over the table.
register(
    id='cube-single-multigoal-v0',
    entry_point='envs.cube:OnlineCubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
        permute_blocks=False,
        add_noise_to_init=False,
        tasks=[
            dict(
                task_name=f'goal_x{gx}_y{gy}',
                init_xyzs=[[0.425, 0.1, 0.02]],
                goal_xyzs=[[gx, gy, 0.02]],
            )
            for gx in (0.325, 0.375, 0.425, 0.475, 0.525)
            for gy in (-0.25, -0.15, -0.05, 0.05, 0.15, 0.25)
        ],
    ),
)
