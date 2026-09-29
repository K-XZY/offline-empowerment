# ogbench's CubeEnv with a custom task set and deterministic (noise-free) resets, for the online cube task.
import mujoco
import numpy as np
from ogbench.manipspace import lie
from ogbench.manipspace.envs.cube_env import CubeEnv


class OnlineCubeEnv(CubeEnv):
    def __init__(self, *args, tasks, add_noise_to_init=False, **kwargs):
        # set before super().__init__, which calls set_tasks()
        self._tasks = tasks
        self._add_noise_to_init = add_noise_to_init
        super().__init__(*args, **kwargs)

    def set_tasks(self):
        self.task_infos = []
        for task in self._tasks:
            task = dict(task)
            for key in ('init_xyzs', 'goal_xyzs'):
                task[key] = np.asarray(task[key], dtype=np.float64)
            self.task_infos.append(task)

    def initialize_arm(self):
        # without init noise the arm starts at the center of its sampling box with zero yaw
        if self._add_noise_to_init:
            eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
            yaw = self.np_random.uniform(-np.pi, np.pi)
        else:
            eff_pos = np.asarray(self._arm_sampling_bounds).mean(axis=0)
            yaw = 0.0
        cur_ori = self._effector_down_rotation
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        assert self._mode == 'task'
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._permute_blocks:
            permutation = self.np_random.permutation(self._num_cubes)
        else:
            permutation = np.arange(self._num_cubes)
        init_xyzs = self.cur_task_info['init_xyzs'].copy()[permutation]
        goal_xyzs = self.cur_task_info['goal_xyzs'].copy()[permutation]

        # First set the scene to the goal state to get the goal observation.
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        self.initialize_arm()
        for i in range(self._num_cubes):
            self._data.joint(f'object_joint_{i}').qpos[:3] = goal_xyzs[i]
            self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
            self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
        mujoco.mj_forward(self._model, self._data)

        # a couple of steps to settle the scene (zero action when resets are deterministic)
        for _ in range(2):
            self.step(self.action_space.sample() if self._add_noise_to_init else np.zeros(self.action_space.shape))

        self._cur_goal_ob = self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
        self._cur_goal_rendered = self.render() if self._render_goal else None

        # Now the actual reset.
        self._data.qpos[:] = saved_qpos
        self._data.qvel[:] = saved_qvel
        self.initialize_arm()
        for i in range(self._num_cubes):
            obj_pos = init_xyzs[i].copy()
            if self._add_noise_to_init:
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
            self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
            yaw = self.np_random.uniform(0, 2 * np.pi) if self._add_noise_to_init else 0.0
            obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
            self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_xyzs[i]
            self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False
