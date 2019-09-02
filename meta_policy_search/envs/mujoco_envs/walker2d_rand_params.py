import numpy as np
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from meta_policy_search.envs.base import MetaEnv
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv


class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv): ##### MetaEnv, gym.utils.EzPickle, MujocoEnv, 去掉试试
    def __init__(self, randomize_tasks=True):

        '''
        import os
        from sys import platform

        if platform == 'linux':
            line = os.environ['MUJOCO_PY_MJPRO_PATH']
            os.environ['MUJOCO_PY_MJPRO_PATH'] = line.replace('mujoco200', 'mjpro131')
        elif platform == 'darwin':
            os.environ['MUJOCO_PY_MJPRO_PATH'] = '~/.mujoco/mjpro131'
        else: # win32 can only use up to 150
            line = os.environ['MUJOCO_PY_MJPRO_PATH']
            os.environ['MUJOCO_PY_MJPRO_PATH'] = line.replace('mjpro150', 'mjpro131')
        '''

        super(WalkerRandParamsWrappedEnv, self).__init__()
        np.random.seed(1337)

        self.NUM_TRAIN = 80
        self.NUM_EVAL = 20
        self.NUM_TASKS = self.NUM_TRAIN + self.NUM_EVAL

        self._tasks = super(WalkerRandParamsWrappedEnv, self).sample_tasks(self.NUM_TASKS) # self.sample_tasks(self.NUM_TASKS)
        self.set_task(self._tasks[0])

    def sample_tasks(self, n_tasks, is_eval=False, start_from=0):
        if is_eval:
            # return [self._tasks[-idx] for idx in np.random.choice(range(1, self.NUM_EVAL+1), size=n_tasks)]
            ret = [self._tasks[idx + self.NUM_TRAIN + start_from] for idx in range(min(n_tasks, self.NUM_EVAL))]
            ret.extend([self._tasks[-idx] for idx in np.random.choice(range(1, self.NUM_EVAL+1), size=n_tasks-len(ret))])
            return ret
        else:
            return [self._tasks[idx] for idx in np.random.choice(range(self.NUM_TRAIN), size=n_tasks)]

    '''
    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 15.0
        forward_vel = (posafter - posbefore) / self.dt
        reward = - np.abs(forward_vel - self.goal_velocity)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
    '''
