from meta_policy_search.samplers.base import Sampler
from meta_policy_search.samplers.vectorized_env_executor import MetaParallelEnvExecutor, MetaIterativeEnvExecutor
from meta_policy_search.utils import utils, logger
from collections import OrderedDict

from pyprind import ProgBar
import numpy as np
import time
import itertools


class MetaSampler(Sampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_policy_search.envs.base.MetaEnv) : environment object
        policy (meta_policy_search.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            rollouts_per_meta_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False
            ):
        super(MetaSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task  #########
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        self.env = env ########### add this for update_batch_size()
        self.vec_envs = {}
        # setup vectorized environment

        if self.parallel:
            self.vec_env = MetaParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MetaIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
    '''
    def update_meta_batch_size(self, meta_batch_size): # num of tasks
        self.meta_batch_size = meta_batch_size
        self.total_samples = self.meta_batch_size * self.batch_size * self.max_path_length
    '''
    def update_batch_size(self, batch_size): # num of rollouts per task
        self.batch_size = batch_size
        self.envs_per_task = batch_size
        self.total_samples = self.meta_batch_size * batch_size * self.max_path_length

        #Parallel = True
        if str(self.envs_per_task) in self.vec_envs:
            self.vec_env = self.vec_envs[str(self.envs_per_task)]
            self.vec_env.reset()
        else:
            self.vec_envs[str(self.envs_per_task)] = MetaParallelEnvExecutor(self.env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
            self.vec_env = self.vec_envs[str(self.envs_per_task)]
            self.vec_env.reset()

        # if self.parallel:
        #     self.vec_env = MetaParallelEnvExecutor(self.env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        # else:
        #     self.vec_env = MetaIterativeEnvExecutor(self.env, self.meta_batch_size, self.envs_per_task, self.max_path_length)



    def update_tasks(self, test=False, start_from=0):
        """
        Samples a new goal for each meta task
        """
        if not test:
            tasks = self.env.sample_tasks(self.meta_batch_size)
            assert len(tasks) == self.meta_batch_size
        else:
            tasks = self.env.sample_tasks(self.meta_batch_size, is_eval=True, start_from=start_from)
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, log=False, log_prefix='', test=False):

        print("--------------obtaining", self.total_samples//self.meta_batch_size//self.max_path_length,
              "rollouts_per_task, for", self.meta_batch_size, "tasks..--------------")

        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0

        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]
        print("=========runnng_paths length:", len(running_paths), "=========")

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy

        # initial reset of envs
        obses = self.vec_env.reset()
        
        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            actions = np.concatenate(actions) # stack meta batch
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses

        pbar.stop()

        if not test:
            self.total_timesteps_sampled += self.total_samples
            print("------------self.total_timesteps_sampled:", self.total_timesteps_sampled, "-----------------")


        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(agent_infos) == self.meta_batch_size
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = sum(agent_infos, [])  # stack agent_infos

        assert len(agent_infos) == self.meta_batch_size * self.envs_per_task == len(env_infos)
        return agent_infos, env_infos


def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], env_infos=[], agent_infos=[])
