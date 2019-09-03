import os
from sys import platform

# escaping check_mujoco_version() in config.py (mjpro have to == mjpro131)
if platform == 'linux':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mjpro131"
if platform == 'darwin':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mjpro131"

from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.rl2_env import rl2env
from meta_policy_search.algos.vpg import VPG
from meta_policy_search.algos.ppo import PPO
from meta_policy_search.trainer import Trainer
from meta_policy_search.samplers.rl2.maml_sampler import MAMLSampler
from meta_policy_search.samplers.rl2.rl2_sample_processor import RL2SampleProcessor
from meta_policy_search.policies.rl2.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.policies.gaussian_rnn_policy import GaussianRNNPolicy
import os
from meta_policy_search.utils import logger
from meta_policy_search.utils.rl2.utils import set_seed, ClassEncoder
import json
import numpy as np


#from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_params import WalkerRandParamsWrappedEnv


import numpy as np
import tensorflow as tf
import json
import argparse
import time

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):

    baseline = LinearFeatureBaseline()
    #env = rl2env(HalfCheetahRandDirecEnv())
    env = rl2env(globals()[config['env']]())  # instantiate env
    obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1

    policy = GaussianRNNPolicy(
            name="meta-policy",
            obs_dim=obs_dim,
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
            cell_type=config['cell_type']
    )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=1,
    )

    sample_processor = RL2SampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
    )
    trainer.train()


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='RL2')
    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    parser.add_argument('--exp', type=str)
    parser.add_argument('--rollouts_per_meta_task', type=int, default=1)

    args = parser.parse_args()

    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else: # use default config

        meta_batch_size = 50
        if args.exp.startswith("W"):
            if platform == 'linux':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mjpro131"
            if platform == 'darwin':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mjpro131"
                meta_batch_size = 5
            env_name = args.exp + "WrappedEnv"
        else:
            if platform == 'linux':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mujoco200"
            if platform == 'darwin':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mujoco200"
                meta_batch_size = 5
            env_name = args.exp + "Env"

        #config = json.load(open(maml_zoo_path + "/configs/rl2_config.json", 'r'))
        config = {
            "algo": "RL2",
            'env': env_name,  # 'AntRandGoalEnv',

            "meta_batch_size": meta_batch_size, #200！
            "hidden_sizes": [64],

            "rollouts_per_meta_task": 3,
            "parallel": True,
            "max_path_length": 200, #100
            "n_itr": 10000, #1000

            "discount": 0.99,
            "gae_lambda": 1.0,
            "normalize_adv": True,
            "positive_adv": False,
            "learning_rate": 1e-3,
            "max_epochs": 5,
            "cell_type": "lstm",
            "num_minibatches": 1,
        }

    dump_path = meta_policy_search_path + "/data/rl2/" + args.exp + "/run_" + time.strftime("%y%m%d_%H%M%S", time.localtime())

    # configure logger
    logger.configure(dir=dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')  # args.dump_path

    # dump run configuration before starting training
    json.dump(config, open(dump_path + '/params.json', 'w'), cls=ClassEncoder) # args.dump_path

    # start the actual algorithm
    main(config)


