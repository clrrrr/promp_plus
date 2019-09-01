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
from meta_policy_search.tester import Tester
from meta_policy_search.samplers.rl2.maml_sampler import MAMLSampler
from meta_policy_search.samplers.rl2.rl2_sample_processor import RL2SampleProcessor
from meta_policy_search.policies.rl2.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.policies.rl2.gaussian_rnn_policy import GaussianRNNPolicy
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
import joblib
import time

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):

    sess = tf.Session()

    with sess.as_default() as sess:

        data = joblib.load(load_path + "/params.pkl")
        policy = data['policy']
        env = data['env']
        baseline = data['baseline']

        config['meta_batch_size'] = env.NUM_EVAL

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

        tester = Tester(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            eff=config['eff'],
        )

        tester.train()
        sess.close()

if __name__=="__main__":
    # idx = int(time.time())
    parser = argparse.ArgumentParser(description='RL2')
    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    # parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--eff', type=int, default=20)
    parser.add_argument('--dir', type=str)

    args = parser.parse_args()

    load_path = meta_policy_search_path + "/data/rl2/" + args.exp + "/" + args.dir
    dump_path = load_path.replace("run", "test")

    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else: # use default config
        if args.dir.startswith("W"): # args.exp
            if platform == 'linux':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mjpro131"
            if platform == 'darwin':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mjpro131"
            # env_name = args.exp + "WrappedEnv"
        else:
            if platform == 'linux':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mujoco200"
            if platform == 'darwin':
                os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mujoco200"
                # env_name = args.exp + "Env"

        config = json.load(open(load_path + "/params.json", 'r'))
        config['eff'] = args.eff


    # configure logger
    logger.configure(dir=dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')  # args.dump_path

    # dump run configuration before starting training
    # json.dump(config, open(dump_path + '/params.json', 'w'), cls=ClassEncoder) # args.dump_path

    # start the actual algorithm
    main(config)