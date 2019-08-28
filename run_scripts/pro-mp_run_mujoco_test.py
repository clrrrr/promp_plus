import os
from sys import platform

# escaping check_mujoco_version() in config.py (mjpro have to == mjpro131)
if platform == 'linux':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = "/home/zhjl/.mujoco/mjpro131"
if platform == 'darwin':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = "/Users/clrrrr/.mujoco/mjpro131"

from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
# from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv

from meta_policy_search.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_policy_search.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_policy_search.envs.mujoco_envs.walker2d_rand_params import WalkerRandParamsWrappedEnv

'''
Usage:
python run_scripts/pro-mp_run_mujoco_test.py --dir xxx --eff(default=20) 10 
e.g. "AntRandGoal/run_1566926648"

'''

from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_tester import Tester
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import numpy as np
import tensorflow as tf
import json
import argparse
import time
import joblib

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    set_seed(config['seed'])
    sess = tf.Session()

    with sess.as_default() as sess:

        data = joblib.load(load_path + "/params.pkl")
        policy = data['policy']
        env = data['env']
        baseline = data['baseline']

        sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=config['rollouts_per_meta_task'],  # Will be modified later
            meta_batch_size=config['meta_batch_size'],
            max_path_length=config['max_path_length'],
            parallel=config['parallel'],
        )

        sample_processor = MetaSampleProcessor(
            baseline=baseline,
            discount=config['discount'],
            gae_lambda=config['gae_lambda'],
            normalize_adv=config['normalize_adv'],
        )

        algo = ProMP(
            policy=policy,
            inner_lr=config['inner_lr'],
            meta_batch_size=config['meta_batch_size'],
            num_inner_grad_steps=config['num_inner_grad_steps'],
            learning_rate=config['learning_rate'],
            num_ppo_steps=config['num_promp_steps'],
            clip_eps=config['clip_eps'],
            target_inner_step=config['target_inner_step'],
            init_inner_kl_penalty=config['init_inner_kl_penalty'],
            adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
        )

        tester = Tester(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            #n_itr=config['n_itr'],
            eff=config['eff'],
            num_inner_grad_steps=config['num_inner_grad_steps'],
        )

        tester.train()
        sess.close()

if __name__=="__main__":
    # idx = int(time.time())
    parser = argparse.ArgumentParser(description='ProMP: Proximal Meta-Policy Search')
    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    # parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)

    parser.add_argument('--eff', type=int, default=20)
    parser.add_argument('--dir', type=str)

    args = parser.parse_args()

    load_path = meta_policy_search_path + "/data/pro-mp/" + args.dir
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