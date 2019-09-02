import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger


class Tester(object):
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            start_itr=0,
            num_inner_grad_steps=1,
            eff=20, ########## maximum of test rollouts
            sess=None,
    ):
        self.eff = eff
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.num_inner_grad_steps = num_inner_grad_steps
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    #############################
    #  Tasks: whole test-split  #
    #  Rollout_per_task:        #
    #   adaptation: 1~20        #
    #    logging: 2             #
    #############################

    def train(self):

        for i in range(1, self.eff+1):

            with self.sess.as_default() as sess:

                logger.log("----------- Adaptation rollouts per meta-task = ", i, " -----------")

                undiscounted_returns = []
                for i in range(0, self.env.NUM_EVAL, self.sampler.meta_batch_size):

                    # initialize uninitialized vars  (only initialize vars that were not loaded)
                    uninit_vars = [var for var in tf.global_variables() if
                                   not sess.run(tf.is_variable_initialized(var))]
                    sess.run(tf.variables_initializer(uninit_vars))

                    logger.log("Sampling set of tasks/goals for this meta-batch...")
                    self.sampler.update_tasks(test=True, start_from=i)  # sample from test split!
                    self.policy.switch_to_pre_update()  # Switch to pre-update policy

                    for step in range(self.num_inner_grad_steps + 1):

                        if step < self.num_inner_grad_steps:
                            self.sampler.update_batch_size(i)
                            logger.log("On step-0: Obtaining samples...")
                        else:
                            self.sampler.update_batch_size(2)
                            logger.log("On step-1: Obtaining samples...")

                        paths = self.sampler.obtain_samples(log=False, test=True) # log_prefix='test-Step_%d-' % step

                        logger.log("On Test: Processing Samples...")
                        samples_data = self.sample_processor.process_samples(paths, log=False) # log='all', log_prefix='test-Step_%d-' % step
                        self.log_diagnostics(sum(list(paths.values()), []), prefix='test-Step_%d-' % step)

                        """ ------------------- Inner Policy Update / logging returns --------------------"""
                        if step < self.num_inner_grad_steps:
                            logger.log("On Test: Computing inner policy updates...")
                            self.algo._adapt(samples_data)
                        else:
                            paths = self.sample_processor.gao_paths(paths)
                            undiscounted_returns.extend([sum(path["rewards"]) for path in paths])

                test_average_return = np.mean(undiscounted_returns)





                for step in range(self.num_inner_grad_steps + 1):

                    logger.log('** Step ' + str(step) + ' **')
                    """ -------------------- Sampling --------------------------"""
                    logger.log("Obtaining samples...")

                    if step < self.num_inner_grad_steps:
                        paths = self.sampler.obtain_samples(log=True, log_prefix='Step_%d-' % step)
                        print("step0-rollouts:", len(paths[0]))
                    else:
                        # sample 2 trajectories for eval
                        self.sampler.update_batch_size(2)
                        paths = self.sampler.obtain_samples(log=True, log_prefix='Step_%d-' % step, test=True)
                        print("step1-rollouts:", len(paths[0]))

                    """ ----------------- Processing Samples ---------------------"""

                    logger.log("Processing samples...")
                    # time_proc_samples_start = time.time()
                    samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % step)
                    #all_samples_data.append(samples_data)
                    # list_proc_samples_time.append(time.time() - time_proc_samples_start)

                    self.log_diagnostics(sum(list(paths.values()), []), prefix='Step_%d-' % step)

                    """ ------------------- Inner Policy Update --------------------"""

                    # time_inner_step_start = time.time()
                    if step < self.num_inner_grad_steps:
                        logger.log("Computing inner policy updates...")
                        self.algo._adapt(samples_data)


                # """ ------------------ Outer Policy Update ---------------------"""

                # logger.log("Optimizing policy...")
                # # This needs to take all samples_data so that it can construct graph for meta-optimization.
                # time_outer_step_start = time.time()
                # self.algo.optimize_policy(all_samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                # logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                # logger.log("Saving snapshot...")
                # params = self.get_itr_snapshot(itr)
                # logger.save_itr_params(itr, params)
                # logger.log("Saved")

                logger.logkv('rollouts_per_meta_task', i)
                logger.dumpkvs()

            logger.log("Testing finished")


    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
