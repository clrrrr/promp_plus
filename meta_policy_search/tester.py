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
            eff,
            sess=None,
            ):
        self.eff=eff
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):

        for i in range(1, self.eff+1):

            with self.sess.as_default() as sess:

                logger.log("----------- Adaptation rollouts per meta-task = ", i, " -----------")
                # self.sampler.rollouts_per_meta_task = 10000
                self.sampler.update_batch_size(i)

                # initialize uninitialized vars  (only initialize vars that were not loaded)
                uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
                sess.run(tf.variables_initializer(uninit_vars))

                self.task = self.env.sample_tasks(self.sampler.meta_batch_size, is_eval=True)
                self.sampler.set_tasks(self.task)

                #logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-')

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                #""" ------------------ Policy Update ---------------------"""

                #logger.log("Optimizing policy...")
                ## This needs to take all samples_data so that it can construct graph for meta-optimization.
                #time_optimization_step_start = time.time()
                #self.algo.optimize_policy(samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                #logger.log("Saving snapshot...")
                #params = self.get_itr_snapshot(itr)
                #logger.save_itr_params(itr, params)
                #logger.log("Saved")

                logger.dumpkvs()
                # if itr == 0:
                    # sess.graph.finalize()

            logger.log("Training finished")
        self.sess.close()

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
