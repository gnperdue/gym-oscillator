import os
import unittest
import tempfile

import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import envs
import gym_oscillator


# PLOT_NAME = '/tmp/tmp.pdf'
# Handle Windows (Win) or Mac path for temp log file
tempdir = tempfile.gettempdir();
slash2 = '\\';  # specific to Windows
bstr = str.find(tempdir, slash2); # check if double slash (Win) or not (Mac)
log_str ='\\tmp.pdf' if bstr > 0 else '/tmp.pdf'; # create log-file string

PLOT_NAME = tempdir + log_str;


class TestTermination(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.env = envs.make('oscillator-v0')

    def tearDown(self):
        pass

    def test_termination(self):
        if os.path.isfile(PLOT_NAME):
            os.remove(PLOT_NAME)
        # the first observation should not set the `done` flag
        observation, reward, done, d = self.env.step(4)
        self.assertFalse(done)
        # just run until we hit the accumulated heat threshold. with no
        # settings adjustments, should have more than 1,500 steps.
        while not done:
            observation, reward, done, d = self.env.step(4)
        # matplotlib is sort of magical - we get the Figure as a return object
        # from the environment, but then we don't call methods on it; instead,
        # `plt` knows how to work with the current figure in memory somehow.
        f = self.env.render(mode='return_figure')
        plt.savefig(PLOT_NAME, bbox_inches='tight')
        plot_file_size = os.stat(PLOT_NAME).st_size
        # plot should really be ~20+kB, but no need to be too picky. an empty
        # file would be around 1.2kB, so 10kB means _something_ plotted.
        self.assertGreater(plot_file_size, 10000)
