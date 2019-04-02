import unittest
import random

import numpy as np

import gym
from gym import envs
import gym_oscillator


class TestTermination(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.env = envs.make('oscillator-v0')

    def tearDown(self):
        pass

    def test_termination(self):
        observation, reward, done, d = self.env.step(4)
        self.assertFalse(done)
        while not done:
            observation, reward, done, d = self.env.step(4)
