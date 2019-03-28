import unittest
import random

import numpy as np

import gym
from gym import envs
import gym_oscillator


class RandomActionWrapper(gym.ActionWrapper):

    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return action


class TestActionWrappers(unittest.TestCase):

    def setUp(self):
        env = envs.make('oscillator-v0')
        self.rando = RandomActionWrapper(env)

    def tearDown(self):
        pass

    def test_actions(self):
        a = [self.rando.action(5) for _ in range(1000)]
        self.assertFalse(np.all(a == 5))
