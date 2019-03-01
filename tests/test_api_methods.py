import unittest

import numpy as np

# import gym
from gym import envs
from gym import spaces
import gym_oscillator
# import gym_oscillator.envs.oscillator as oscillator


class TestAPIMethods(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.env = envs.make('oscillator-v0')
        pass

    def tearDown(self):
        pass

    def test_spaces(self):
        self.assertEqual(self.env.action_space.n, 9)
        self.assertIsInstance(self.env.action_space, spaces.discrete.Discrete)
        # print(self.env.observation_space)
        self.assertIsInstance(self.env.observation_space, spaces.box.Box)

    def test_api_methods(self):
        observation, reward, done, d = self.env.step(4)
        self.assertFalse(done)
        self.assertEqual(len(observation), 82)
        self.assertNotEqual(reward, 0.0)
        self.env.reset()
        self.env.render()
        self.env.close()

    def test_multi_step(self):
        # observation is 20 x 4 sensor readings, setting, time
        observation, reward, done, d = self.env.step(4)
        self.assertAlmostEqual(observation[-2], 10.0)
        self.assertAlmostEqual(observation[-1], 0.01)
        observation, reward, done, d = self.env.step(4)
        self.assertAlmostEqual(observation[-2], 10.0)
        self.assertAlmostEqual(observation[-1], 0.02)
        observation, reward, done, d = self.env.step(4)
        self.assertAlmostEqual(observation[-2], 10.0)
        self.assertAlmostEqual(observation[-1], 0.03)
        self.env.reset()
        observation, reward, done, d = self.env.step(4)
        self.assertAlmostEqual(observation[-2], 10.0)
        self.assertAlmostEqual(observation[-1], 0.01)

    def test_actions(self):
        observation, reward, done, d = self.env.step(4)
        self.assertAlmostEqual(observation[-2], 10.0)
        self.assertAlmostEqual(observation[-1], 0.01)

        self.env.reset()
        observation, reward, done, d = self.env.step(0)
        self.assertAlmostEqual(observation[-2], 9.5)
        self.assertAlmostEqual(observation[-1], 0.01)

        self.env.reset()
        observation, reward, done, d = self.env.step(8)
        self.assertAlmostEqual(observation[-2], 10.5)
        self.assertAlmostEqual(observation[-1], 0.01)

        # print(observation)
