import unittest

# import gym
from gym import envs
import gym_oscillator
# import gym_oscillator.envs.oscillator as oscillator


class TestRegistration(unittest.TestCase):

    def test_make(self):
        env = envs.make('oscillator-v0')
        self.assertEqual(env.spec.id, 'oscillator-v0')
