from collections import deque
import gzip
import shutil
import os
import string  # temp log
import random  # temp log

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


# default data amplitudes
D1AMP, D2AMP, D3AMP, D4AMP = 10.0, 1.0, 0.5, 0.1
# default data update frequencies / pi
D1FRQ, D2FRQ, D3FRQ, D4FRQ = 1.0, 0.1, 3.0, 10.0
DTYPE = np.float32
# default noise amplitudes
N1AMP, N2AMP, N3AMP, N4AMP = 0.05, 0.04, 0.02, 0.01

# machine update "dial settings"
DEFAULT_COMMANDS = np.array([-0.5, -0.375, -0.25, -0.125, 0.0,
                             0.125, 0.25, 0.375, 0.5], dtype=DTYPE)

MAX_HEAT = 100.0
MAX_SETTING = 15.0
MIN_SETTING = -15.0


class DataGenerator(object):
    '''
    generate a set of oscillation patterns of different frequencies as function
    of a discrete time-step - user must manually `step` the system with time.
    '''
    # amp = np.asarray([D1AMP, D2AMP, D3AMP, D4AMP], dtype=DTYPE)
    # frq = np.pi * np.asarray([D1FRQ, D2FRQ, D3FRQ, D4FRQ], dtype=DTYPE)

    def __init__(self, amplitudes, frequencies, time_step=0.01):
        # -21 because 20 steps compose a full observation. we want to init to
        # -1 so `env.reset()` ticks forward to t==0
        self.t = -21.0 * time_step
        self.time_step = time_step
        self.amp = amplitudes
        self.frq = frequencies
        assert len(self.amp) == len(self.frq)

    def _gen_point(self):
        raw_vs = []
        for i in range(len(self.amp)):
            raw_vs.append(
                self.amp[i] * np.cos(self.frq[i] * self.t)
            )
        return np.asarray(raw_vs, dtype=DTYPE)

    def step(self):
        self.t = self.t + self.time_step
        data = self._gen_point()
        return data


class NoiseModel(object):
    '''
    generate a set of noise complimentary to the 'true' values from a
    `DataGenerator` object. noise is a function of the true data values and
    not a function of time.
    '''
    default_noise_scale = [N1AMP, N2AMP, N3AMP, N4AMP]

    def __init__(self, drop_probability=0.0, noise_array=None):
        self.noise_scale = noise_array or np.asarray(
            NoiseModel.default_noise_scale, dtype=DTYPE
        )
        assert len(self.noise_scale) == 4

    def gen_noise(self, data):
        assert len(data) == 4
        noise_values = []
        for i, d in enumerate(data):
            noise_values.append(
                self.noise_scale[i] * data[i] * np.random.randn()
            )
        return np.asarray(noise_values, dtype=DTYPE)


class MachineStateTextRecorder(object):
    '''
    record observation values to a .csv file and gzip after calling `close()`.
    '''

    def __init__(self, log_name):
        self.log_name = log_name
        self.gzfile = self.log_name + '.gz'
        self._cleanup_files()

    def _cleanup_files(self):
        for f in [self.log_name, self.gzfile]:
            if os.path.isfile(f):
                os.remove(f)

    def write_data(self, data):
        with open(self.log_name, 'ab+') as f:
            msg = ','.join([str(i) for i in data]) + '\n'
            f.write(bytes(msg, 'utf8'))
        return True

    def read_data(self):
        '''
        do not call this on large files (we just read it all).
        NOTE: we are assuming gzip compression has occurred!
        '''
        with gzip.open(self.log_name + '.gz', 'rb') as f:
            content = f.readlines()
            content = [x.decode('utf8').strip() for x in content]
        return content

    def close(self):
        '''zip the log file'''
        if not os.path.isfile(self.log_name):
            return
        with open(self.log_name, 'rb') as f_in:
            with gzip.open(self.gzfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if os.path.isfile(self.gzfile) and (os.stat(self.gzfile).st_size > 0):
            os.remove(self.log_name)
        else:
            raise IOError('Compressed file not produced!')


class SimulationMachine(object):
    '''
    intended operation...

    when finished, call `close_logger()` to zip the log file.
    '''

    def __init__(
        self, setting, data_generator, noise_model, logger=None, commands=None
    ):
        self._data_generator = data_generator
        self._noise_model = noise_model
        self._setting = setting
        self._commands = commands or DEFAULT_COMMANDS
        self._logger = logger
        self._true_instantaneous_sensor_vals = None
        self._observation = deque([], maxlen=80)
        self._initialize()
        self._enforce_settings_bounds()

    def _initialize(self):
        for i in range(20):
            data = self._data_generator.step()
            noise = self._noise_model.gen_noise(data)
            measured = data + noise
            for m in measured:
                self._observation.append(m)
        self._true_instantaneous_sensor_vals = list(data)

    def _enforce_settings_bounds(self):
        self._setting = min(self._setting, MAX_SETTING)
        self._setting = max(self._setting, MIN_SETTING)

    def _true_state(self):
        return sum(self._true_instantaneous_sensor_vals)

    def update_machine(self, command):
        '''command is the index of the step change'''
        self._setting = self._setting + self._commands[command]
        self._enforce_settings_bounds()

    def step(self):
        data = self._data_generator.step()
        self._true_instantaneous_sensor_vals = list(data)
        noise = self._noise_model.gen_noise(data)
        measured = data + noise
        for m in measured:
            self._observation.append(m)
        heat = self.get_heat()

        return_value = list(self._observation) + \
            [self._setting, self._data_generator.t, heat]

        if self._logger is not None:
            self._logger.write_data(return_value)

        return return_value

    def get_heat(self):
        return min((self._true_state() - self._setting) ** 2, MAX_HEAT)

    def get_time(self):
        return self._data_generator.t

    def get_setting(self):
        return self._setting

    def get_commands(self):
        return list(self._commands)

    def get_sensor_values(self):
        return list(self._observation)[-4:]

    def close_logger(self):
        self._logger.close()


def log_namer():
    opts = list(map(str, range(1, 11))) + list(string.ascii_lowercase)
    return '/tmp/log' + \
        ''.join([random.choice(opts) for _ in range(30)]) + '.txt'


class OscillatorEnv(gym.Env):
    '''
    Note: observations always include the previous 19 steps. (Observations are
    80 numbers - 20 for each sensor.) To start at "t=0" means 19 previous steps
    are included. Heat is instantaneous though, so there is no penalty for a
    "bad" setting before step=0.

    Steps adjust the device setting prior to sensor readout.
    '''
    metadata = {'render.modes': ['human']}

    # TODO - we can pass params, see Atari example for usage pattern
    def __init__(self):
        self.amp = np.asarray([D1AMP, D2AMP, D3AMP, D4AMP], dtype=DTYPE)
        self.frq = np.pi * np.asarray([D1FRQ, D2FRQ, D3FRQ, D4FRQ],
                                      dtype=DTYPE)
        self.setting = 10.0
        self.machine = None
        self.reset()

        self.action_space = spaces.Discrete(len(DEFAULT_COMMANDS))
        # observation is last 20 values for 4 sensors, setting, time ->
        # 82 floating point elements (don't explicitly include time values
        # for previous steps included in the observation)
        high = np.array([np.finfo(np.float32).max] * 82)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action):
        self.machine.update_machine(action)
        data = self.machine.step()
        observation = data[:-1]
        reward = -1 * data[-1]  # heat
        return observation, reward, False, {}

    def reset(self):
        if self.machine is not None:
            self.machine.close_logger()
        data_generator = DataGenerator(self.amp, self.frq)
        noise_generator = NoiseModel()
        recorder_log = log_namer()
        recorder = MachineStateTextRecorder(recorder_log)
        self.machine = SimulationMachine(
            setting=self.setting, data_generator=data_generator,
            noise_model=noise_generator, logger=recorder
        )
        data = self.machine.step()
        observation = data[:-1]
        return observation

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.machine.close_logger()
