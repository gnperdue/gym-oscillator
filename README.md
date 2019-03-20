# Gym Oscillator

This environment reports the values of four independent oscillators. They
represent noisy sensors on a device. The device has one calibration setting.
The device emits "heat" based on the difference between the sum of the true
values of the sensors and the calibration setting. Rewards are accrued by
minimizing the heat (the reward is the negative of the heat).

Note, the "heat" is capped at 100, so the minimum instantaneous reward possible
is -100 and the maximum instantaneous reward is 0.
