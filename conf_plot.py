import atcenv.TempConfig as tc
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

num_episodes = 400
conf = np.array([])
for i in range(num_episodes):
    conf = np.append(conf,tc.load_pickle(f'results/save/numberconflicts_EPISODE_{i}'))

plt.plot(moving_average(conf,100))
plt.show()