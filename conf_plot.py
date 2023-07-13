import atcenv.TempConfig as tc
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def stepwise_average(x,w):
    temp = np.array([])
    for i in range(0,int(np.ceil(len(x)/w))):
        temp = np.append(temp,np.mean(x[int(i*w):int((i+1)*w)]))
    return temp

average = moving_average
average = stepwise_average

dt = 5.0
num_ac = 10 
density = 1 #factor higher than default

num_episodes =  70000
rolling_window =  100

test = ''
test = 'test_'

mvp = ''
# mvp= '_mvp'



conf = np.genfromtxt(f'{test}conflict{mvp}.csv')
conf = conf.flatten()
print(len(conf))
plt.figure()
plt.plot(average((10/num_ac)*conf[conf<20000]*(dt/5)/density,rolling_window))


drift = np.genfromtxt(f'{test}drift{mvp}.csv')
drift = drift.flatten()
plt.figure()
plt.plot(average(drift[0:num_episodes],rolling_window))


reward = np.genfromtxt(f'{test}reward{mvp}.csv')
reward = reward.flatten()
plt.figure()
plt.plot(average(reward[:num_episodes],rolling_window))
plt.show()

reward = average(reward[:num_episodes],rolling_window)
conf = average((10/num_ac)*conf[conf<200000]*(dt/5)/density,rolling_window)
drift = average(drift[0:num_episodes],rolling_window)

print(np.mean(conf))

print('episode:',np.argmax(reward)+rolling_window)
print(conf[np.argmax(reward)])
print(np.arccos(1+drift[np.argmax(reward)])*(180/3.1415))

print('episode:',np.argmin(conf)+rolling_window)
print(conf[np.argmin(conf)])
print(np.arccos(1+drift[np.argmin(conf)])*(180/3.1415))

print('episode:',reward==np.max(reward[conf==0])+rolling_window)
print(conf[reward==np.max(reward[conf==0])])
print(np.arccos(1+drift[reward==np.max(reward[conf==0])])*(180/3.1415))
