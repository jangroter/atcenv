import numpy as np
import matplotlib.pyplot as plt
from atcenv.src.functions import moving_average

# folder = "FeedForward_local_benchmark"
# folder = "Transformer_relative_distance_benchmark_2"
# folder = "Transformer_relative_benchmark_2"
folder = "FeedForward_ownref_straight_Actlrx10"
# folder = "Transformer_ownref_straight"
# folder = "Transformer_ownref_straight_Actlrx10"
# folder = "Transformer5Heads_ownref_straight_Actlrx10"
# folder = "TransformerTest"
folder = "FeedForward_absolute_20AC_Actlrx10"
# folder = "Transformer_ownref_20AC_Actlrx10"
# folder = "Straight_benchmark"
# folder = "MVP_straight_look1000"

max_index = 28000

reward = np.genfromtxt("atcenv/output/"+folder+"/results/reward.csv")[:max_index]
conflict = np.genfromtxt("atcenv/output/"+folder+"/results/conflicts.csv")[:max_index]*150
drift = np.genfromtxt("atcenv/output/"+folder+"/results/drift_angle.csv")[:max_index]*180/np.pi
# q_loss = np.genfromtxt("atcenv/output/"+folder+"/results/q_loss.csv")


ave_window = 100

plt.plot(moving_average(reward,ave_window))

# print(moving_average(reward,100)[1000])
# print(moving_average(conflict,100)[1000])
# print(moving_average(drift,100)[1000])



r_index = np.argmax(moving_average(reward,ave_window))
c_index = np.argmin(moving_average(conflict,ave_window))
d_index = np.argmin(moving_average(drift,ave_window))

print(f'best ao100 reward observed: {np.max(moving_average(reward,ave_window)):.2f}, at index: {r_index} ')
print(f'at this average the number of conflicts was: {moving_average(conflict,ave_window)[r_index]:.5f}')
print(f'and the drift was: {moving_average(drift,ave_window)[r_index]:.2f} \n')

print(f'lowest number of conflicts: {np.min(moving_average(conflict,ave_window)):.5f}, at index: {c_index} ')
print(f'with a drift of: {moving_average(drift,ave_window)[c_index]:.2f} \n')

print(f'lowest drift: {np.min(moving_average(drift,ave_window)):.2f}, at index: {d_index} ')
print(f'with a total number of conflicts of: {moving_average(conflict,ave_window)[d_index]:.5f}')



plt.plot(moving_average(drift,ave_window))
plt.plot(moving_average(conflict,ave_window))
plt.show()

# plt.plot(q_loss)
# plt.yscale('log')
# plt.show()