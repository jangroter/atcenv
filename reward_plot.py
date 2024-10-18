import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from atcenv.src.functions import moving_average
import pandas as pd
import seaborn as sns

def moving_std(x, w):
    return np.abs((np.array([np.std(x[i:i+w]) for i in range(len(x) - w + 1)]))/moving_average(x,w))

def moving_iqr_lower(x, w):
    return np.array([np.percentile(x[i:i+w], 25) for i in range(len(x) - w + 1)])

def moving_iqr_upper(x, w):
    return np.array([np.percentile(x[i:i+w], 75) for i in range(len(x) - w + 1)])

# folder = "FeedForward_local_benchmark"
# folder = "Transformer_relative_distance_benchmark_2"
# folder = "Transformer_relative_benchmark_2"
folder = "FeedForward_ownref_straight_Actlrx10"
# folder = "Transformer_ownref_straight"
# folder = "Transformer_ownref_straight_Actlrx10"
# folder = "Transformer5Heads_ownref_straight_Actlrx10"
# folder = "TransformerTest"
folder = "archive/Transformer_ownref_20AC_Actlrx10"
folder = "scaled_dot_3"

folder2 = "archive_2/Transformer_ownref_20AC_3head_additive_basic"
folder3 = "archive_2/Transformer_ownref_20AC_3head_additive_basic_intx10"
# folder4 = "archive/Transformer_ownref_20AC_Actlrx10"
folder4 = "scaled_dot_2"
# folder4 = "archive_2/Transformer_ownref_20AC_3head_neural_net_score_test"

folder3 = "scaled_dot_1"
folder2 = "best/SAC_v5_additive_basic_lower_std_spread"

folder = "final_results/scaled_dot_1"
folder2 = "final_results/scaled_dot_2"
folder3 = "final_results/scaled_dot_3"
folder4 = "final_results/additive_1"
folder5 = "final_results/additive_2"
folder6 = "final_results/additive_3"
folder7 = "final_results/neural_net_1"
folder8 = "final_results/neural_net_2"
folder9 = "final_results/neural_net_3"
folder10 = "LSTM_descending"
folder11 = "LSTM_ascending_correct"
folder12 = "final_results/LSTM_test_2"

# folder = "final_results/LSTM_test_2"
# folder2 = "LSTM_ascending"
# folder = "LSTM_descending"
# folder = "Straight_benchmark"
# folder = "MVP_straight_look1000"

max_index = 5000

conflict = np.genfromtxt("atcenv/output/"+folder+"/results/conflicts.csv")[:max_index]*150
conflict2 = np.genfromtxt("atcenv/output/"+folder2+"/results/conflicts.csv")[:max_index]*150
conflict3 = np.genfromtxt("atcenv/output/"+folder3+"/results/conflicts.csv")[:max_index]*150



conflict4 = np.genfromtxt("atcenv/output/"+folder4+"/results/conflicts.csv")[:max_index]*150
conflict5 = np.genfromtxt("atcenv/output/"+folder5+"/results/conflicts.csv")[:max_index]*150
conflict6 = np.genfromtxt("atcenv/output/"+folder6+"/results/conflicts.csv")[:max_index]*150
conflict7 = np.genfromtxt("atcenv/output/"+folder7+"/results/conflicts.csv")[:max_index]*150
conflict8 = np.genfromtxt("atcenv/output/"+folder8+"/results/conflicts.csv")[:max_index]*150
conflict9 = np.genfromtxt("atcenv/output/"+folder9+"/results/conflicts.csv")[:max_index]*150
conflict10 = np.genfromtxt("atcenv/output/"+folder10+"/results/conflicts.csv")[:max_index]*150
conflict11 = np.genfromtxt("atcenv/output/"+folder11+"/results/conflicts.csv")[:max_index]*150
conflict12 = np.genfromtxt("atcenv/output/"+folder12+"/results/conflicts.csv")[:max_index]*150

drift = np.genfromtxt("atcenv/output/"+folder+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift2 = np.genfromtxt("atcenv/output/"+folder2+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift3 = np.genfromtxt("atcenv/output/"+folder3+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift4 = np.genfromtxt("atcenv/output/"+folder4+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift5 = np.genfromtxt("atcenv/output/"+folder5+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift6 = np.genfromtxt("atcenv/output/"+folder6+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift7 = np.genfromtxt("atcenv/output/"+folder7+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift8 = np.genfromtxt("atcenv/output/"+folder8+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift9 = np.genfromtxt("atcenv/output/"+folder9+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift10 = np.genfromtxt("atcenv/output/"+folder10+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift11 = np.genfromtxt("atcenv/output/"+folder11+"/results/drift_angle.csv")[:max_index]*180/np.pi
drift12 = np.genfromtxt("atcenv/output/"+folder12+"/results/drift_angle.csv")[:max_index]*180/np.pi
# q_loss = np.genfromtxt("atcenv/output/"+folder+"/results/q_loss.csv")
reward = np.genfromtxt("atcenv/output/"+folder+"/results/reward.csv")[:max_index]
reward2 = np.genfromtxt("atcenv/output/"+folder2+"/results/reward.csv")[:max_index]
reward3 = np.genfromtxt("atcenv/output/"+folder3+"/results/reward.csv")[:max_index]
reward4 = np.genfromtxt("atcenv/output/"+folder4+"/results/reward.csv")[:max_index]
reward5 = np.genfromtxt("atcenv/output/"+folder5+"/results/reward.csv")[:max_index]
reward6 = np.genfromtxt("atcenv/output/"+folder6+"/results/reward.csv")[:max_index]
reward7 = np.genfromtxt("atcenv/output/"+folder7+"/results/reward.csv")[:max_index]
reward8 = np.genfromtxt("atcenv/output/"+folder8+"/results/reward.csv")[:max_index]
reward9 = np.genfromtxt("atcenv/output/"+folder9+"/results/reward.csv")[:max_index]
reward10 = np.genfromtxt("atcenv/output/"+folder10+"/results/reward.csv")[:max_index]
reward11 = np.genfromtxt("atcenv/output/"+folder11+"/results/reward.csv")[:max_index]
reward12 = np.genfromtxt("atcenv/output/"+folder12+"/results/reward.csv")[:max_index]

ave_window = 10

plt.figure(figsize=(10,5))
# plt.plot(moving_average(reward,ave_window),'g')
# plt.plot(moving_average(reward2,ave_window),'g')

# plt.plot(moving_average(reward4,ave_window),'orange')
# plt.plot(moving_average(reward5,ave_window),'orange')
plt.plot(moving_average(reward6,ave_window),color=colormaps['tab20c'].colors[0],label='ADD')
# plt.plot(moving_average(reward7,ave_window),'r')
# plt.plot(moving_average(reward8,ave_window),'r')

plt.plot(moving_average(reward9,ave_window),color=colormaps['tab20c'].colors[1],label='CA')
plt.plot(moving_average(reward3,ave_window),color=colormaps['tab20c'].colors[2],label='SD')
plt.plot(moving_average(reward10,ave_window),color=colormaps['tab20c'].colors[4],linestyle='--',label='LSTM Ascending')
plt.plot(moving_average(reward11,ave_window),color=colormaps['tab20c'].colors[5],linestyle='--',label='LSTM Descending')
plt.plot(moving_average(reward12,ave_window),color=colormaps['tab20c'].colors[6],linestyle='--',label='LSTM random')
# plt.axhline(y=-3.65, color='black', linestyle='--')

plt.yscale('symlog')
# from matplotlib.ticker import LogLocator
# plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
# from matplotlib.ticker import AutoMinorLocator
# plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
plt.yticks(ticks=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-20,-30,-40],minor=True)
plt.ylabel('Sum of Rewards')
plt.xlabel('Episode')
plt.legend()
plt.show()

# print(moving_average(reward,100)[1000])
# print(moving_average(conflict,100)[1000])
# print(moving_average(drift,100)[1000])
# plt.plot(moving_average(reward,ave_window),'g')
# plt.plot(moving_average(reward2,ave_window),'g')
# plt.plot(moving_average(reward3,ave_window),linestyle='--',color=(0.12156862745098039, 0.47058823529411764, 0.7058823529411765),label='SD')
# plt.plot(moving_average(reward4,ave_window),'orange')
# plt.plot(moving_average(reward5,ave_window),'orange')
# plt.plot(moving_average(reward6,ave_window),color=(0.12156862745098039, 0.47058823529411764, 0.7058823529411765),label='ADD')
# plt.show()


r_index = np.argmax(moving_average(reward3,ave_window))
c_index = np.argmin(moving_average(conflict3,ave_window))
d_index = np.argmin(moving_average(drift3,ave_window))

print(f'best ao100 reward observed: {np.max(moving_average(reward3,ave_window)):.2f}, at index: {r_index} ')
print(f'at this average the number of conflicts was: {moving_average(conflict3,ave_window)[r_index]:.5f}')
print(f'and the drift was: {moving_average(drift3,ave_window)[r_index]:.2f} \n')

print(f'lowest number of conflicts: {np.min(moving_average(conflict3,ave_window)):.5f}, at index: {c_index} ')
print(f'with a drift of: {moving_average(drift3,ave_window)[c_index]:.2f} \n')

print(f'lowest drift: {np.min(moving_average(drift3,ave_window)):.2f}, at index: {d_index} ')
print(f'with a total number of conflicts of: {moving_average(conflict3,ave_window)[d_index]:.5f}')


plt.figure(figsize=(10,5))
# plt.plot(moving_average(drift,ave_window),'g')
# plt.plot(moving_average(drift2,ave_window),'g')
# plt.plot(moving_average(drift4,ave_window),'orange')
# plt.plot(moving_average(drift5,ave_window),'orange')
plt.plot(moving_average(drift6,ave_window),color=colormaps['tab20c'].colors[0],label='ADD')
# plt.plot(moving_average(drift7,ave_window),'r')
# plt.plot(moving_average(drift8,ave_window),'r')
plt.plot(moving_average(drift9,ave_window),color=colormaps['tab20c'].colors[1],label='CA')
plt.plot(moving_average(drift3,ave_window),color=colormaps['tab20c'].colors[2],label='SD')
plt.plot(moving_average(drift10,ave_window),color=colormaps['tab20c'].colors[4],linestyle='--',label='LSTM Ascending')
plt.plot(moving_average(drift11,ave_window),color=colormaps['tab20c'].colors[5],linestyle='--',label='LSTM Descending')
plt.plot(moving_average(drift12,ave_window),color=colormaps['tab20c'].colors[6],linestyle='--',label='LSTM random')
# plt.axhline(y=12.58, color='black', linestyle='--')
plt.ylabel('Average Track Deviation')
plt.xlabel('Episode')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
# plt.plot(moving_average(conflict,ave_window),'g')
# plt.plot(moving_average(conflict2,ave_window),'g')

# plt.plot(moving_average(conflict4,ave_window),'orange')
# plt.plot(moving_average(conflict5,ave_window),'orange')
plt.plot(moving_average(conflict6,ave_window),color=colormaps['tab20c'].colors[0],label='ADD')
# plt.plot(moving_average(conflict7,ave_window),'r')
# plt.plot(moving_average(conflict8,ave_window),'r')
plt.plot(moving_average(conflict9,ave_window),color=colormaps['tab20c'].colors[1],label='CA')
plt.plot(moving_average(conflict3,ave_window),color=colormaps['tab20c'].colors[2],label='SD')
plt.plot(moving_average(conflict10,ave_window),color=colormaps['tab20c'].colors[4],linestyle='--',label='LSTM Ascending')
plt.plot(moving_average(conflict11,ave_window),color=colormaps['tab20c'].colors[5],linestyle='--',label='LSTM Descending')
plt.plot(moving_average(conflict12,ave_window),color=colormaps['tab20c'].colors[6],linestyle='--',label='LSTM random')
# plt.axhline(y=0.77, color='black', linestyle='--')
plt.yscale('log')
plt.ylabel('Total Number of Intrusions')
plt.xlabel('Episode')
plt.legend()
plt.show()

ave_window = 100
max_index = 1000000

qloss = np.genfromtxt("atcenv/output/"+folder+"/results/q_loss.csv")[:max_index]
qloss2 = np.genfromtxt("atcenv/output/"+folder2+"/results/q_loss.csv")[:max_index]
qloss3 = np.genfromtxt("atcenv/output/"+folder3+"/results/q_loss.csv")[:max_index]
qloss4 = np.genfromtxt("atcenv/output/"+folder4+"/results/q_loss.csv")[:max_index]

plt.plot(moving_average(qloss,ave_window),'b')
plt.plot(moving_average(qloss2,ave_window),'r')
plt.plot(moving_average(qloss3,ave_window),'g')
plt.plot(moving_average(qloss4,ave_window),'orange')
plt.yscale('log')
plt.show()

# plt.plot(q_loss)
# plt.yscale('log')
# plt.show()


# Assuming you've already loaded your data into conflict, conflict2, and conflict3
# Combine the arrays into a single 2D array
# sd_combined_conflicts = np.vstack([moving_average(conflict,ave_window), moving_average(conflict2,ave_window), moving_average(conflict3,ave_window)])

# conflicts_df = pd.DataFrame()
# conflicts_df = conflicts_df.append(conflict)

# Combine the arrays into one array
all_conflicts = np.concatenate([moving_average(conflict,ave_window), moving_average(conflict2,ave_window), moving_average(conflict3,ave_window), moving_average(conflict4,ave_window), moving_average(conflict5,ave_window), moving_average(conflict6,ave_window), moving_average(conflict7,ave_window), moving_average(conflict8,ave_window), moving_average(conflict9,ave_window)])

# Create the index array that resets for each original array's length
index = np.tile(np.arange(1, len(moving_average(conflict,ave_window)) + 1), 9)

# Create the labels array
labels = np.concatenate([
    np.repeat('SD', len(moving_average(conflict,ave_window)) * 3),  # Label 'SD' for conflict, conflict2, conflict3
    np.repeat('ADD', len(moving_average(conflict,ave_window)) * 3),  # Label 'ADD' for conflict4, conflict5, conflict6
    np.repeat('CA', len(moving_average(conflict,ave_window)) * 3)
])

index = ((index+5) // 10) * 10


# Create the DataFrame
df = pd.DataFrame({'Index': index, 'Conflict': all_conflicts, 'Label': labels})
plt.figure(figsize=(10,5))
sns.lineplot(df,x='Index',y='Conflict',hue='Label',errorbar=('pi',50),err_style='bars')  
plt.yscale('log')
plt.show()

# Combine the arrays into one array
all_rewards = np.concatenate([moving_average(reward,ave_window), moving_average(reward2,ave_window),moving_average(reward3,ave_window), moving_average(reward4,ave_window), moving_average(reward5,ave_window), moving_average(reward6,ave_window), moving_average(reward7,ave_window), moving_average(reward8,ave_window), moving_average(reward9,ave_window)])

# Create the index array that resets for each original array's length
index = np.tile(np.arange(1, len(moving_average(reward,ave_window)) + 1), 9)

# Create the labels array
labels = np.concatenate([
    np.repeat('SD', len(moving_average(reward,ave_window)) * 3),  # Label 'SD' for conflict, conflict2, conflict3
    np.repeat('ADD', len(moving_average(reward,ave_window)) * 3),  # Label 'ADD' for conflict4, conflict5, conflict6
    np.repeat('CA', len(moving_average(reward,ave_window)) * 3)
])

index = ((index+5) // 10) * 10

# Create the DataFrame
df = pd.DataFrame({'Episode': index, 'Sum of Rewards': all_rewards, 'Method': labels})
plt.figure(figsize=(10,5))
sns.lineplot(df,x='Episode',y='Sum of Rewards',hue='Method',errorbar=('pi',50),palette=[colormaps['tab20c'].colors[1],colormaps['tab20c'].colors[0],colormaps['tab20c'].colors[4]])
# plt.yscale('symlog')
# plt.yticks(ticks=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-20,-30,-40],minor=True)
plt.show()


# Combine the arrays into one array
all_rewards = np.concatenate([moving_average(reward4,ave_window), moving_average(reward5,ave_window), moving_average(reward6,ave_window), moving_average(reward7,ave_window), moving_average(reward8,ave_window), moving_average(reward9,ave_window)])

# Create the index array that resets for each original array's length
index = np.tile(np.arange(1, len(moving_average(reward,ave_window)) + 1), 6)

# Create the labels array
labels = np.concatenate([
    np.repeat('ADD', len(moving_average(reward,ave_window)) * 3),  # Label 'ADD' for conflict4, conflict5, conflict6
    np.repeat('CA', len(moving_average(reward,ave_window)) * 3)
])

index = ((index+5) // 10) * 10

# Create the DataFrame
df = pd.DataFrame({'Index': index, 'Reward': all_rewards, 'Label': labels})
plt.figure(figsize=(10,5))
sns.lineplot(df,x='Index',y='Reward',hue='Label',errorbar=('pi',50))
# plt.yscale('symlog')
plt.show()


# Combine the arrays into one array
all_drift = np.concatenate([moving_average(drift,ave_window), moving_average(drift2,ave_window), moving_average(drift3,ave_window), moving_average(drift4,ave_window), moving_average(drift5,ave_window), moving_average(drift6,ave_window), moving_average(drift7,ave_window), moving_average(drift8,ave_window), moving_average(drift9,ave_window)])

# Create the index array that resets for each original array's length
index = np.tile(np.arange(1, len(moving_average(drift,ave_window)) + 1), 9)

# Create the labels array
labels = np.concatenate([
    np.repeat('SD', len(moving_average(drift,ave_window)) * 3),  # Label 'SD' for conflict, conflict2, conflict3
    np.repeat('ADD', len(moving_average(drift,ave_window)) * 3),  # Label 'ADD' for conflict4, conflict5, conflict6
    np.repeat('CA', len(moving_average(drift,ave_window)) * 3)
])

index = ((index+5) // 10) * 10

# Create the DataFrame
df = pd.DataFrame({'Index': index, 'Average track deviation': all_drift, 'Label': labels})
plt.figure(figsize=(10,5))
sns.lineplot(df,x='Index',y='Average track deviation',hue='Label',errorbar=('pi',50))
plt.yscale('log')
plt.show()


# # Calculate the mean and standard deviation (or standard error)
# sd_mean_conflict = np.mean(sd_combined_conflicts, axis=0)
# sd_std_conflict = np.std(sd_combined_conflicts, axis=0)  # Use this for standard deviation as error bars
# sd_stderr_conflict = sd_std_conflict / np.sqrt(sd_combined_conflicts.shape[0])  # Use this for standard error as error bars

# # Create the line plot with error bars
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=np.arange(len(sd_mean_conflict)), y=sd_mean_conflict, label="Mean Conflicts")
# plt.fill_between(np.arange(len(sd_mean_conflict)), sd_mean_conflict - sd_std_conflict, sd_mean_conflict + sd_std_conflict, alpha=0.3)

# # Labels and title
# plt.xlabel('Steps')
# plt.ylabel('Conflicts')
# plt.title('Conflicts over Time with Error Bars')
# plt.legend()
# plt.show()