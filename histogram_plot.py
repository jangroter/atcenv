import numpy as np
import matplotlib.pyplot as plt
from atcenv.src.functions import moving_average
import pandas as pd
import seaborn as sns

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

ave_window = 500


# Creating list of the arrays
arrays = [conflict3, conflict6, conflict9, conflict10, conflict11, conflict12]
labels = ['SD', 'ADD', 'CA', 'LSTM Ascending', 'LSTM Descending', 'LSTM random']

# Initialize an empty DataFrame
data = pd.DataFrame()

# Split each array into segments and add to the DataFrame
for i, arr in enumerate(arrays):
    for j in range(5):
        start = j * 1000
        end = (j + 1) * 1000
        segment = arr[start:end]
        temp_df = pd.DataFrame({
            'Value': segment,
            'Episode': f'{start}-{end}',
            'Conflict': labels[i]
        })
        data = pd.concat([data, temp_df])

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Episode', y='Value', hue='Conflict', data=data)

# Set the labels and title
plt.xlabel('Number of Episodes')
plt.ylabel('Total number of Intrusions')
plt.title('Boxplot of Intrusions by Conflict and Episode')
plt.legend(title='Method', loc='upper right')
# plt.yscale('log')
# Show plot
plt.show()



# Creating list of the arrays
arrays = [reward3, reward6, reward9, reward10, reward11, reward12]
labels = ['SD', 'ADD', 'CA', 'LSTM Ascending', 'LSTM Descending', 'LSTM random']

# Initialize an empty DataFrame
data = pd.DataFrame()

# Split each array into segments and add to the DataFrame
for i, arr in enumerate(arrays):
    for j in range(5):
        start = j * 1000
        end = (j + 1) * 1000
        segment = arr[start:end]
        temp_df = pd.DataFrame({
            'Value': segment,
            'Episode': f'{start}-{end}',
            'Conflict': labels[i]
        })
        data = pd.concat([data, temp_df])

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Episode', y='Value', hue='Conflict', data=data)

# Set the labels and title
plt.xlabel('Number of Episodes')
plt.ylabel('Total Reward')
plt.title('Rewards over Episodes')
plt.legend(title='Method', loc='lower right')
plt.yscale('symlog')
# Show plot
plt.show()

# Creating list of the arrays
arrays = [drift3, drift6, drift9, drift10, drift11, drift12]
labels = ['SD', 'ADD', 'CA', 'LSTM Ascending', 'LSTM Descending', 'LSTM random']

# Initialize an empty DataFrame
data = pd.DataFrame()

# Split each array into segments and add to the DataFrame
for i, arr in enumerate(arrays):
    for j in range(5):
        start = j * 1000
        end = (j + 1) * 1000
        segment = arr[start:end]
        temp_df = pd.DataFrame({
            'Value': segment,
            'Episode': f'{start}-{end}',
            'Conflict': labels[i]
        })
        data = pd.concat([data, temp_df])

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Episode', y='Value', hue='Conflict', data=data)

# Set the labels and title
plt.xlabel('Number of Episodes')
plt.ylabel('Average Track Deviation')
plt.title('Track Deviation over Episodes')
plt.legend(title='Method', loc='upper right')

# Show plot
plt.show()