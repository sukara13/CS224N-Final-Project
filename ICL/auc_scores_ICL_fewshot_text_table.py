import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt

# Define the directory path and the list of CSV files
directory_path = '/Users/sukara/Documents/CS224/'
csv_files = ['Llama-3-70b-chat-hf-text-1.csv',
'Llama-3-70b-chat-hf-text-2.csv',
'Llama-3-70b-chat-hf-text-4.csv',
'Llama-3-70b-chat-hf-text-8.csv',
'Llama-3-70b-chat-hf-text-16.csv',
'Llama-3-70b-chat-hf-text-32.csv',
'Llama-3-70b-chat-hf-text-64.csv',
'Llama-3-70b-chat-hf-text-128.csv',
'Llama-3-70b-chat-hf-table-1.csv',
'Llama-3-70b-chat-hf-table-2.csv',
'Llama-3-70b-chat-hf-table-4.csv',
'Llama-3-70b-chat-hf-table-8.csv',
'Llama-3-70b-chat-hf-table-16.csv',
'Llama-3-70b-chat-hf-table-32.csv',
'Llama-3-70b-chat-hf-table-64.csv',
'Llama-3-70b-chat-hf-table-100.csv'
]

# Define the possible labels
labels = ["unacceptable", "acceptable", "good", "very good"]

# Function to calculate the macro AUC for a given CSV file
def calculate_macro_auc(file_path):
    data = pd.read_csv(file_path, header=None)

    # Extract true labels and predicted labels
    true_labels = data.iloc[:, 0]
    pred_labels = data.iloc[:, 1]

    # Binarize the labels for one-vs-rest calculation
    true_binary = label_binarize(true_labels, classes=labels)
    pred_binary = label_binarize(pred_labels, classes=labels)

    # Calculate macro AUC using one-vs-rest approach
    auc = roc_auc_score(true_binary, pred_binary, average="macro", multi_class="ovr")

    return auc

# Dictionary to store AUC scores for each file
auc_scores = {}

# Calculate AUC for each file and store the results
for file_name in csv_files:
    file_path = os.path.join(directory_path, file_name)
    auc = calculate_macro_auc(file_path)
    auc_scores[file_name] = auc
    print(f"AUC score for {file_name}: {auc}")

# Print all AUC scores
print("\nText then Table AUC scores:")
for file_name, auc in auc_scores.items():
    print(f"{file_name}: {auc}")

# PLOTTING
import matplotlib.pyplot as plt

# Data
num_shots = [1, 2, 4, 8, 16, 32, 64, 128]
auc_scores_text = [0.718074179407511, 0.7240687308911993, 0.726785930010922, 0.7130629241868579, 
                   0.7066585972471436, 0.6948177200377754, 0.7231599695210835, 0.7923480702172089]

# Updated Data for Table
num_shots_table = [1, 2, 4, 8, 16, 32, 64, 100]
auc_scores_table = [0.7214961372392579, 0.6821639792140446, 0.6989341833597554, 0.6519361865140506, 
                    0.6499029039306807, 0.6555042814966191, 0.6724761792359875, 0.7009928532444738]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_shots, auc_scores_text, marker='o', label='Text')
plt.plot(num_shots_table, auc_scores_table, marker='o', label='Table')

# Adding titles and labels
plt.title('AUC Scores for Text vs Table')
plt.xlabel('Number of Shots')
plt.ylabel('AUC Score')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.xticks(num_shots, num_shots)
plt.show()
