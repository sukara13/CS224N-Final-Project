import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt

# Define the directory path and the list of CSV files
directory_path = '/Users/sukara/Downloads/CS224/ICL/'
csv_files = ['gemma-7b-it-text-user-1.csv',
'gemma-7b-it-text-user-2.csv',
'gemma-7b-it-text-user-4.csv',
'gemma-7b-it-text-user-8-new.csv',
'gemma-7b-it-text-user-16.csv',
'gemma-7b-it-text-user-32.csv',
'gemma-7b-it-text-user-64.csv',
'gemma-7b-it-text-user-128.csv',
'gemma-7b-it-table-user-1-new.csv',
'gemma-7b-it-table-user-2-new.csv',
'gemma-7b-it-table-user-4.csv',
'gemma-7b-it-table-user-8.csv',
'gemma-7b-it-table-user-16.csv',
'gemma-7b-it-table-user-32.csv',
'gemma-7b-it-table-user-64.csv',
'gemma-7b-it-table-user-128.csv'
]

# Define the possible labels
labels = ["unacceptable", "acceptable", "good", "very good"]

# Function to calculate the macro AUC for a given CSV file
# def calculate_macro_auc(file_path):
#     data = pd.read_csv(file_path, header=None)

#     # Extract true labels and predicted labels
#     true_labels = data.iloc[:, 0]
#     pred_labels = data.iloc[:, 1]

#     # Binarize the labels for one-vs-rest calculation
#     true_binary = label_binarize(true_labels, classes=labels)
#     pred_binary = label_binarize(pred_labels, classes=labels)

#     # Calculate macro AUC using one-vs-rest approach
#     auc = roc_auc_score(true_binary, pred_binary, average="macro", multi_class="ovr")

#     return auc

def calculate_macro_auc(file_path):
    data = pd.read_csv(file_path, header=None)
    print(f"Data from {file_path}:\n{data.head()}")  # Debug print

    # Filter out rows where true labels or predicted labels are '*fail*'
    data = data[(data.iloc[:, 0] != '*fail*') & (data.iloc[:, 1] != '*fail*')]

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
auc_scores_text = [0.5461680777558012, 0.6335585049520521, 0.5803560463181101, 0.606907777099349, 
                   0.5239927417995163, 0.674902168046781, 0.598696214615382, 0.5245792909371356]

# Updated Data for Table
num_shots_table = [1, 2, 4, 8, 16, 32, 64, 128]
auc_scores_table = [0.5980768276204045, 0.5885784943955785, 0.6148925918000896, 0.5658514786651474, 
                    0.5, 0.504563512048655, 0.5, 0.5]

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
