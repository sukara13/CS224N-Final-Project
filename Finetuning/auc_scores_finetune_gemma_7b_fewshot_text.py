import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import os
import matplotlib.pyplot as plt

# Define the directory path and the list of CSV files
directory_path = '/Users/sukara/Downloads/CS224/'
csv_files = ['sukara13_gemma7bcars16-text.csv',
'sukara13_gemma7bcars128-text.csv',
'sukara13_gemma7bcars512-text.csv'
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
num_shots_text = [16, 128, 512]
accuracy_text = [60.65, 78.21, 89.11]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_shots_text, accuracy_text, marker='o', label='Text', color='blue')

# Adding titles and labels
plt.title('Fine-tuning Gemma-7b: AUC Score vs. Number of Shots', fontsize=16)
plt.xlabel('Number of Shots', fontsize=14)
plt.ylabel('AUC Score', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xscale('log')
plt.xticks(num_shots_text, num_shots_text)
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
plt.tight_layout()
plt.show()

