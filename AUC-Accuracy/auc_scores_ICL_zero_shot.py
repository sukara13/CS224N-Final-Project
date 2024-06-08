import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import os

# Define the directory path and the list of CSV files
directory_path = '/Users/sukara/Documents/CS224/'
csv_files = [
    'Llama-3-70b-chat-hf-text-0.csv',
    'Llama-3-70b-chat-hf-csv-0.csv',
    'Llama-3-70b-chat-hf-table-0.csv',
    'Llama-3-70b-chat-hf-div-0.csv',
    'Llama-3-70b-chat-hf-json-0.csv',
    'Llama-3-70b-chat-hf-xml-0.csv',
    'Llama-3-70b-chat-hf-yaml-0.csv'
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
print("\nAll AUC scores:")
for file_name, auc in auc_scores.items():
    print(f"{file_name}: {auc}")

