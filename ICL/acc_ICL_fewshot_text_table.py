import matplotlib.pyplot as plt

# Data
num_shots_text = [1, 2, 4, 8, 16, 32, 64, 128]
accuracy_text = [64.79, 51.62, 55.68, 54.07, 65.89, 67.16, 65.62, 74.06]

num_shots_table = [1, 2, 4, 8, 16, 32, 64, 100]
accuracy_table = [67.40, 49.25, 63.98, 55.47, 61.86, 62.44, 61.42, 63.57]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_shots_text, accuracy_text, marker='o', label='Text')
plt.plot(num_shots_table, accuracy_table, marker='o', label='Table')

# Adding titles and labels
plt.title('Accuracy for Text vs Table')
plt.xlabel('Number of Shots')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.xticks(num_shots_text, num_shots_text)
plt.show()
