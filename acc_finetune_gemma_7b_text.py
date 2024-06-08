import matplotlib.pyplot as plt

# Data
num_shots_text = [16, 128, 512]
accuracy_text = [60.86, 85.94, 91.04]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(num_shots_text, accuracy_text, marker='o', label='Text', color='blue')

# Adding titles and labels
plt.title('Fine-tuning Gemma-7b: Accuracy vs. Number of Shots', fontsize=16)
plt.xlabel('Number of Shots', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xscale('log')
plt.xticks(num_shots_text, num_shots_text)
plt.ylim(0, 100)
plt.yticks(range(0, 101, 10))
plt.tight_layout()
plt.show()
