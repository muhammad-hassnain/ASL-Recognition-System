import pandas as pd
train_df = pd.read_csv('dataset/sign_mnist_train.csv')

label_counts_train = train_df['label'].value_counts()

label_counts_train = label_counts_train.sort_index()

print(label_counts_train)


test_df = pd.read_csv('dataset/sign_mnist_test.csv')

label_counts_test = test_df['label'].value_counts()

label_counts_test = label_counts_test.sort_index()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure
plt.figure(figsize=(10, 6))

# Define the width of the bars and the positions for each set of bars
bar_width = 0.35
x = np.arange(len(label_counts_train))  # X-axis positions for labels

# Create the bar chart for training data with a soothing color (light blue)
plt.bar(x - bar_width / 2, label_counts_train.values, width=bar_width, label='Train', color='#5A99B2')  # Light blue

# Create the bar chart for test data with a soothing color (light green)
plt.bar(x + bar_width / 2, label_counts_test.values, width=bar_width, label='Test', color='#A1D79B')  # Light green

# Adding titles and labels
# plt.title('Grouped Label Distribution in Training and Test Data', fontsize=16)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Customize x-ticks
plt.xticks(x, label_counts_train.index)  # Labels from 0 to 25

# Adding legend
plt.legend(title='Dataset')

# Save the figure as a PNG file
plt.tight_layout()
plt.savefig('label_distribution.png', format='png')

# Show the plot
plt.show()