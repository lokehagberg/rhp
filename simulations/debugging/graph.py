import matplotlib.pyplot as plt
import numpy as np

# Create NumPy arrays for the x-values and y-values
x = np.arange(1, 11)
y = np.zeros(10)

# Create a Matplotlib figure and axis
fig, ax = plt.subplots()

# Plot the lines
for i in range(10):
    y[i] = 1
    ax.plot(x, y, label=f'Line {i+1}')
    y[i] = 0

# Add a legend and set the title
ax.legend()
ax.set_title('Ten Lines with One Nonzero Value')

# Show the plot
plt.show()
