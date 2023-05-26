import matplotlib.pyplot as plt

data1 = [11, 12, 13, 14, 15, 16, 17]
data2 = [15.5, 12.5, 11.7, 9.50, 12.50, 11.50, 14.75]

# Add labels and title
plt.title("Interactive Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Set the limit for each axis
plt.xlim(11, 17)
plt.ylim(9, 16)

# Plot a line graph
plt.plot(data1, data2)

plt.show()
