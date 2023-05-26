import matplotlib.pyplot as plt

year = [2001, 2002, 2003, 2004, 2005, 2006]
unit = [50, 60, 75, 45, 70, 105]

# Plot the bar graph
plot = plt.bar(year, unit)

# Add the data value on head of the bar
for value in plot:
	height = value.get_height()
	plt.text(value.get_x() + value.get_width() / 2.,
	         1.002 * height, '%d' % int(height), ha='center', va='bottom')

# Add labels and title
plt.title("Bar Chart")
plt.xlabel("Year")
plt.ylabel("Unit")

# Display the graph on the screen
plt.show()
