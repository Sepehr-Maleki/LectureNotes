import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
plt.style.use('fivethirtyeight')


# Generte some random data of length n
n = 20
range_min = 0
range_max = 50
loc = (range_max - range_min)/2 # Location of the unknown point
c = 'k' # Colour of the unknown point
blue_x = [np.random.randint(range_min, range_max) for i in range(n)]
blue_y = [np.random.randint(range_min, range_max) for i in range(n)]
red_x = [np.random.randint(range_min, range_max) for i in range(n)]
red_y = [np.random.randint(range_min, range_max) for i in range(n)]
blues = [(i, j) for i, j in zip(blue_x, blue_y)]
reds = [(i, j) for i, j in zip(red_x, red_y)]


# KNN
K=6
## We want to classify a black dot to be coloured either red or blue based on its K-nearest neighbours.
## The black dot is located at loc = (range_max - range_min)/2.

## We start by measuring the distance from the black dot to all other points.
blue = [(i, j) for i, j in zip(blue_x, blue_y)]
red = [(i, j) for i, j in zip(red_x, red_y)]
black = (loc, loc)

black_from_blue = [(np.sqrt((x[0] - loc)**2 + (x[1] - loc)**2), (x[0], x[1])) for x in blue]
black_from_red = [(np.sqrt((x[0] - loc)**2 + (x[1] - loc)**2), (x[0], x[1])) for x in red]

black_from_blue.extend(black_from_red)
black_from_blue.sort()

knn = black_from_blue[0:K]
r = knn[-1][0]
circle = plt.Circle((loc, loc), r, color=c, fill=False)


# Determine the black point is closer to more blue points or red points
close_to_blues = sum(knn[k][1] in blues for k in range(K))
close_to_reds = sum(knn[k][1] in reds for k in range(K))
print(close_to_reds, close_to_blues)


# Determine which class the black point belongs to
if close_to_blues > close_to_reds:
	c='b'
elif close_to_blues < close_to_reds:
	c='r'
else:
	c='k'


# Animation 
plt.ion()
plt.figure(figsize=(10, 10))
plt.scatter(blue_x, blue_y, color='b')
plt.scatter(red_x, red_y, color='r')
plt.scatter(loc, loc, color='k', s=200)
plt.Circle((loc,loc),100*r)
plt.xticks([], [])
plt.yticks([], [])
plt.title("Demonstration of KNN for K={0}".format(K))
plt.pause(1)
text = plt.text(loc-.4, loc+1, "?", fontsize=18)
plt.pause(2)
for i in range(n):
	arr = plt.arrow(loc, loc, blue_x[i]-loc, blue_y[i]-loc, ec='b')
	plt.pause(.1)
	arr.remove()
	arr = plt.arrow(loc, loc, red_x[i]-loc, red_y[i]-loc, ec='r')
	plt.pause(.1)
	arr.remove()

fig = plt.gcf()
ax = fig.gca()
ax.add_artist(circle)
plt.pause(1)
text.remove()
plt.scatter(loc, loc, color=c, s=200)
plt.show(block=True)


