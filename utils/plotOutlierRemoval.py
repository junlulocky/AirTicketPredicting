import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

## the data
N=1000
x = np.random.randn(600)
y = np.random.randn(600)

## left panel
cluster1 = plt.scatter(x,y,color='blue',s=15,edgecolor='none')

x = np.random.randn(300)
y = np.random.randn(300)
cluster2 = plt.scatter(x+3,y+3,color='green',s=20,edgecolor='none', marker='D')

circle=plt.Circle((-0.5,-0.5),3.5, fill=False, color='b')
circle2=plt.Circle((3.2,3.2),3.5, fill=False, color='green')
fig = plt.gcf()
#fig.gca().add_artist(circle)
#fig.gca().add_artist(circle2)
plt.legend([cluster1, cluster2], ['Cluster Wait', 'Cluster Buy'])

frame1 = plt.gca()
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.show()
