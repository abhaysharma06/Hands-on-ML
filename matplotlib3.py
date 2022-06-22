import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,6,7,8]
y = [5,6,3,6,2,4,8,6,4]

plt.scatter (x,y,label='skitscat' , color ='red',s= 500)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
