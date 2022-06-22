import matplotlib.pyplot as plt

x = [1,2,3]
y = [4,5,7]

x2 = [2,3,4]
y2 = [4,5,6]

plt.plot(x,y , label= 'First Line')
plt.plot(x2,y2,label='Second Line')
plt.xlabel('PLOT NUMBER')
plt.ylabel('IMPORTANT NUMBER')
plt.title('Intresting Graph \n Check it out')
plt.legend()
plt.show()
