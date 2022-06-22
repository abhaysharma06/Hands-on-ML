import matplotlib.pyplot as plt

'''x = [2,4,6,8,10]
y = [6,7,8,9,10]
                          bar graph
x2 = [3,5,7,9,12]
y2 = [2,3,4,5,6]

plt.bar(x,y,label='bar1')
plt.bar(x2,y2,label='bar2')
'''

population_ages = [22,34,55,76,34,21,32,11,34,10,34,23,12,13,43,42,21,32,53,32,21,54,55,75,64,76,75,64,65,76,31,54,23,32,12,12,43,42,43,43,54,13]
ids = [x for x in range(len(population_ages))]
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population_ages,bins,histtype='bar',rwidth=0.8,label='hist1')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Bar Graph')
plt.legend()
plt.show()
