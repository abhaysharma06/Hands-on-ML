import matlplotlib.pyplot as plt

days =[1,2,3,4,5,6,7]

sleeping = [8,7,6,9,4,11]
eating = [1,2,1,2,3,1]
working = [4,5,4,6,7,8]
playing =[4,5,6,1,2,2]

plt.stackplot(days , sleeping,eating,working,playing)

plt.xlabel()
plt.ylabel()
plt.legend()
plt.show()


