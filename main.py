## Copyright 2018 Mohammad Imrul Jubair

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

fig = plt.figure()
plt.suptitle('Perceptron Simulator')
plt.subplots_adjust(wspace=0.5, top=0.8)
ax0 = fig.add_subplot(121)
ax1 = fig.add_subplot(122)
ax0.set_xlim(-100, 100)
ax0.set_ylim(-100, 100)
ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)
fig.set_figheight(6)
fig.set_figwidth(12)

line1, = ax0.plot([], [],'--', color='black')


class Perceptron:
    lernRate = 0.1

    def __init__(self, s):
        self.weight = []
        for i in range(s):
            self.weight.append(random.uniform(-1,1))

    def predict(self, data):
        y = 0
        for i in range(len(self.weight)):
            y = y + self.weight[i] * data[i]

        return activation(y)

    def training(self, data, label):
        guess = self.predict(data)
        error = label - guess

        for i in range(len(self.weight)):
            self.weight[i] = self.weight[i] + error * data[i] * self.lernRate


class Data:

    def __init__(self, num):
        self.dataPoints = []
        for k in range(num):
            self.dataPoints.append(random.uniform(-100,100))
            
        self.trainData = []

        for i in range(len(self.dataPoints)):
            if f(self.dataPoints[0], self.dataPoints[1]) > 0:
                self.trainData = [self.dataPoints[0], self.dataPoints[1], 1]
            else:
                self.trainData = [self.dataPoints[0], self.dataPoints[1], 0]


def activation(y):
    if y > 1:
        return 1
    else:
        return 0

def f(x,y):
	return 7*x-3*y+5

def f_1(x):
	return (7/3)*x+(5/3)

def createdata(numOfPoints):
    dataset = []
    for i in range(0, numOfPoints):
        d = Data(2)
        dataset.append(d.trainData)

    with open('listfile.txt', 'w') as filehandle:
        json.dump(dataset, filehandle)


def predictline(x, weights):
    yp = []
    for k in range(len(weights)):
        weight0 = weights[k][0]
        weight1 = weights[k][1]
        bias = weights[k][2]
        y = -(weight0/weight1)*x + bias
        yp.append(y)
    return yp

def update(i, x, y):
    #print(i+1)
    line1.set_data(x[i], y[i])
    return line1,


def main():
	
    numOfPoints = 400 #specify the number of datapoints for training 
	
	#uncomment the following line if you want to recreate the dataset
    #createdata(numOfPoints) 

	
    with open('listfile.txt', 'r') as filehandle: #storing the dataset
        dataset = json.load(filehandle)

	# Ploting the datapoints from the dataset:
	
    for i in range(0, numOfPoints):
        if dataset[i][2] == 1:
            ax0.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=7, mec='black')
        else:
            ax0.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=7, mec='black')

    # Perceptron starts...
    
    percep = Perceptron(3) # perceptron with 3 weights (2 weights + 1 bias)
    biasValue = 1 # default bias = 1
    allWeights = []

	# Adding subtitle
    ax0.set_title('Being trained with initial weights:\n ('+str(format(percep.weight[0],'.2f'))+
							', '+ str(format(percep.weight[1],'.2f'))+', '
								+ str(format(percep.weight[2],'.2f'))+')'
								, fontsize=10)
    
    # Perceptron is trained:  
    for k in range(0, numOfPoints):
         percep.training([dataset[k][0], dataset[k][1], biasValue], dataset[k][2])
         allWeights.append([percep.weight[0], percep.weight[1], percep.weight[2]])
         

    with open('weightfile.txt', 'w') as filehandle:
        json.dump(allWeights, filehandle)


	# Unkwon data to be trained by perceptron:
    inputData = [random.uniform(-100, 100), random.uniform(-100, 100), biasValue]
    cls = percep.predict(inputData)
    print('class: '+ str(cls))
    
    
    # Code for animation:    
    xp1 = -100
    xp2 = 100
    yp1 = predictline(xp1, allWeights)
    yp2 = predictline(xp2, allWeights)

    x = []
    y = []
    indx = []

    for i in range(len(allWeights)):
        x.append([xp1, xp2])
        y.append([yp1[i], yp2[i]])
        indx.append(i)
        
    ani = animation.FuncAnimation(fig, update, frames=indx, fargs=(x, y,), 
									blit=True, interval=50, repeat=False)
    for i in range(0, numOfPoints):
        guess = percep.predict([dataset[i][0], dataset[i][1], biasValue])
    
        if dataset[i][2] == 1:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=7, mec='black')
        else:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=7, mec='black')
            
        if guess == 1:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=3, mec='black')
        else:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=3, mec='black')
            
    ax1.plot(x[-1],y[-1],'--', color='black')
    
    ax1.plot([xp1, xp2], [f_1(xp1),f_1(xp2)], '-', color='green')
 
    if cls==1:
        ax1.plot(inputData[0], inputData[1], 's', color='red', markersize=8, mec='black')
    else:
        ax1.plot(inputData[0], inputData[1], 's', color='yellow', markersize=8, mec='black')
    
    ax1.set_title('Trained with adjusted weights:\n ('+str(format(percep.weight[0],'.2f'))
													+', '+ str(format(percep.weight[1],'.2f'))+', '
													+ str(format(percep.weight[2],'.2f'))+')'
													, fontsize=10)
	
    
    plt.show()
    #ani.save('perceptron.gif', dpi=80, writer='imagemagick')
    
    
    

main()
