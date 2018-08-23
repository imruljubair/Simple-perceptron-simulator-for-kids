import matplotlib; matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json

fig = plt.figure()
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(1,2,2)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax1.set_xlim(-100, 100)
ax1.set_ylim(-100, 100)
fig.set_figheight(5)
fig.set_figwidth(10)

line1, = ax1.plot([], [],'--', color='black')

class Perceptron:
    lernRate = 0.1

    def __init__(self, s):
        self.weight = np.random.uniform(-1, 1, size=s)
        print("Initialized weights: " + str(self.weight[0]) + " and " + str(self.weight[1]))
        print("Initialized bias: " + str(self.weight[2]))

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
        self.dataPoints = np.random.uniform(-100, 100, size=num)
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
    print(i+1)
    line1.set_data(x[i], y[i])
    return line1,


def main():
    numOfPoints = 400

    #createdata(numOfPoints)

    with open('listfile.txt', 'r') as filehandle:
        dataset = json.load(filehandle)

    for i in range(0, numOfPoints):
        if dataset[i][2] == 1:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=7, mec='black')
        else:
            ax1.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=7, mec='black')

    percep = Perceptron(3)
    biasValue = 1
    allWeights = []
    
    for k in range(0, numOfPoints):
         percep.training([dataset[k][0], dataset[k][1], biasValue], dataset[k][2])
         allWeights.append([percep.weight[0], percep.weight[1], percep.weight[2]])

    print("Adjusted weights: " + str(percep.weight[0]) + " and " + str(percep.weight[1]))
    print("Adjusted bias: " + str(percep.weight[2]))

    with open('weightfile.txt', 'w') as filehandle:
        json.dump(allWeights, filehandle)

    
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
            ax.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=7, mec='black')
        else:
            ax.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=7, mec='black')
            
        if guess == 1:
            ax.plot(dataset[i][0], dataset[i][1], 'o', color='red', markersize=3, mec='black')
        else:
            ax.plot(dataset[i][0], dataset[i][1], 'o', color='yellow', markersize=3, mec='black')
            
    ax.plot(x[-1],y[-1],'-')

    plt.show()
    
    

main()
