# Simple perceptron simulator for kids

This is one of my hobby projects. I have built a very simple perceptron that can classify linearly seperable data. It is built from the scratch, even numpy has not been used. It helped me to understand how perceptron really works. I used matplotlib animation to render the steps of learning.

In the code, a linearly seperated dataset is created based on function f(x,y), i.e. f(x,y) = 7x-3y+5=0, which seperates the data into two classes. Then the perceptron starts with random values of weights and bias which has no idea about the classes. Then it adjusts the values iteratively by correcting itself. Finally it comes up with a decision and draws a line to seperate two classes.

![Alt text](https://github.com/imruljubair/Simple-perceptron-simulator-for-kids/blob/master/perceptron.gif)

I am thankful to: https://youtu.be/XJ7HLz9VYz0 which provides me the basic of perceptron.

