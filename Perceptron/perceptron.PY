import numpy as np

w = [0.0,0.0,0.0,0.0,0.0]
x = [   
    [1,     5.1,    3.5,    1.4,    0.2],
    [1,     4.9,    3,      1.4,    0.2],
    [1,     4.7,    3.2,    1.3,    0.2],
    [1,     7,      3.2,    4.7,    1.4],
    [1,     6.4,    3.2,    4.5,    1.5],
    [1,     6.9,    3.1,    4.9,    1.5]
]
y = [1,1,1,-1,-1,-1]

threshold = 0.05
learning_rate = 0.1
max_epoch = 10

def predict(x):
    sum = 0
    for i in range(len(w)):
        sum += w[i]*x[i]
    # return 1 if sum > 0 else -1
    return sum

# Perceptron Learning Rule
def PerceptronLearningRule(x,y,w):
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        print("Epoch",epoch)

        error = 0
        for i in range(len(x)):
            prediction = predict(x[i])
            delta = y[i] - prediction
            error += delta * delta / 2
            print("Error",error,"Weight",w, "Prediction", prediction)
            for j in range(len(w)):
                w[j] += learning_rate * delta * x[i][j]
                w[j] = round(w[j],2)

        if error < threshold:
            break
    return w

# w = PerceptronLearningRule(x,y,w)
# print("Final Weight",w)

# Perceptron Batch Gradient Descent
def PerceptronBatchGradientDescent(x,y,w):
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        print("Epoch",epoch)

        error = 0
        delta_W = [0,0,0,0,0]
        for i in range(len(x)):
            prediction = predict(x[i])
            delta = y[i] - prediction
            error += delta * delta / 2
            for j in range(len(w)):
                delta_W[j] += learning_rate * delta * x[i][j]
            print("Error",error,"Weight",delta_W, "Prediction", prediction)
        
        for j in range(len(w)):
            w[j] += delta_W[j]

        if error < threshold:
            break
    return w

# w = PerceptronBatchGradientDescent(x,y,w)
# print("Final Weight",w)

# Perceptron Stochastic Gradient Descent
def PerceptronStochasticGradientDescent(x,y,w):
    epoch = 0
    while epoch < max_epoch:
        epoch += 1
        print("Epoch",epoch)

        error = 0
        for i in range(len(x)):
            prediction = predict(x[i])
            delta = y[i] - prediction
            error += delta * delta / 2
            print("Error",error,"Weight",w, "Prediction", prediction)
            for j in range(len(w)):
                w[j] += learning_rate * delta * x[i][j]

        if error < threshold:
            break

    return w

w = PerceptronStochasticGradientDescent(x,y,w)
print("Final Weight",w)