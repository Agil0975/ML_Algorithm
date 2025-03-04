import numpy as np
import time

class ActivationFunction:
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_prime(self, z):
        return np.where(z > 0, 1, 0)

class FFNN:
    def __init__(self, layers: list):
        """ Initialize the Feed Forward Neural Network

        Args:
            layers (list): list of integers representing the number of neurons in each layer (including input layer)
        """
        self.layers = layers
        self.biases = [np.random.randn(layers[i]) for i in range(1,len(layers))]
        self.weights = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
        self.activation = ActivationFunction()

    def feed_forward(self, a):
        """ Feed Forward Neural Network

        Args:
            a (np.array): input array

        Returns:
            np.array: output array
        """
        for w, b in zip(self.weights, self.biases):
            a = self.activation.sigmoid(np.dot(w, a) + b)
            # a = self.activation.relu(np.dot(w, a) + b)
        return a
    
    def back_propagation(self, x, y):
        """ Back Propagation

        Args:
            x (np.array): input array
            y (np.array): output array
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]      # Gradient of each bias
        nabla_w = [np.zeros(w.shape) for w in self.weights]     # Gradient of each weight
        
        # Feed Forward
        a = x
        a_list = [x]    # List of activations
        z_list = []     # List of linear combinations
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            z_list.append(z)
            a = self.activation.sigmoid(z)
            # a = self.activation.relu(z)
            a_list.append(a)

        # Backward Pass
        nabla_a = [np.zeros(a.shape) for a in a_list]       # Gradient of each activation
        for i in range(len(self.layers)-1, 0, -1):
            if i == len(self.layers)-1:
                nabla_a[i] = -2 * (y - a_list[i]) * self.activation.sigmoid_prime(z_list[i-1])
                # nabla_a[i] = -2 * (y - a_list[i]) * self.activation.relu_prime(z_list[i-1])
            else:
                weight = self.weights[i]
                nabla_a[i] = np.dot(weight.T, nabla_a[i+1]) * self.activation.sigmoid_prime(z_list[i-1])
                # nabla_a[i] = np.dot(weight.T, nabla_a[i+1]) * self.activation.relu_prime(z_list[i-1])

            nabla_b[i-1] = nabla_a[i]
            nabla_w[i-1] = a_list[i-1] * nabla_a[i].reshape(-1,1)

        return (nabla_b, nabla_w)
    
    def train(self, x, y, learning_rate, epochs):
        """ Train the Feed Forward Neural Network

        Args:
            x (np.array): input array
            y (np.array): output array
            learning_rate (float): learning rate
            epochs (int): number of epochs
        """
        for i in range(epochs):
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for j in range(len(x)):
                delta_nabla_b, delta_nabla_w = self.back_propagation(x[j], y[j])
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w - (learning_rate / len(x)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (learning_rate / len(x)) * nb for b, nb in zip(self.biases, nabla_b)]

def main():
    x = np.array([[0,0], [0,1], [1,0], [1,1]])

    y = np.array([0,0,0,1]).reshape(-1,1)   # AND
    y = np.array([0,1,1,1]).reshape(-1,1)   # OR
    # y = np.array([0,1,1,0]).reshape(-1,1)   # XOR
    # y = np.array([1,0,0,1]).reshape(-1,1)   # XNOR
    # y = np.array([1,0,0,1]).reshape(-1,1)   # NAND
    # y = np.array([1,0,0,0]).reshape(-1,1)   # NOR
    
    nn = FFNN([2,2,1])
    start = time.time()
    nn.train(x, y, 0.1, 100000)
    print(time.time() - start)

    for i in range(len(x)):
        print(nn.feed_forward(x[i]))

    # for w, b in zip(nn.weights, nn.biases):
    #     print(w, b)

if __name__ == "__main__":
    main()