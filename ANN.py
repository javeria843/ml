from numpy import *

class NeuralNet(object):
    def __init__(self):
        random.seed(1)
        # Initialize weights randomly with shape (3,1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def _sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, training_iterations):
        for iteration in range(training_iterations):  # Use Python 3's range
            output = self.learn(inputs)
            error = outputs - output
            factor = dot(inputs.T, error * self._sigmoid_derivative(output))
            self.synaptic_weights += factor

    def learn(self, inputs):
        return self._sigmoid(dot(inputs, self.synaptic_weights))


# ---------- RUN THE NEURAL NETWORK ----------
if __name__ == "__main__":
    # Inputs: 3 samples, 3 features each
    inputs = array([[0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1]])

    # Outputs: 3 expected values
    outputs = array([[1, 0, 1]]).T

    # Create and train the neural network
    neural_network = NeuralNet()
    neural_network.train(inputs, outputs, 10000)

    # Test on a new input
    result = neural_network.learn(array([1, 0, 1]))
    print("Prediction for input [1, 0, 1]:", result)
