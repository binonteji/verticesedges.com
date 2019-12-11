import numpy as np

class NAND_NN():

	def __init__(self):
		#np.random.seed(1)
		self.learning_rate = 0.1

		# initialize the weights matrix of 2 * 1 between -1 and +1
		self.weights = np.random.rand(2, 1) * 2 - 1

		# will get 1*1 matrix of bias value	 between 0.0 and 1.0
		self.bias = np.random.uniform(0.0, 1.0, (1, 1))    

	def sigmoid(self, a):
		return 1 / (1 + np.exp(-a))

	def sderivative(self, b):
		return b * (1 - b)

	def train(self, inputs, output, iterations):
		for _ in range(iterations):

			# Feedforward
			layer = np.dot(inputs, self.weights) + self.bias
			layer_output = self.sigmoid(layer)

			# Backpropagation starts here
			error_output = output - layer_output

			# how much error of the network should be adjusted
			delta_error_output = error_output * self.sderivative(layer_output) 

			# Weight and Bias updation
			self.weights += self.learning_rate * np.dot(inputs.T, delta_error_output)
			self.bias += self.learning_rate * np.sum(delta_error_output)

		return layer_output
	
if __name__ == '__main__':

	training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	training_outputs = np.array([[1], [1], [1], [0]])
	training_iterations = 50000

	nn = NAND_NN()

	print('\nWeights before training : ', *nn.weights)
	print('Bias before training : ', *nn.bias)
	predicted_result = nn.train(training_inputs, training_outputs, training_iterations)
	print('\nWeights after training : ', *nn.weights)
	print('Bias before training : ', *nn.bias)
	print('Predicted Output : ', *predicted_result)
