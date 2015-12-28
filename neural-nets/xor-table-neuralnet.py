from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# The following program will construct and train a neural network
# to predict values for the XOR logical operator.
#
#	XOR Table
#	p | q | p ^ q
#   -------------
#	0 | 0 |   0
#	0 | 1 |   1
#	1 | 0 |   1
#	1 | 1 |   0

# create neural network with 2 neurons in input layer, 3 in hidden layer, 1 in output layer
neuralNetwork = buildNetwork(2, 3, 1)

# use 2 dimensions for input and 1 for output
dataSet = SupervisedDataSet(2, 1)

# add XOR table data
dataSet.addSample((0, 0), (0,))
dataSet.addSample((0, 1), (1,))
dataSet.addSample((1, 0), (1,))
dataSet.addSample((1, 1), (0,))

trainer = BackpropTrainer(neuralNetwork, dataSet)

# train the neural network 100,000 times and show progress every 1000 iterations
for i in range (0, 10000):
	trainer.train()

	if i % 1000 == 0:
		print("Iteration " + str(i) + ":")
		print(neuralNetwork.activate([0, 0]))
		print(neuralNetwork.activate([0, 1]))
		print(neuralNetwork.activate([1, 0]))
		print(neuralNetwork.activate([1, 1]))
		print("--")
