
#........@srjk & @umg

import numpy as np
from Perceptron import Perceptron

training_inputs = []
training_inputs.append(np.array([1, 1]))
training_inputs.append(np.array([1, 0]))
training_inputs.append(np.array([0, 1]))
training_inputs.append(np.array([0, 0]))

labels = np.array([0, 1, 1, 0])
perceptron = []
for i in range(0, 4):
    perceptron.append(Perceptron(2))

perceptron.append(Perceptron(4))

labels = np.array([0, 0, 0, 1])
perceptron[0].train(training_inputs, labels)

labels = np.array([0, 0, 1, 0])
perceptron[1].train(training_inputs, labels)

labels = np.array([0, 1, 0, 0])
perceptron[2].train(training_inputs, labels)

labels = np.array([1, 0, 0, 0])
perceptron[3].train(training_inputs, labels)

for i in perceptron[:-1]:
    print(i.weights)

outputlayerInput = []
for i in perceptron[:-1]:

    hiddenLayerOutput = []

    for j in range(0, 4):
        hiddenLayerOutput.append(i.predict(training_inputs[j]))

    outputlayerInput.append(hiddenLayerOutput)

print(outputlayerInput)

outputlayerInput1 = []
for i in outputlayerInput:
    a = np.array([])
    for j in i:
        a = np.append(a, j)

    outputlayerInput1.append(a)

print(training_inputs)
print(outputlayerInput1)

labels = np.array([0, 1, 1, 0])
perceptron[4].train(outputlayerInput1, labels)

print(perceptron[4].weights)

def predictXOR(input):
    predictoutput = []

    for i in perceptron[:-1]:
        predictoutput.append(i.predict(inputs))

    print(perceptron[4].predict(predictoutput))


inputs = np.array([1, 1])
predictXOR(inputs)

inputs = np.array([0, 0])
predictXOR(inputs)

inputs = np.array([0, 1])
predictXOR(inputs)

inputs = np.array([1, 0])
predictXOR(inputs)
