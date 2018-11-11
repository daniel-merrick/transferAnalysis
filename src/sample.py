import numpy.random as random
import math

def sampleNeurons(numLayers, modelLayers, percentageNeuronsSampled):
	sampledNeurons = []
        allNeurons = []
	percentageNeuronsSampled = float(percentageNeuronsSampled)
        for x in range(numLayers):
                if 'fc' in modelLayers[x][0] or 'classifier' in modelLayers[x][0]:
			allNeurons = modelLayers[x][1]
			sampledNeurons.append(sampleFCLNeurons(allNeurons, percentageNeuronsSampled, x))
			
		# Not sure if convolutional layers utilize the exact same neuron format
		# Need to look into it more	
		elif 'conv' in modelLayers[x][0] or 'features' in modelLayers[x][0]:
			allNeurons = modelLayers[x][1]
			sampledNeurons.append(sampleFCLNeurons(allNeurons, percentageNeuronsSampled, x))

	return sampledNeurons
	
	# Return Format
	# [[0, 1, [list of indices to prune]],
	#  [1, 1, [list of indices to prune]],
	#  [2, 1, [list of indices to prune]],
	#  [3, 1, [list of indices to prune]],
	#  [4, 1, [list of indices to prune]],
	#  ..................................
	#  [Number of Model Layers - 1, 1, [list of indices to prune]],
	# ]


def sampleFCLNeurons(layerNeurons, percentageNeuronsSampled, layerNum):
	numNeurons = len(layerNeurons)
	numToSample = int(math.ceil(numNeurons * percentageNeuronsSampled) / 100.0)
	currSampled = random.choice(list(range(0, numNeurons)), numToSample, replace=False)
	return [layerNum, 1, currSampled]

def sampleConvNeurons(layerNeurons, percentageNeuronsSampled, layerNum):
	numNeurons = len(layerNeurons)
        numToSample = int(math.ceil(numNeurons * percentageNeuronsSampled) / 100.0)
        currSampled = random.choice(list(range(0, numNeurons)), numToSample, replace=False)
	return [layerNum, 1, currSampled]
