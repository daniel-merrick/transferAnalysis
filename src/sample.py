from getModelLayers import getModelLayers as GML

def sampleNeurons(numLayers, model, percentageNeuronsSampled):
	totalNeuronsToSample = 0
	totalNeurons = 0;
	modelLayers = GML(model)
	sampledNeurons = []
	for x in range(length(modelLayers)):
		currLayerNeurons = []
		if 'fcl' in modelLayers[x][0]:
			allNeurons = modelLayers[x][1]
			numNeurons = length(allNeurons)
			numToSample = numNeurons * percentageNeuronsSampled
			currSampled = random.sample(list(range(0, numNeurons)), numToSample)
			sampledNeurons.append([x, 1, currSampled])
		
		# Not sure if convolutional layers utilize the exact same neuron format
		# Need to look into it more	
		else if 'conv' in modelLayers[x][0]:
			allNeurons = modelLayers[x][1]
			numNeurons = length(allNeurons)
			numToSample = numNeurons * percentageNeuronsSampled
			currSampled = random.sample(list(range(0, numNeurons)), numToSample)
			sampledNeurons.append([x, 1, currSampled])
		
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