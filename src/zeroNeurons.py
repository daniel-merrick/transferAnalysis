#Functions for zeroing weights


##################################################################################################
# given a weight matrix - either a convolutional kernal or fc neuron return the matrix zeroed

def zeroWeight(weight_tensor):
    return(weight_tensor - weight_tensor)



###################################################################################################
# given a list_of_weight_index return the modelLayers list with zeroed neurons
# list_of_weight_indexes should be of the following format:
#
#   list_of_weight_indexes = [
#                               ['layer index in modelLayers', 1 for weights (2 for biases), [list of neuron indexes]]
#                             ]
#
#   For Example:
#   list_of_weight_indexes = [
#                               [0, 1, [0, 2, 5]],
#                               [3, 1, [2, 4, 5]]
#                            ]

def zeroListOfWeightIndexes(list_of_weight_indexes, modelLayers):
    for layer_index, modelLayerSubIdx, weight_indexes in list_of_weight_indexes:
        
        #1 for weights
        for input_index in weight_indexes:
            modelLayers[layer_index][modelLayerSubIdx][input_index] = zeroWeight(modelLayers[layer_index][modelLayerSubIdx][input_index])

    return modelLayers


