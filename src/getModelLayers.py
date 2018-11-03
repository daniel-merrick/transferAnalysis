##################################################################################################################################
# def getModelLayers(model): is a function to return the layers of a model in a more clean and organized manner
#                            that is efficient and easier for pruning purposes
#
#
# The format of the returned data will be a LIST of TUPLES or as followed:
#    cleanedLayers[i] represents the i_th layer (that needs to be pruned) counting from the output
#         ---cleanedLayers[0] is the last layer of the model that needs to be pruned:
#            -Therefore if the last layer is a convolutional layer or fully connected layer, it will be stored...
#            -However, if the last layer is a batch norm layer (doesn't need to be pruned) it will not store it.
#            -If the last layer is batch norm and the second-to-last layer is a convolutional layer, cleanedLayers[0] will 
#            store the second-to-last layer and so forth.
#
#    At cleanedLayers[i] will be a tuple of the following data:
#         ---cleanedLayers[i][0]: Name of the i_th to last layer to be pruned (string)
#         ---cleanedLayers[i][1]: weights of the i_th to last layer, of type tensor
#         ---cleanedLayers[i][2]: biases of the i_th to last layer, of type tensor
#
# NOTES:
#         ---memory is shared between the model that is passed into getModelLayers and the returned list,
#            therefore any changed values are permanately changed unless a deep copy is made

def getModelLayers(model):
    
    #load state dictionary - state dictionary of a model holds its parameter
    sdict = model.state_dict()
    
    #these variables are used to help reorganize the layers
    pruningWeights = dict()
    pruningBiases = dict()
    
    #only add layers that are going to be pruned -- no need to add pooling layers or batch norm layers etc
    for (layer_name, params) in sdict.items():
        if (not 'bn' in layer_name and not 'downsample' in layer_name and not 'pool' in layer_name):
            if 'weight' in layer_name:
                pruningWeights[layer_name] = (params)
            elif('bias' in modelLayers[i][0]):
                pruningBiases[layer_name] = (params)
    
    #append layers by (name, weights, biases) in cleanedLayers
    cleanedLayers = []    
    for weightName, weights in pruningWeights.items():
        
        #if layer as biases append weights + biases
        if weightName[:-len('weight')]+'bias' in pruningBiases:
            cleanedLayers.append((weightName[:-len('weight')], weights, pruningBiases[weightName[:-len('weight')]+'bias']))
        
        #if layer doesn't have biases, append weights and an empty tensor for biases
        else:
            cleanedLayers.append((weightName[:-len('weight')], weights, torch.FloatTensor([])))
    
    #reverse the list so cleanedLayers[0] is the last layer of the network -- aka the layer closest to output
    cleanedLayers.reverse()

    return (cleanedLayers)
##############################################################################################################################

