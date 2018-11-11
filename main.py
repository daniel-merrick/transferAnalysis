import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.autograd import Variable

from PIL import Image
import os
import copy
import pprint

import sys
sys.path.insert(0, 'src/')

from zeroNeurons import zeroListOfWeightIndexes, zeroWeight
from getModelLayers import getModelLayers
from sample import sampleNeurons


#update the original model with its new parameters
def updateModelParameters(model, tempModelLayers):
    #build new partial state_dict
    partial_state_dict = dict()
    for layer_name, weights, biases in tempModelLayers:
        
        partial_state_dict[layer_name+'weight'] = weights
        partial_state_dict[layer_name+'bias'] = biases

    current_sdict = model.state_dict()

    current_sdict.update(partial_state_dict)
    model.load_state_dict(current_sdict)
    return(model)

def removeNegativeTransfer(num_layers_to_prune, model, data, percent_of_neurons_per_layer=10.0):
    
    #get model layers - Dan
    modelLayers = getModelLayers(model)

    #deep copy so we don't forget values when we temporarily zero weights
    tempModelLayers = copy.deepcopy(modelLayers)
   
    #get list of sampled indexes - Karthik
    sampledIdxs = sampleNeurons(num_layers_to_prune, tempModelLayers, percent_of_neurons_per_layer)
   
    #determine impact of each sampled neuron across all layers and return structure similar to sampledIdxs
    #with indexes to zero weights of neurons instead - Andrew
    #noImpactIdxs = determineImpacts(tempModelLayers, modelLayers, model, data)
    
    noImpactIdxs = sampledIdxs #temporary
    
    #zero out neurons in noImpactIdxs - Dan
    tempModelLayers = zeroListOfWeightIndexes(noImpactIdxs, tempModelLayers)
     
    #freeze Gradients - Zohar
    #freezeGradients(noImpactIdxs, model)

    #copy the new layers to the model - Dan
    model = updateModelParameters(model, tempModelLayers)

    #retrain model on data - Unassigned
    #model = trainModel(model, data)

    #evaluate newly trained model - Unassigned
    #results = evalutateModel(model, data)
    

    return(model)#, results)  
    

def main():
    
    #number of layers to prune, starting from end of network
    NUM_LAYERS_TO_PRUNE = 16

    DATA_PATH = ''
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    SHUFFLE = False

    #transforms = transforms.Compose([
    #                                    transforms.Resize(224),             # resize the image to 64x64 
    #                                    transforms.ToTensor()])             # transform it into a PyTorch Tensor
    #test_loader = DataLoader(Dataset(DATA_PATH, transorms), batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)


    VGG16 = models.vgg16()
    prunedVGG16 = removeNegativeTransfer(NUM_LAYERS_TO_PRUNE, VGG16, 'data')

if __name__ == '__main__':
    main()
