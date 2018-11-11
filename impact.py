import os
import sys
import torch
import glob
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from pruning import getModelLayers

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

def impact(img, model):
	
	# Preprocess parameters
	normalize = transforms.Normalize(
	   mean=[0.485, 0.456, 0.406],
	   std=[0.229, 0.224, 0.225]
	)
	preprocess = transforms.Compose([
	   transforms.Scale(256),
	   transforms.CenterCrop(224),
	   transforms.ToTensor(),
	   normalize
	])	 

	# Load in the labels
	labels = {0: 'uncertain', 1: 'person', -1: 'no_person'}

	# Forward pass with a single image
	model = model.cpu()
	images = glob.glob(img)
	for image in images:
		img = Image.open(image)
		# img.show()
		img_tensor = preprocess(img)
		img_tensor.unsqueeze_(0)
		img_variable = Variable(img_tensor)
		fc_out = model(img_variable)
		break
	print('\n'+labels[fc_out.data.numpy().argmax()]+'\n')

def main():
	model = torch.load(sys.argv[1])
	print(model)
	layers = getModelLayers(model)
	img = "/home/data/pascal_voc/VOCdevkit/VOC2012/transfer/train/person/*jpg"
	impact(img, model)
	

if __name__ == '__main__':
    main()
