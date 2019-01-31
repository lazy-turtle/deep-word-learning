#!/usr/bin/python

# Copyright 2017 Giorgia Fenoglio
#
# This file is part of NNsTaxonomicResponding.
#
# NNsTaxonomicResponding is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NNsTaxonomicResponding is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NNsTaxonomicResponding.  If not, see <http://www.gnu.org/licenses/>.


from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags

import datetime


tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
"""Display this many predictions.""")

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          './retrain10classes', 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          './retrain10classes', 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)


  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''

    return self.node_lookup[node_id]



def calcoloAttivazione(classPath,imgsPath,sess,N,jpeg_data_tensor, bottleneck_tensor):

	""" activations of a level of the net starting from a set of images
		classPath: path to the folder of the images
		imgsPath : list of images
		sess: current session of tensorflow
		n: number of images to extract from the folder classPath
		bottleneck_tensor: level of the net in which we want calculate the activation
	"""
	classe = classPath.split('/')[-2]
	print('activation of class '+classe+' ...')

	# lista di immagini
	if len(imgsPath)==0:
		filesClass = sorted(glob.glob(classPath))
	else:
		filesClass = sorted(imgsPath)

	nfilesClass = len(filesClass)

  # activations = matrix of size (number of images X lenght output intrested leve)

	bottValues = np.zeros(shape=(N,2048))


	i = 0
	for p in filesClass:
		print(str(i)+' -- '+p)
		image_data = gfile.FastGFile(p, 'rb').read()
		bottleneck_values = InceptionPrototipi.run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
		bottValues[i] = bottleneck_values
		i = i+1
		if i>(N-1):
			break


  # print the activations on a file
	printToOneFileCSV(bottValues,filesClass)


	return bottValues

def calcoloSoftMax(classPath,sess,N,jpeg_data_tensor, softmax_tensor):
	""" activations of the last level of the net starting from a set of images
		classPath: path to the folder of the images
		imgsPath : list of images
		sess: current session of tensorflow
		n: number of images to extract from the folder classPath
		bottleneck_tensor: level of the net in which we want calculate the activation
	"""
	classe = classPath.split('/')[-2]
	print('activation of class '+classe+' ...')

	# list of images
	filesClass = sorted(glob.glob(classPath))
	nfilesClass = len(filesClass)
	nClassi = softmax_tensor.get_shape()[1]
  # activations = matrix of size (number of images X lenght output intrested leve)
	bottValues = np.zeros(shape=(N,nClassi))


	i = 0

	scoreElements = np.array([])
	for p in filesClass:
		image_data = gfile.FastGFile(p, 'rb').read()
		predictions = InceptionPrototipi.run_softmax_on_image(sess, image_data)

		#to each image is associated the score of the better class
		classScore = handlePredictions(predictions,classe)
		scoreElements = np.append(scoreElements,classScore)
		bottValues[i] = predictions
		i = i+1
		if i>(N-1):
			break


	#extract the images with the maximum score
	scoreSort = scoreElements.argsort()
	NUMTOPRINT = 10
	last = len(scoreSort)-1
	print('-----more significative images :----')
	for i in range(last, last-NUMTOPRINT, -1):
		print(filesClass[scoreSort[i]]+' --> score('+str(scoreElements[scoreSort[i]])+')')

	#estraggo le NUMTOPRINT immagini con score piu basso
	NUMTOPRINT = 10
	print('-----Less significative images:----')
	for i in range(0,NUMTOPRINT):
		print(filesClass[scoreSort[i]]+' --> score('+str(scoreElements[scoreSort[i]])+')')


def handlePredictions(predictions,className):
	#print the best predictions
	node_lookup = NodeLookup()
	#number of predictions to consider
	NUMPRED = 4

	top_k = predictions.argsort()[-NUMPRED:][::-1]


	scoreToReturn = 0
	for node_id in top_k:
		human_string = node_lookup.id_to_string(node_id)
		score = predictions[node_id]
		if human_string == className:
			scoreToReturn = score

	return scoreToReturn

def extraxtCorrectClassified(classPath,sess,N,jpeg_data_tensor, softmax_tensor):

	"""
		extract N images form the classPath folder that are correctly classified
		sess: current session of tensorflow
		N: number of images to extract
		softmax_tensor : level in which consider the classification
	"""
	# Esclusion of the images of the training set
	fTraining = open('FILE OF THE TRAINING SET','r')
	imgTrain = list()
	for l in fTraining:
		imgTrain.append(l)

	classe = classPath.split('/')[-2]
	print('activation class '+classe+' ...')

	# llist of images
	filesClass = sorted(glob.glob(classPath))
	nfilesClass = len(filesClass)
	nClassi = softmax_tensor.get_shape()[1]

	imgList = list()

	node_lookup = NodeLookup()

	scoreElements = np.array([])
	for p in filesClass:
		print(p)
		if p not in imgTrain:
			print('correctly classified: '+str(len(imgList)))
			image_data = gfile.FastGFile(p, 'rb').read()
			#activation of the last level
			predictions = InceptionPrototipi.run_softmax_on_image(sess, image_data)

			#more activated neuron
			top_prediction = predictions.argsort()[-1:][::-1]
			print(top_prediction)
			#extraction of the class associated to the neuron
			human_string = node_lookup.id_to_string(top_prediction[0])
			print(human_string)
			#check that is the correct class
			if human_string == classe:
				imgList.append(p)
				#
				if len(imgList) >= N:
					break

	return imgList

# def varianza(vectors, prototipo):
#   """
	#extraction of the variance of the vectors from the prototype prototipo
# 	"""
#   varianza = np.zeros(shape=(1,2048))

#   N = vectors.shape[0]


#   for p in vectors:
#     for i in range(0,2048):
#       varianza[0][i] = varianza[0][i] + (p[i]-prototipo[i])**2

#   for i in range(0,2048):
#     varianza[0][i] = varianza[0][i]/N

#   varianza = np.mean(varianza, axis=1)

#   return varianza

def printToOneFileCSV(activations,files):
	"""
		#print of the activations on a single csv file
	"""
	# number of element to print
	N = activations.shape[0]
	today = datetime.date.today()

  # creation of the folder
	if not os.path.exists('./csv'):
		os.makedirs('./csv')

	fOutput = './csv/attivazioni10classiTESTSET'+str(today)+'.csv'

	new = False
	if not os.path.exists(fOutput):
		new = True

	f = open(fOutput,'a')

  # print of the attributes
  	if new:
		f.write('name,')
		for i in range(0,2048):
			f.write('att'+str(i))
			if i!=2048:
				f.write(',')
		f.write('\n')

  # print of the activations
	j = 0
	for i in activations:
		st = ""+files[j]+","
		for v in i:
			st = st+str(v)+','
		st = st[0:-1]
		f.write(st+'\n')
		j = j+1
	f.close()

def printToFileCSV(activations,c,files):
	""" #print of the activations on a csv file
	"""

	N = activations.shape[0]
	today = datetime.date.today()

  # creation of the folder
	if not os.path.exists('./csv'):
		os.makedirs('./csv')

	fOutput = './csv/attivazioni'+c+str(today)+'.csv'

  # print in csv of the attributes
	f = open(fOutput,'w')
	f.write('name,')
	for i in range(0,2048):
		f.write('att'+str(i))
		if i!=2048:
			f.write(',')
	f.write('\n')

  # print of the activations
	j = 0
	for i in activations:
		st = ""+files[j]+","
		for v in i:
			st = st+str(v)+','
		st = st[0:-1]
		f.write(st+'\n')
		j = j+1
	f.close()

def findMaximumNeuron(activations):
	"""
		find the neuron with the maximum activation
	"""
	IOldOldNeuron = -1
	IOldNeuron = -1
	IMaxNeuron = -1
	MaxNeuron = -1
	for v in activations:
		maxAct = np.argmax(v)
		if (v[maxAct] > MaxNeuron) and (maxAct != IMaxNeuron):
			IOldOldNeuron = IOldNeuron
			IOldNeuron = IMaxNeuron
			IMaxNeuron = maxAct
			MaxNeuron = v[maxAct]


	return IMaxNeuron, IOldNeuron, IOldOldNeuron

def handleWeights(softmax_weight):
	"""
		get the weights to the last level and calculate some statistics
	"""
	(r,c) = softmax_weight.get_shape()

	with tf.Session():
		weight = softmax_weight.eval()

		#max and minimum
		maxW = np.amax(weight, axis=0)
		minW = np.amin(weight, axis=0)

		#mean and variance
		meanW = np.mean(weight, axis=0)
		varW = np.var(weight, axis=0)

		print('Mean')
		print(meanW)
		print('Variance')
		print(varW)

		#tresholding
		percSoglia = 10
		sogliaWmax = np.subtract(maxW,((np.subtract(maxW, minW)) * percSoglia / 100))
		sogliaWmin = np.add(minW,((np.subtract(maxW, minW)) * percSoglia / 100))

		#print(sogliaWmin)
		#print(sogliaWmax)

		#set to 0 the weights under the treshold
		for r in weight:
			for i in range(0,c):
				if (r[i] < sogliaWmax[i]) and (r[i] > sogliaWmin[i]):
					r[i] = 0

		#count of the remaining values
		countRemain = [0, 0, 0, 0]
		indexRemain = [[],[],[],[]]
		valueRemain = [[],[],[],[]]
		rid = 0
		for r in weight:
			for i in range(0,c):
				if r[i] != 0:
					countRemain[i] += 1
					indexRemain[i].append(rid)
					valueRemain[i].append(r[i])
			rid = rid+1

		# check of the common weights
		for i in range(0,c):
			for j in range(i+1,c):
				if i!=j:
					print('weights in common classes ' + str(i) + ' and ' + str(j))
					print(len(np.intersect1d(indexRemain[i], indexRemain[j])))

		print('-------------- \n Number of remaining elements')
		print(countRemain)
		print('-------------- \n Indexes of remaining elements')
		print(indexRemain)
		print('-------------- \n Remaining values')
		print(valueRemain)


def getAllInputClass(className,fileInput):
	f = open(fileInput,'r')
	inputC = []
	for l in f:
		lSplit = l.split(',')
		if className in lSplit[0]:
			print (lSplit[0])
			inputC.append(np.array(lSplit[1:]).astype(float))
	f.close()
	inputCl = np.asarray(inputC)
	return inputCl

# def fCalcoloVarianze():
# 	classes = []
# 	fClassesName = './retrain10classes/output_labels.txt'
# 	fClasses = open(fClassesName, 'r')
# 	for l in fClasses:
# 		classes.append(l[:-1])

# 	fInputsName = './csv/attivazioni10classi2016-10-10.csv'

# 	fOutputName = './csv/varianze.csv'
# 	fOutput = open(fOutputName,'a')
# 	fOutput.write('classe,')

# 	exClasses = dict()
# 	for c in classes:
# 		exClasses[c] = getAllInputClass(c,fInputsName)
# 		fOutput.write(c+',')
# 	fOutput.write('\n')

# 	for c in classes:
# 		print('--- da classe '+c)
# 		prototipo = np.mean(exClasses[c], axis=0)
# 		fOutput.write(c+',')
#   		for c2 in classes:
#   			print(' a '+c2)
#   			v = varianza(exClasses[c2], prototipo)
#   			fOutput.write(str(v[0])+',')
#   		fOutput.write('\n')


if __name__ == '__main__':

	# # initialization of the model
   [graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor,sess, softmax_weight, softmax_tensor] = InceptionPrototipi.main('_')

   folderPaths = 'FOLDER WITH THE IMAGES'

   classPaths = []
   classFolders = glob.glob(folderPaths+'*')
   for c in classFolders:
   	classPaths.append((c.split('/')[-1]).split('.')[0])

   print(classPaths)

 #  #definition of the classes
   for cat in classPaths:
   	print('CLASS '+cat)
	firstClassPath = folderPaths + cat +'/*.JPEG'

	#number of images for each class
	N = 100
	print('Extraction images correctly classified '+cat)
	imgsToEval = extraxtCorrectClassified(firstClassPath,sess,N,jpeg_data_tensor, softmax_tensor)
	print(imgsToEval)
	print('Calculation activations in class '+cat)
	bottValues1 = calcoloAttivazione(firstClassPath, imgsToEval,sess,N,jpeg_data_tensor, bottleneck_tensor)
