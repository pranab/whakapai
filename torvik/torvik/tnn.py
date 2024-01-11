#!/usr/local/bin/python3

# Author: Pranab Ghosh
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License. You may
# obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

# Package imports
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import sklearn as sk
from sklearn.neighbors import KDTree
import matplotlib
import random
import jprops
from random import randint
import statistics
from matumizi.util import *
from matumizi.mlutil import *

"""
forward hook function
"""
intermedOut = {}
lvalues = list()

def hookFn(m, i, o):
	"""
	call back for latent values
	"""
	#intermedOut[m] = o
	lv = o.data.cpu().numpy()
	lv = lv[0].tolist()
	lvalues.append(lv)
	#print(lv)

def getLatValues():
	"""
	"""
	return lvalues
	
class FeedForwardNetwork(torch.nn.Module):
	def __init__(self, configFile, addDefValues=None):
		"""
    	In the constructor we instantiate two nn.Linear modules and assign them as
    	member variables.
    	
		Parameters
			configFile : config file path
			addDefValues : dictionary of additional default values	
		"""
		defValues = dict() if addDefValues is None else addDefValues.copy()
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.scaling.method"] = ("zscale", None)
		defValues["common.scaling.minrows"] = (50, None)
		defValues["common.scaling.param.file"] = (None, None)
		defValues["common.verbose"] = (False, None)
		defValues["common.device"] = ("cpu", None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.data.out.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.layer.data"] = (None, "missing layer data")
		defValues["train.input.size"] = (None, None)
		defValues["train.output.size"] = (None, "missing  output size")
		defValues["train.output.clabels"] = (None, None)
		defValues["train.batch.size"] = (10, None)
		defValues["train.loss.reduction"] = ("mean", None)
		defValues["train.loss.margin"] = (1.0, None)
		defValues["train.num.iterations"] = (500, None)
		defValues["train.lossFn"] = ("mse", None) 
		defValues["train.optimizer"] = ("sgd", None) 
		defValues["train.opt.learning.rate"] = (.0001, None)
		defValues["train.opt.weight.decay"] = (0, None) 
		defValues["train.opt.momentum"] = (0, None) 
		defValues["train.opt.eps"] = (1e-08, None) 
		defValues["train.opt.dampening"] = (0, None) 
		defValues["train.opt.momentum.nesterov"] = (False, None) 
		defValues["train.opt.betas"] = ([0.9, 0.999], None) 
		defValues["train.opt.alpha"] = (0.99, None) 
		defValues["train.save.model"] = (False, None) 
		defValues["train.track.error"] = (False, None) 
		defValues["train.epoch.intv"] = (5, None) 
		defValues["train.batch.intv"] = (5, None) 
		defValues["train.print.weights"] = (False, None) 
		defValues["valid.data.file"] = (None, None)
		defValues["valid.accuracy.metric"] = (None, None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.use.saved.model"] = (True, None)
		defValues["predict.output"] = ("binary", None)
		defValues["predict.feat.pad.size"] = (60, None)
		defValues["predict.print.output"] = (True, None)
		defValues["calibrate.num.bins"] = (10, None)
		defValues["calibrate.pred.prob.thresh"] = (0.5, None)
		defValues["calibrate.num.nearest.neighbors"] = (10, None)
		self.config = Configuration(configFile, defValues)
		
		super(FeedForwardNetwork, self).__init__()
    	
	def setConfigParam(self, name, value):
		"""
		set config param
		
		Parameters
			name : config name
			value : config value
		"""
		self.config.setParam(name, value)

	def getConfig(self):
		"""
		get config object
		"""
		return self.config

	def setVerbose(self, verbose):
		self.verbose = verbose
		
	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		torch.manual_seed(9999)

		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		numinp = self.config.getIntConfig("train.input.size")[0]
		if numinp is None:
			numinp = len(self.config.getIntListConfig("train.data.feature.fields")[0])
		self.outputSize = self.config.getIntConfig("train.output.size")[0]
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
		optimizer = self.config.getStringConfig("train.optimizer")[0]
		self.lossFnStr = self.config.getStringConfig("train.lossFn")[0]
		self.accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]
		self.trackErr = self.config.getBooleanConfig("train.track.error")[0]
		self.batchIntv = self.config.getIntConfig("train.batch.intv")[0]
		self.restored = False
		clabels = self.config.getStringConfig("train.output.clabels")[0]
		self.clabels = self.config.getStringListConfig("train.output.clabels")[0] if clabels is not None else None
		
		#build network
		layers = list()
		ninp = numinp
		trData =  self.config.getStringConfig("train.layer.data")[0].split(",")
		for ld in trData:
			lde = ld.split(":")
			assert len(lde) == 5, "expecting 5 items for layer data"
			
			#num of units, activation, whether batch normalize, whether batch normalize after activation, dropout fraction
			nunit = int(lde[0])
			actStr = lde[1]
			act = FeedForwardNetwork.createActivation(actStr) if actStr != "none"  else None
			bnorm = lde[2] == "true"
			afterAct = lde[3] == "true"
			dpr = float(lde[4])
			
			layers.append(torch.nn.Linear(ninp, nunit))			
			if bnorm:
				#with batch norm
				if afterAct:
					safeAppend(layers, act)
					layers.append(torch.nn.BatchNorm1d(nunit))
				else:
					layers.append(torch.nn.BatchNorm1d(nunit))
					safeAppend(layers, act)
			else:
				#without batch norm
				safeAppend(layers, act)
			
			if dpr > 0:
				layers.append(torch.nn.Dropout(dpr))
			ninp = nunit
			
		self.layers = torch.nn.Sequential(*layers)	
		
		self.device = FeedForwardNetwork.getDevice(self)
		
		#training data
		dataFile = self.config.getStringConfig("train.data.file")[0]
		(featData, outData) = FeedForwardNetwork.prepData(self, dataFile)
		self.featData = torch.from_numpy(featData)
		self.outData = torch.from_numpy(outData)

		#validation data
		dataFile = self.config.getStringConfig("valid.data.file")[0]
		(featDataV, outDataV) = FeedForwardNetwork.prepData(self, dataFile)
		self.validFeatData = torch.from_numpy(featDataV)
		self.validOutData = torch.from_numpy(outDataV)

		# loss function and optimizer
		self.lossFn = FeedForwardNetwork.createLossFunction(self, self.lossFnStr)
		self.optimizer =  FeedForwardNetwork.createOptimizer(self, optimizer)
		
		self.yPred  = None
		self.restored = False
		
		#mode to device
		self.device = FeedForwardNetwork.getDevice(self)	
		self.featData = self.featData.to(self.device)
		self.outData = self.outData.to(self.device)
		self.validFeatData = self.validFeatData.to(self.device)
		self.to(self.device)

	@staticmethod
	def createMultLayPercepNetwork(lrData, numinp):
		"""
		creates MLP network based on layer data
		
		Parameters
			lrData : layer list data
			numinp : num of input element
		"""
		layers = list()
		ninp = numinp
		for ld in lrData:
			lde = ld.split(":")
			assert len(lde) == 5, "expecting 5 items for layer data"
			
			#num of units, activation, whether batch normalize, whether batch normalize after activation, dropout fraction
			nunit = int(lde[0])
			actStr = lde[1]
			act = FeedForwardNetwork.createActivation(actStr) if actStr != "none"  else None
			bnorm = lde[2] == "true"
			afterAct = lde[3] == "true"
			dpr = float(lde[4])
			
			layers.append(torch.nn.Linear(ninp, nunit))			
			if bnorm:
				#with batch norm
				if afterAct:
					safeAppend(layers, act)
					layers.append(torch.nn.BatchNorm1d(nunit))
				else:
					layers.append(torch.nn.BatchNorm1d(nunit))
					safeAppend(layers, act)
			else:
				#without batch norm
				safeAppend(layers, act)
			
			if dpr > 0:
				layers.append(torch.nn.Dropout(dpr))
			ninp = nunit
		
		return torch.nn.Sequential(*layers)
		
	@staticmethod
	def getDevice(model):
		"""
		gets device
		
		Parameters
			model : torch model
		"""
		devType = model.config.getStringConfig("common.device")[0]
		if devType == "cuda":
			if torch.cuda.is_available():
				device = torch.device("cuda")
			else:
				exitWithMsg("cuda not available")
		else:
			device = torch.device("cpu")
		return device
			
	def setValidationData(self, dataSource, prep=True):
		"""
		sets validation data
		
		Parameters
			dataSource : data source str if file path or 2D array
			prep : if True load and prepare 
		"""
		if prep:
			(featDataV, outDataV) = FeedForwardNetwork.prepData(self, dataSource)
			self.validFeatData = torch.from_numpy(featDataV)
			self.validOutData = outDataV
		else:
			self.validFeatData = torch.from_numpy(dataSource[0])
			self.validOutData = dataSource[1]		
		
		self.validFeatData = self.validFeatData.to(self.device)
 	
	@staticmethod
	def createActivation(actName):
		"""
		create activation
		
		Parameters
			actName : activation name
		"""
		if actName is None:
			activation = None
		elif actName == "relu":
			activation = torch.nn.ReLU()
		elif actName == "tanh":
			activation = torch.nn.Tanh()
		elif actName == "sigmoid":
			activation = torch.nn.Sigmoid()
		elif actName == "softmax":
			activation = torch.nn.Softmax(dim=1)
		else:
			exitWithMsg("invalid activation function name " + actName)
		return activation

	@staticmethod
	def createLossFunction(model, lossFnName):
		"""
		create loss function
		
		Parameters
			lossFnName : loss function name
		"""
		config = model.config
		lossRed = config.getStringConfig("train.loss.reduction")[0]
		if lossFnName == "ltwo" or lossFnName == "mse":
			lossFunc = torch.nn.MSELoss(reduction=lossRed)
		elif lossFnName == "ce":
			lossFunc = torch.nn.CrossEntropyLoss(reduction=lossRed)
		elif lossFnName == "lone" or lossFnName == "mae":
			lossFunc = torch.nn.L1Loss(reduction=lossRed)
		elif lossFnName == "bce":
			lossFunc = torch.nn.BCELoss(reduction=lossRed)
		elif lossFnName == "bcel":
			lossFunc = torch.nn.BCEWithLogitsLoss(reduction=lossRed)
		elif lossFnName == "sm":
			lossFunc = torch.nn.SoftMarginLoss(reduction=lossRed)
		elif lossFnName == "mlsm":
			lossFunc = torch.nn.MultiLabelSoftMarginLoss(reduction=lossRed)
		elif lossFnName == "triplet":
			marg = config.getFloatConfig("train.loss.margin")[0]
			lossFunc = torch.nn.TripletMarginLoss(margin=marg, reduction=lossRed)
		elif lossFnName == "nll":
			lossFunc = torch.nn.NLLLoss(reduction=lossRed)
		else:
			exitWithMsg("invalid loss function name " + lossFnName)
		return lossFunc

	@staticmethod
	def createOptimizer(model, optName):
		"""
		create optimizer
		
		Parameters
			optName : optimizer name
		"""
		config = model.config
		learnRate = config.getFloatConfig("train.opt.learning.rate")[0]
		weightDecay = config.getFloatConfig("train.opt.weight.decay")[0]
		momentum = config.getFloatConfig("train.opt.momentum")[0]
		eps = config.getFloatConfig("train.opt.eps")[0]
		if optName == "sgd":
			dampening = config.getFloatConfig("train.opt.dampening")[0]
			momentumNesterov = config.getBooleanConfig("train.opt.momentum.nesterov")[0]
			optimizer = torch.optim.SGD(model.parameters(),lr=learnRate, momentum=momentum, 
			dampening=dampening, weight_decay=weightDecay, nesterov=momentumNesterov)
		elif optName == "adam":
		   	betas = config.getFloatListConfig("train.opt.betas")[0]
		   	betas = (betas[0], betas[1]) 
		   	optimizer = torch.optim.Adam(model.parameters(), lr=learnRate,betas=betas, eps = eps,
    		weight_decay=weightDecay)
		elif optName == "rmsprop":
			alpha = config.getFloatConfig("train.opt.alpha")[0]
			optimizer = torch.optim.RMSprop(model.parameters(), lr=learnRate, alpha=alpha,
			eps=eps, weight_decay=weightDecay, momentum=momentum)
		else:
			exitWithMsg("invalid optimizer name " + optName)
		return optimizer


	def forward(self, x):
		"""
    	In the forward function we accept a Tensor of input data and we must return
    	a Tensor of output data. We can use Modules defined in the constructor as
    	well as arbitrary (differentiable) operations on Tensors.
		
		Parameters
			x : data batch
		"""
		y = self.layers(x)	
		return y

	@staticmethod
	def addForwardHook(model, l, cl = 0):
		"""
		register forward hooks
		
		Parameters
			l : 
			cl :
		"""
		for name, layer in model._modules.items():
			#If it is a sequential, don't register a hook on it
			# but recursively register hook on all it's module children
			print(str(cl) + " : " + name)
			if isinstance(layer, torch.nn.Sequential):
				FeedForwardNetwork.addForwardHook(layer, l, cl)
			else:
			#	 it's a non sequential. Register a hook
				if cl == l:
					print("setting hook at layer " + str(l))
					layer.register_forward_hook(hookFn)
				cl += 1
		
	@staticmethod
	def prepData(model, dataSource, includeOutFld=True):
		"""
		loads and prepares  data
		
		Parameters
			dataSource : data source str if file path or 2D array
			includeOutFld : True if target freld to be included
		"""
		# parameters
		fieldIndices = model.config.getIntListConfig("train.data.fields")[0]
		featFieldIndices = model.config.getIntListConfig("train.data.feature.fields")[0]

		#all data and feature data
		isDataFile = isinstance(dataSource, str)
		selFieldIndices = fieldIndices if includeOutFld else fieldIndices[:-1]
		if isDataFile: 
			#source file path 
			(data, featData) = loadDataFile(dataSource, ",", selFieldIndices, featFieldIndices)
		else:
			# tabular data
			data = tableSelFieldsFilter(dataSource, selFieldIndices)
			featData = tableSelFieldsFilter(data, featFieldIndices)
			#print(featData)
			featData = np.array(featData)
			
		if (model.config.getStringConfig("common.preprocessing")[0] == "scale"):
		    scalingMethod = model.config.getStringConfig("common.scaling.method")[0]
		    
		    #scale only if there are enough rows
		    nrow = featData.shape[0]
		    minrows = model.config.getIntConfig("common.scaling.minrows")[0]
		    if nrow > minrows:
		    	#in place scaling
		    	featData = scaleData(featData, scalingMethod)
		    else:
		    	#use pre computes scaling parameters
		    	spFile = model.config.getStringConfig("common.scaling.param.file")[0]
		    	if spFile is None:
		    		exitWithMsg("for small data sets pre computed scaling parameters need to provided")
		    	scParams = restoreObject(spFile)
		    	featData = scaleDataWithParams(featData, scalingMethod, scParams)
		    	featData = np.array(featData)
		    	
		# target data
		if includeOutFld:
			outFieldIndices = model.config.getIntListConfig("train.data.out.fields")[0]
			if isDataFile:
				outData = data[:,outFieldIndices]
			else:
				outData = tableSelFieldsFilter(data, outFieldIndices)
				outData = np.array(outData)
			foData = (featData.astype(np.float32), outData.astype(np.float32))
		else:
			foData = featData.astype(np.float32)
		return foData


	@staticmethod
	def prepDataNoLabel(model, dataSource):
		"""
		loads and prepares data without label
		
		Parameters
			dataSource : data source str if file path or 2D array
		"""
		# parameters
		fieldIndices = model.config.getIntListConfig("train.data.fields")[0]
		featFieldIndices = model.config.getIntListConfig("train.data.feature.fields")[0]

		#all data and feature data
		isDataFile = isinstance(dataSource, str)
		selFieldIndices = fieldIndices
		if isDataFile: 
			#source file path 
			(data, featData) = loadDataFile(dataSource, ",", selFieldIndices, featFieldIndices)
		else:
			# tabular data
			data = tableSelFieldsFilter(dataSource, selFieldIndices)
			featData = tableSelFieldsFilter(data, featFieldIndices)
			#print(featData)
			featData = np.array(featData)
			
		if (model.config.getStringConfig("common.preprocessing")[0] == "scale"):
		    scalingMethod = model.config.getStringConfig("common.scaling.method")[0]
		    
		    #scale only if there are enough rows
		    nrow = featData.shape[0]
		    minrows = model.config.getIntConfig("common.scaling.minrows")[0]
		    if nrow > minrows:
		    	#in place scaling
		    	featData = scaleData(featData, scalingMethod)
		    else:
		    	#use pre computes scaling parameters
		    	spFile = model.config.getStringConfig("common.scaling.param.file")[0]
		    	if spFile is None:
		    		exitWithMsg("for small data sets pre computed scaling parameters need to provided")
		    	scParams = restoreObject(spFile)
		    	featData = scaleDataWithParams(featData, scalingMethod, scParams)
		    	featData = np.array(featData)
		    	
		# target data
		foData = featData.astype(np.float32)
		return foData

	@staticmethod
	def saveCheckpt(model):
		"""
		checkpoints model
		
		Parameters
			model : torch model
		"""
		print("..saving model checkpoint")
		modelDirectory = model.config.getStringConfig("common.model.directory")[0]
		assert os.path.exists(modelDirectory), "model save directory does not exist"
		modelFile = model.config.getStringConfig("common.model.file")[0]
		filepath = os.path.join(modelDirectory, modelFile)
		state = {"state_dict": model.state_dict(), "optim_dict": model.optimizer.state_dict()}
		torch.save(state, filepath)
		if model.verbose:
			print("model saved")

	@staticmethod
	def restoreCheckpt(model, loadOpt=False):
		"""
		restored checkpointed model
		
		Parameters
			model : torch model
			loadOpt : True if optimizer to be loaded
		"""
		if not model.restored:
			print("..restoring model checkpoint")
			modelDirectory = model.config.getStringConfig("common.model.directory")[0]
			modelFile = model.config.getStringConfig("common.model.file")[0]
			filepath = os.path.join(modelDirectory, modelFile)
			assert os.path.exists(filepath), "model save file does not exist"
			checkpoint = torch.load(filepath)
			model.load_state_dict(checkpoint["state_dict"])
			model.to(model.device)
			if loadOpt:
				model.optimizer.load_state_dict(checkpoint["optim_dict"])
			model.restored = True

	@staticmethod
	def processClassifOutput(yPred, config):
		"""
		extracts probability label 1 or label with highest probability
		
		Parameters
			yPred : predicted output
			config : config object
		"""
		outType = config.getStringConfig("predict.output")[0]
		if outType == "prob":
			outputSize = config.getIntConfig("train.output.size")[0]
			if outputSize == 2:
				#return prob of pos class for binary classifier 
				yPred = yPred[:, 1]
			else:
				#return  class index (ot value) and probability for multi classifier 
				yCl = np.argmax(yPred, axis=1)
				yPred = list(map(lambda y : y[0][y[1]], zip(yPred, yCl)))
				yPred = zip(yCl, yPred)
				
				if self.clabels is not None:
					yPred = list(map(lambda y : (self.clabels[y[0]], y[1]), yPred))
		
		elif outType == "discrete":
			#return class index or value
			yPred = np.argmax(yPred, axis=1)
			if self.clabels is not None:
				yPred =  list(map(lambda y : self.clabels[y], yPred))
		
		else:
			#raw for multi output regression
			pass
			
		return yPred
		
	@staticmethod
	def printPrediction(yPred, config, dataSource):
		"""
		prints input feature data and prediction
		
		Parameters
			yPred : predicted output
			config : config object
			dataSource : data source str if file path or 2D array
		"""
		#prDataFilePath = config.getStringConfig("predict.data.file")[0]
		padWidth = config.getIntConfig("predict.feat.pad.size")[0]
		i = 0
		if type(dataSource) == str:
			for rec in fileRecGen(dataSource, ","):
				feat = (",".join(rec)).ljust(padWidth, " ")
				rec = feat + "\t" + str(yPred[i])
				print(rec)
				i += 1
		else:
			for rec in dataSource:
				srec = toStrList(rec, 6)
				feat = (",".join(srec)).ljust(padWidth, " ")
				srec = feat + "\t" + str(yPred[i])
				print(srec)
				i += 1
			
			
	@staticmethod
	def allTrain(model):
		"""
		train with all data
		
		Parameters
			model : torch model
		"""
		# train mode
		model.train()
		for t in range(model.numIter):

	
			# Forward pass: Compute predicted y by passing x to the model
			yPred = model(model.featData)

			# Compute and print loss
			loss = model.lossFn(yPred, model.outData)
			if model.verbose and  t % 50 == 0:
				print("epoch {}  loss {:.6f}".format(t, loss.item()))

			# Zero gradients, perform a backward pass, and update the weights.
			model.optimizer.zero_grad()
			loss.backward()
			model.optimizer.step()    	

		#validate
		model.eval()
		yPred = model(model.validFeatData)
		yPred = yPred.data.cpu().numpy()
		yActual = model.validOutData
		if model.verbose:
			result = np.concatenate((yPred, yActual), axis = 1)
			print("predicted  actual")
			print(result)
		
		score = perfMetric(model.accMetric, yActual, yPred)
		print(formatFloat(3, score, "perf score"))
		return score

	@staticmethod
	def batchTrain(model):
		"""
		train with batch data
		
		Parameters
			model : torch model
		"""
		model.restored = False
		trainData = TensorDataset(model.featData, model.outData)
		trainDataLoader = DataLoader(dataset=trainData, batch_size=model.batchSize, shuffle=True)
		epochIntv = model.config.getIntConfig("train.epoch.intv")[0]

		# train mode
		model.train()
 
		if model.trackErr:
			trErr = list()
			vaErr = list()
		#epoch
		for t in range(model.numIter):
			#batch
			b = 0
			epochLoss = 0.0
			for xBatch, yBatch in trainDataLoader:
	
				# Forward pass: Compute predicted y by passing x to the model
				xBatch, yBatch = xBatch.to(model.device), yBatch.to(model.device)
				yPred = model(xBatch)
				
				#if model.verbose:
					#print(yPred.shape)
					#print(yBatch.shape)
				
				# Compute and print loss
				loss = model.lossFn(yPred, yBatch)
				if model.verbose and t % epochIntv == 0 and b % model.batchIntv == 0:
					print("epoch {}  batch {}  loss {:.6f}".format(t, b, loss.item()))
				
				if model.trackErr and model.batchIntv == 0:
					epochLoss += loss.item()
				
				#error tracking at batch level
				if model.trackErr and model.batchIntv > 0 and b % model.batchIntv == 0:
					trErr.append(loss.item())
					vloss = FeedForwardNetwork.evaluateModel(model)
					vaErr.append(vloss)

				# Zero gradients, perform a backward pass, and update the weights.
				model.optimizer.zero_grad()
				loss.backward()
				model.optimizer.step()    	
				b += 1
			
			#error tracking at epoch level
			if model.trackErr and model.batchIntv == 0:
				epochLoss /= len(trainDataLoader)
				trErr.append(epochLoss)
				vloss = FeedForwardNetwork.evaluateModel(model)
				vaErr.append(vloss)
			
		#validate
		model.eval()
		yPred = model(model.validFeatData)
		yPred = yPred.data.cpu().numpy()
		yActual = model.validOutData
		if model.verbose:
			vsize = yPred.shape[0]
			print("\npredicted \t\t actual")
			for i in range(vsize):
				print(str(yPred[i]) + "\t" + str(yActual[i]))
		
		score = perfMetric(model.accMetric, yActual, yPred)
		print(yActual)
		print(yPred)
		print(formatFloat(3, score, "perf score"))
		model.yActual = yActual.data.cpu().numpy()
		model.yPred = yPred

		#save
		modelSave = model.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			FeedForwardNetwork.saveCheckpt(model)

		if model.trackErr:
			FeedForwardNetwork.errorPlot(model, trErr, vaErr)
		
		if model.config.getBooleanConfig("train.print.weights")[0]:
			print("model weights")
			for param in model.parameters():
				print(param.data)
		return score

	def fit(self):
		"""
		train with batch data
		
		"""
		return FeedForwardNetwork.batchTrain(self)

	@staticmethod
	def errorPlot(model, trErr, vaErr=None):
		"""
		plot errors
		
		Parameters
			trErr : training error list	
			vaErr : validation error list	
		"""
		x = np.arange(len(trErr))
		plt.plot(x,trErr,label = "training error")
		if vaErr is not None:
			plt.plot(x,vaErr,label = "validation error")
		plt.xlabel("iteration")
		plt.ylabel("error")
		if vaErr is not None:
			plt.legend(["training error", "validation error"], loc='upper left')
		plt.show()

	def getModelValidationData(self):
		"""
		get actual and prediction for validation data
		
		"""
		r = (self.yActual, self.yPred)
		return r

	@staticmethod
	def modelPredict(model, dataSource = None):
		"""
		predict
		
		Parameters
			model : torch model
			dataSource : data source
		"""
		#train or restore model
		useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			FeedForwardNetwork.restoreCheckpt(model)
		else:
			FeedForwardNetwork.batchTrain(model) 

		#predict
		if dataSource is None:
			dataSource = model.config.getStringConfig("predict.data.file")[0]
		featData  = FeedForwardNetwork.prepData(model, dataSource, False)
		#print(featData)
		featData = torch.from_numpy(featData)
		featData = featData.to(model.device)
		
		model.eval()
		yPred = model(featData)
		yPred = yPred.data.cpu().numpy()
		#print(yPred)
		
		if model.outputSize > 1:
			#classification
			yPred = FeedForwardNetwork.processClassifOutput(yPred, model.config)
			
		# print prediction
		if model.config.getBooleanConfig("predict.print.output")[0]:
			FeedForwardNetwork.printPrediction(yPred, model.config, dataSource)
		
		return yPred
	
	def predict(self, dataSource = None):
		"""
		predict
		
		Parameters
			dataSource : data source
		"""
		return FeedForwardNetwork.modelPredict(self, dataSource)
		
	@staticmethod
	def evaluateModel(model):
		"""
		evaluate model
		
		Parameters
			model : torch model
		"""
		model.eval()
		with torch.no_grad():
			yPred = model(model.validFeatData)
			#yPred = yPred.data.cpu().numpy()
			yActual = model.validOutData

			score = model.lossFn(yPred, yActual).item()
		model.train()
		return score
    	
	@staticmethod
	def prepValidate(model, dataSource=None):
		"""
		prepare for validation
		
		Parameters
			model : torch model
			dataSource : data source
		"""
		#train or restore model
		if not model.restored:
			useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
			if useSavedModel:
				FeedForwardNetwork.restoreCheckpt(model)
			else:
				FeedForwardNetwork.batchTrain(model)
			model.restored = True
			
		if 	dataSource is not None:
			model.setValidationData(dataSource)
 
	@staticmethod
	def validateModel(model, retPred=False):
		"""
		model validation
		
		Parameters
			model : torch model
			retPred : if True return prediction
		"""
		model.eval()
		yPred = model(model.validFeatData)
		#print("ypred ", yPred.shape)
		yPred = yPred.data.cpu().numpy()
		yActual = model.validOutData
		#print("yActual ", yActual.shape)
		model.yActual = yActual.data.cpu().numpy()
		model.yPred = yPred
		
		vsize = yPred.shape[0]
		if model.verbose:
			print("\npredicted \t actual")
			if model.outputSize == 1:
				for i in range(vsize):
					print("{:.3f}\t\t{:.3f}".format(yPred[i][0], yActual[i][0]))
			else:
				for i in range(vsize):
					print("{}\t\t{}".format(str(yPred[i]), str(yActual[i])))
				
			
		score = perfMetric(model.accMetric, yActual, yPred)
		print(formatFloat(3, score, "perf score"))
		
		if retPred:
			if model.outputSize == 1:
				y = list(map(lambda i : (yPred[i][0], yActual[i][0]), range(vsize)))
			else:
				y = list(map(lambda i : (yPred[i], yActual[i]), range(vsize)))
			res = (y, score)
			return res
		else:	
			return score
 		
	def validate(self, dataSource=None):
		"""
		model validation
		
		Parameters
			dataSource : data source
		"""
		FeedForwardNetwork.prepValidate(self, dataSource)
		return FeedForwardNetwork.validateModel(self, True)
	
	
		
   	