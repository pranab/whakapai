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
import matplotlib
import random
import jprops
from random import randint
from matumizi.util import *
from matumizi.mlutil import *
from .tnn import *


class FeedForwardMultiNetwork(FeedForwardNetwork):
	"""
	contrastive learning multi feed forward network
	"""
	def __init__(self, configFile):
		defValues = dict()
		defValues["predict.data.file.raw"] = (None, None)
		defValues["predict.class.labels"] = (None, None)
		super(FeedForwardMultiNetwork, self).__init__(configFile, defValues)

	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		super().buildModel()
		
		#final fully connected after merge
			
		feCount = self.config.getIntConfig("train.input.size")[0]
		self.vaFe1 = self.validFeatData[:,:feCount]
		self.vaFe2 = self.validFeatData[:,feCount:2*feCount]
		self.vaFe3 = self.validFeatData[:,2*feCount:]

	def forward(self, x1, x2, x3):
		"""
    	Go through layers twice
		"""
		y1 = self.layers(x1)	
		y2 = self.layers(x2)
		y3 = self.layers(x3)
		y = (y1, y2, y3)
		return y
		
	@staticmethod
	def batchTrain(model):
		"""
		train with batch data
		"""
		feCount = model.config.getIntConfig("train.input.size")[0]
		fe1 = model.featData[:,:feCount]
		fe2 = model.featData[:,feCount:2*feCount]
		fe3 = model.featData[:,2*feCount:]
		
		#print(fe1.shape)
		#print(fe2.shape)
		#print(fe3.shape)
		trainData = TensorDataset(fe1, fe2, fe3)
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
			for x1Batch, x2Batch, x3Batch in trainDataLoader:
	
				# Forward pass: Compute predicted y by passing x to the model
				yPred = model(x1Batch, x2Batch, x3Batch)
				
				# Compute and print loss
				loss = model.lossFn(yPred[0], yPred[1], yPred[2])
				if model.verbose and t % epochIntv == 0 and model.batchIntv > 0 and b % model.batchIntv == 0:
					print("epoch {}  batch {}  loss {:.6f}".format(t, b, loss.item()))
				
				if model.trackErr and model.batchIntv == 0:
					epochLoss += loss.item()
				
				#error tracking at batch level
				if model.trackErr and model.batchIntv > 0 and b % model.batchIntv == 0:
					trErr.append(loss.item())
					vloss = FeedForwardTwinNetwork.evaluateModel(model)
					vaErr.append(vloss)

				# Zero gradients, perform a backward pass, and update the weights.
				model.optimizer.zero_grad()
				loss.backward()
				model.optimizer.step()    	
				b += 1
			
			#error tracking at epoch level
			if model.trackErr and model.batchIntv == 0:
				epochLoss /= b
				if model.verbose:
					print("epoch {}  loss {:.6f}".format(t, epochLoss))
				trErr.append(epochLoss)
				vloss = FeedForwardMultiNetwork.evaluateModel(model)
				vaErr.append(vloss)
			
		
		#save
		modelSave = model.config.getBooleanConfig("train.model.save")[0]
		if modelSave:
			FeedForwardNetwork.saveCheckpt(model)

		if model.trackErr:
			FeedForwardNetwork.errorPlot(model, trErr, vaErr)
			
		return 1.0
		
		
	@staticmethod
	def evaluateModel(model):
		"""
		evaluate model
		
		Parameters
			model : torch model
		"""
		model.eval()
		with torch.no_grad():
			yPred = model(model.vaFe1, model.vaFe2, model.vaFe3)
			score = model.lossFn(yPred[0], yPred[1], yPred[2]).item()
		model.train()
		return score

	@staticmethod
	def testModel(model):
		"""
		test model
		
		Parameters
			model : torch model
		"""
		useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			FeedForwardNetwork.restoreCheckpt(model)
		else:
			FeedForwardTwinNetwork.batchTrain(model) 
		
		dataSource = model.config.getStringConfig("valid.data.file")[0]	
		featData = FeedForwardNetwork.prepData(model, dataSource, False)
		featData = torch.from_numpy(featData)
		feCount = model.config.getIntConfig("train.input.size")[0]
		fe1 = featData[:,:feCount]
		fe2 = featData[:,feCount:2*feCount]
		fe3 = featData[:,2*feCount:]
		
		
		model.eval()
		with torch.no_grad():
			yp = model(fe1, fe2, fe3)
			for y in yp:
				model.correctAllZeros(y)
			
			cos = torch.nn.CosineSimilarity()
			s1 = cos(yp[0], yp[1]).data.cpu().numpy()
			s2 = cos(yp[0], yp[2]).data.cpu().numpy()
			#print(s1.shape)
			
			"""
			print(yp[0][:9,:])
			print("\n\n")
			print(yp[1][:9,:])
			print("\n\n")
			print(s1[:9])
			print("\n\n")
			"""
			
			n = yp[0].shape[0]
			if model.verbose:
				print("num rows ", n)
				for i in range(15):
					if i % 3 == 0:
						print("next")
					print(yp[0][i])
					print(yp[1][i])
					print(yp[2][i])
					print("similarity  {:.3f}  {:.3f}".format(s1[i], s2[i]))
				
			tc = 0
			cc = 0
			outputSize = model.config.getIntConfig("train.output.size")[0]
			for i in range(0, n, outputSize):
				#for each sample outputSize no of rows
				msi = None
				imsi = None
				for j in range(outputSize): 
					#first one positive , followed by all negative
					si = (s1[i+j] + s2[i+j]) / 2
					if msi == None or si > msi:
						msi = si
						imsi = j
				tc += 1
				#first one positive i.e same class as focus record
				if imsi == 0:
					cc += 1
				
		score = cc / tc
		print("score: {:.3f}".format(score))	
		model.train()
		return score


	@staticmethod
	def predictModel(model):
		"""
		predict and return class labels
		
		Parameters
			model : torch model
		"""
		useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
		if useSavedModel:
			FeedForwardNetwork.restoreCheckpt(model)
		else:
			FeedForwardTwinNetwork.batchTrain(model) 
		
		dataSource = model.config.getStringConfig("predict.data.file")[0]	
		featData = FeedForwardNetwork.prepData(model, dataSource, False)
		featData = torch.from_numpy(featData)
		feCount = model.config.getIntConfig("train.input.size")[0]
		fe1 = featData[:,:feCount]
		fe2 = featData[:,feCount:2*feCount]
		fe3 = featData[:,2*feCount:]
		
		
		model.eval()
		with torch.no_grad():
			yp = model(fe1, fe2, fe3)
			for y in yp:
				model.correctAllZeros(y)
			
			cos = torch.nn.CosineSimilarity()
			s1 = cos(yp[0], yp[1]).data.cpu().numpy()
			s2 = cos(yp[0], yp[2]).data.cpu().numpy()
			#print(s1.shape)
			
			#print(yp[0][:9,:])
			#print("\n\n")
			#print(yp[1][:9,:])
			
			rawPredFilepath = model.config.getStringConfig("predict.data.file.raw")[0]
			assert rawPredFilepath is not None, "missing raw prediction data file path"
			tdataRaw = getFileLines(rawPredFilepath, None)
			
			clLabels = model.config.getStringListConfig("predict.class.labels")[0]
			assert clLabels is not None, "missing class label list"
			
			n = yp[0].shape[0]
			ri = 0
			outputSize = model.config.getIntConfig("train.output.size")[0]
			cp = list()
			padWidth = model.config.getIntConfig("predict.feat.pad.size")[0]
			preds = list()
			for i in range(0, n, outputSize):
				#for each sample outputSize no of rows
				msi = None
				imsi = None
				for j in range(outputSize): 
					#first one positive , followed by all negative
					si = (s1[i+j] + s2[i+j]) / 2
					#print("j {}  si {:.3f}".format(j, si))
					if msi == None or si > msi:
						msi = si
						imsi = j
				#first one positive i.e same class as focus record
				cl = clLabels[imsi]
				cp.append(cl)
				if model.config.getBooleanConfig("predict.print.output")[0]:
					feat = tdataRaw[ri].ljust(padWidth, " ")
					rec = "{}\t{}  ({:.3f})".format(feat, cl, msi)
					print(rec)
					
				pred = list()
				pred.append(tdataRaw[ri])
				pred.append(cl)
				pred.append(msi)
				preds.append(pred)
				ri += 1
				
		model.train()
		return preds
		
	def correctAllZeros(self, yp):
		"""
		correct for all zero
		
		Parameters
			yp : output tensor		
		"""
		rs = yp.shape[1]
		cvals = np.full((rs), .0001)
		cvals = torch.from_numpy(cvals)
		ri = list()
		i = 0
		for r in yp:
			cnt = 0
			for c in r:
				if c < .0001:
					cnt += 1
			if cnt == rs:
				ri.append(i)
			i += 1
		
		for i in ri:
			yp[i] = cvals
			 
				
			
		

		
