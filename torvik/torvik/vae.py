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
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
import random
import jprops
from random import randint
from matumizi.util import *
from matumizi.mlutil import *
from .tnn import FeedForwardNetwork

"""
Variational auto encoder
"""
class VarAutoEncoder(nn.Module):
	def __init__(self, configFile):
		"""
    	constructor 

		Parameters
			configFile : config file path
		"""
		defValues = dict()
		defValues["common.mode"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.scaling.method"] = ("zscale", None)
		defValues["common.verbose"] = (False, None)
		defValues["common.device"] = ("cpu", None)
		defValues["train.data.shape"] = (None, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.fields"] = (None, "missing training data field ordinals")
		defValues["train.data.feature.fields"] = (None, "missing training data feature field ordinals")
		defValues["train.num.input"] = (None, "missing input size")
		defValues["train.num.latent"] = (None, "missing latent size")
		defValues["train.num.enc.output"] = (None, "missing encoder output size")
		defValues["train.network.type"] = ("mlp", None)
		defValues["train.enc.layer.data"] = (None, "missing encoder layer datar")
		defValues["train.dec.layer.data"] = (None, "missing decoder layer datar")
		defValues["train.batch.size"] = (50, None)
		defValues["train.num.iterations"] = (500, None)
		defValues["train.optimizer"] = ("adam", None) 
		defValues["train.opt.learning.rate"] = (.0001, None)
		defValues["train.opt.weight.decay"] = (0, None)
		defValues["train.opt.momentum"] = (0, None) 
		defValues["train.opt.eps"] = (1e-08, None) 
		defValues["train.opt.dampening"] = (0, None) 
		defValues["train.opt.momentum.nesterov"] = (False, None) 
		defValues["train.opt.betas"] = ([0.9, 0.999], None) 
		defValues["train.opt.alpha"] = (0.99, None) 
		defValues["train.model.save"] = (False, None)
		defValues["train.track.error"] = (False, None) 
		defValues["train.batch.intv"] = (5, None) 
		defValues["encode.use.saved.model"] = (True, None)
		defValues["encode.data.file"] = (None, "missing enoding data file")
		defValues["valid.accuracy.metric"] = (None, None)
		defValues["pred.data.file"] = (None, "missing prediction data file")
		self.config = Configuration(configFile, defValues)

		super(VarAutoEncoder, self).__init__()

	def getConfig(self):
		"""
    	get configuration
		"""
		return self.config
		
	def buildModel(self):
		"""
    	Loads configuration and builds the various piecess necessary for the model
		"""
		torch.manual_seed(9999)
		self.verbose = self.config.getStringConfig("common.verbose")[0]
		self.numinp = self.config.getIntConfig("train.num.input")[0]
		self.numeout = self.config.getIntConfig("train.num.enc.output")[0]
		self.numlat = self.config.getIntConfig("train.num.latent")[0]
		nwType = self.config.getStringConfig("train.network.type")[0]
		encLayers = self.config.getStringConfig("train.enc.layer.data")[0].split(",")
		decLayers = self.config.getStringConfig("train.dec.layer.data")[0].split(",")
		if nwType == "mlp":
			self.encoder = FeedForwardNetwork.createMultLayPercepNetwork(encLayers, self.numinp)
			self.decoder = FeedForwardNetwork.createMultLayPercepNetwork(decLayers, self.numlat)
		
		self.fcMean = nn.Linear(self.numeout, self.numlat)
		self.fcVar = nn.Linear(self.numeout, self.numlat)
		self.rpSampler = torch.distributions.Normal(0, 1)
		self.kl = None
		
		self.dshape = self.config.getIntListConfig("train.data.shape")[0]
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.numIter = self.config.getIntConfig("train.num.iterations")[0]
		self.optimizerStr = self.config.getStringConfig("train.optimizer")[0]
		self.modelSave = self.config.getBooleanConfig("train.model.save")[0]
		self.useSavedModel = self.config.getBooleanConfig("encode.use.saved.model")[0]
		self.trackErr = self.config.getBooleanConfig("train.track.error")[0]
		self.batchIntv = self.config.getIntConfig("train.batch.intv")[0]
		self.accMetric = self.config.getStringConfig("valid.accuracy.metric")[0]		
		self.restored = False

		self.device = FeedForwardNetwork.getDevice(self)
		self.to(self.device)
		
		self.rpSampler.loc = self.rpSampler.loc.to(self.device)
		self.rpSampler.scale = self.rpSampler.scale.to(self.device)
		self.optimizer = FeedForwardNetwork.createOptimizer(self, self.optimizerStr)

	def forward(self, x):
		"""
    	forward pass
		
		Parameters
			x : data batch
		"""
		#encode and then mean var of latent 
		xe = self.encoder(x)	
		mean = self.fcMean(xe)
		std = torch.exp(self.fcVar(xe))
		
		# sample with reparam trick from latent distr and then decode
		self.kl = (std ** 2 + mean ** 2 - torch.log(std) - 1/2).sum()
		z = mean + std * self.rpSampler.sample(mean.shape)
		xh = self.decoder(z)
		return xh

	@staticmethod
	def trainModel(model):
		"""
		train model
		
		Parameters
			model : torch model
		"""
		
		if model.dshape is None:
			#flat data
			trDataFile = model.config.getStringConfig("train.data.file")[0]
			featData = FeedForwardNetwork.prepDataNoLabel(model, trDataFile)
			featData = torch.from_numpy(featData)
			featData = featData.to(model.device)
			dataloader = DataLoader(featData, batch_size=model.batchSize, shuffle=True)
						
		for it in range(model.numIter):
			epochLoss = 0.0
			for x in dataloader:
				xh = model(x)
				loss = ((x - xh) ** 2).sum() + model.kl
				model.optimizer.zero_grad()
				loss.backward()
				model.optimizer.step()
				epochLoss += loss.item()
			epochLoss /= len(dataloader)
			print('epoch [{}-{}], loss {:.6f}'.format(it + 1, model.numIter, epochLoss))
	
		model.evaluateModel()
		
		if model.modelSave:
			FeedForwardNetwork.saveCheckpt(model)
			
	def regen(self, enData=None):
		"""
		get regenerated data
		"""
		self.eval()
		
		if enData is None:
			teDataFile = self.config.getStringConfig("encode.data.file")[0]
			enData = FeedForwardNetwork.prepDataNoLabel(self, teDataFile)
			
		enData = torch.from_numpy(enData)
		enData = enData.to(self.device)
		with torch.no_grad():
			regenData = self(enData)
			regenData = regenData.data.cpu().numpy()
		data = (enData.data.cpu().numpy(), regenData)
		return data

	def evaluateModel(self):
		"""
		evaluate model
		"""
		(enData, regenData) = self.regen()
		score = perfMetric(self.accMetric, enData, regenData)
		print("test regen  error {:.6f}".format(score))
		
	@staticmethod
	def predModel(model, doPlot=False):
		"""
		predict model regen error
		
		Parameters
			model : torch model
			doPlot : True if original and regnerated data are plotted
		"""
		if (model.useSavedModel):
			# load saved model
			FeedForwardNetwork.restoreCheckpt(model)
		else:
			#train
			VarAutoEncoder.trainModel(model)
		
		prDataFile = model.config.getStringConfig("pred.data.file")[0]
		enData = FeedForwardNetwork.prepDataNoLabel(model, prDataFile)
		scores = list()
		x = list(range(model.numinp))
		i = 1
		for ed in enData:
			enData, regenData = model.regen(ed)
			score = perfMetric(model.accMetric, enData, regenData)
			print("next rec {}   regen error {:.6f}".format(i, score))
			scores.append(score)
			if doPlot:
				drawPairPlot(x, enData, regenData, "time", "amplitude", "original", "regenerated")
			i += 1
		return scores
		
		

