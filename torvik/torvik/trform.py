
#!/usr/local/bin/python3

# avenir-python: Machine Learning
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
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import sklearn as sk
import matplotlib
import random
import jprops
from random import randint
from matumizi.util import *
from matumizi.mlutil import *
from .tnn import FeedForwardNetwork

"""
Transformer implementation
Inspired by https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

class Transformer(nn.Module):
	"""
	"""
	def __init__(self, configFile):
		defValues = dict()
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file"] = (None, None)
		defValues["common.preprocessing"] = (None, None)
		defValues["common.scaling.method"] = ("zscale", None)
		defValues["common.scaling.minrows"] = (50, None)
		defValues["common.verbose"] = (False, None)
		defValues["common.device"] = ("cpu", None)
		defValues["train.data.file"] = (None, "missing training data file path")
		defValues["train.data.type"] = ("numeric", None)
		defValues["train.data.delim"] = (",", None)
		defValues["train.hidden.size"] = (None, "missing  hidden size")
		defValues["train.embed.dict.size"] = (1000000, None)
		defValues["train.seq.len"] = (1, None)
		defValues["train.batch.size"] = (32, None)
		defValues["train.batch.first"] = (False, None)
		defValues["train.drop.prob"] = (0, None)
		defValues["train.optimizer"] = ("adam", None)
		defValues["train.opt.learning.rate"] = (.0001, None)
		defValues["train.opt.weight.decay"] = (0, None)
		defValues["train.opt.momentum"] = (0, None)
		defValues["train.opt.eps"] = (1e-08, None)
		defValues["train.opt.dampening"] = (0, None)
		defValues["train.opt.momentum.nesterov"] = (False, None)
		defValues["train.opt.betas"] = ([0.9, 0.999], None)
		defValues["train.opt.alpha"] = (0.99, None)
		defValues["train.out.sequence"] = (True, None)
		defValues["train.out.activation"] = ("sigmoid", None)
		defValues["train.loss.fn"] = ("mse", None) 
		defValues["train.loss.reduction"] = ("mean", None)
		defValues["train.grad.clip"] = (5, None) 
		defValues["train.num.iterations"] = (500, None)
		defValues["train.save.model"] = (False, None) 
		defValues["valid.data.file"] = (None, "missing validation data file path")
		defValues["valid.accuracy.metric"] = (None, None)
		defValues["predict.data.file"] = (None, None)
		defValues["predict.use.saved.model"] = (True, None)
		defValues["predict.output"] = ("binary", None)
		defValues["predict.feat.pad.size"] = (60, None)

		self.config = Configuration(configFile, defValues)
		super(Transformer, self).__init__()
    
    
	def buildModel(self):
		"""
		Loads configuration and builds the various piecess necessary for the model
		"""
		torch.manual_seed(9999)
		self.seqLen = self.config.getIntConfig("train.seq.len")[0]
		self.modSize = self.config.getIntConfig("train.hidden.size")[0]
		self.numHeads = self.config.getIntConfig("train.num.heads")[0]
		self.numEncLayer = self.config.getIntConfig("train.num.enc.layers")[0]
		self.numDecLayer = self.config.getIntConfig("train.num.dec.layers")[0]
		self.dropProb = self.config.getFloatConfig("train.drop.prob")[0]
		self.batchSize = self.config.getIntConfig("train.batch.size")[0]
		self.ffSize = self.config.getIntConfig("train.ff.size")[0]
		self.activation = self.config.getStringConfig("train.activation")[0]
		self.device = FeedForwardNetwork.getDevice(self)
		embedDicSize = self.config.getIntConfig("train.embed.dict.size")[0]
		self.positionalEncoder = PositionalEncoding(self.modSize, self.dropProb)
		self.embedding = nn.Embedding(embedDicSize, self.modSize)
		self.model = nn.Transformer(d_model=self.modSize, nhead=self.numHeads, num_encoder_layers=self.numEncLayer,
			num_decoder_layers=self.numDecLayer, dim_feedforward=self.ffSize, dropout=self.dropProb, activation=self.activation, batch_first=True)
		self.out = nn.Linear(self.modSize, self.seqLen)
		self.SOSToken = np.array([2])
		self.EOSToken = np.array([3])
		
		optimizerName = self.config.getStringConfig("train.optimizer")[0]
		self.optimizer = FeedForwardNetwork.createOptimizer(self, optimizerName)
		lossFnName = self.config.getStringConfig("train.loss.fn")[0]
		self.lossFn = FeedForwardNetwork.createLossFunction(self, lossFnName)
    	
	def forward(self, src, tgt, tgtMask=None, srcPadMask=None, tgtPadMask=None):
		"""
		forward pass
		Parameters
			src : source tensor (batch_size, src sequence length)
			tgt : target tensor (batch_size, tgt sequence length)
			tgtMask : target mask
			srcPadMask : src pad mask
			tgtPadMask : target pad mask
		"""
		src = self.embedding(src) * math.sqrt(self.modSize)
		tgt = self.embedding(tgt) * math.sqrt(self.modSize)
		src = self.positionalEncoder(src)
		tgt = self.positionalEncoder(tgt)
		
		#transformer blocks - Out size = (sequence length, batch_size, num_tokens)
		tout = self.model(src, tgt, tgt_mask=tgtMask, src_key_padding_mask=srcPadMask, tgt_key_padding_mask=tgtPadMask)
		out = self.out(tout)
		
		return out
      
	def getTgtMask(self, size):
		"""
    	generates square target mask matrix
    	
    	Parameters
    		size: mask size
		"""
		mask = torch.tril(torch.ones(size, size) == 1)
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf'))
		mask = mask.masked_fill(mask == 1, float(0.0))
		return mask
    
	def createPadMask(self, matrix: torch.tensor, padToken: int):
		"""
    	
		"""
		return (matrix == padToken)
        
    
	def loadData(self, fpath):
		"""
		loads data and creates batches
		
		Parameter
			fpath: data file path
		"""
		dataFile = self.config.getStringConfig("train.data.file")[0]
		data = np.loadtxt(file, delimiter=",")
		pdata = list()
		for r in data:
			x = np.concatenate((self.SOSToken, r, self.EOSToken))
			y = np.concatenate((self.SOSToken, r, self.EOSToken))
			pdata.append([x, y])
		np.random.shuffle(pdata)
		batches = list()
		nbatch = int(len(pdata) / self.batchSize)
		for idx in range(0, nbatch * self.batchSize, self.batchSize):
			batches.append(np.array(data[idx : idx +  self.batchSize]).astype(np.int64))
		return batches
    	
	def train(self):
		"""
    	train model
		"""
		batches = self.loadData(self.config.getStringConfig("train.data.file")[0])
		self.train()
		numIter = self.config.getIntConfig("train.num.iterations")[0]
		
		for it in range(numIter):
			b = 0
			for batch in batches:
				x, y = batch[:, 0], batch[:, 1]
				x, y = torch.tensor(X).to(self.device), torch.tensor(y).to(self.device)
    			
				#we shift the tgt by one so with the <SOS> we predict the token at pos 1
				yInp = y[:,:-1]
				yExp = y[:,1:]
				
				#get mask to mask out the next words
				tgtMask = self.getTgtMask(self.seqLen).to(self.device)
				
				#predict
				yPred = self.model(x, yInp, tgtMask)
				loss = self.lossFn(yPred, yExp)
				
				if self.verbose and it % 50 == 0 and b % 10 == 0:
					print("epoch {}  batch {}  loss {:.6f}".format(it, b, loss.item()))
    			
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				b += 1
		
		self.validate()
    	
	def validate(self):
		self.eval()
		batches = self.loadData(self.config.getStringConfig("valid.data.file")[0])
    		
		with torch.no_grad():
			for batch in batches:
				x, y = batch[:, 0], batch[:, 1]
				x, y = torch.tensor(X).to(self.device), torch.tensor(y).to(self.device)
				
				#we shift the tgt by one so with the <SOS> we predict the token at pos 1
				yInp = y[:,:-1]
				yExp = y[:,1:]
				
				#get mask to mask out the next words
				tgtMask = self.getTgtMask(self.seqLen).to(self.device)
				
				#predict
				yPred = self.model(x, yInp, tgtMask)
				loss = self.lossFn(yPred, yExp)
				tloss += loss.detach().item()
		return tloss / len(batches)			

class PositionalEncoding(nn.Module):
	def __init__(self, modelSize, dropoutProb, maxLen=5000):
		"""
    	
		"""
		super().__init__()
		self.dropout = nn.Dropout(dropoutProb)
		
		# Encoding - From formula
		posEncoding = torch.zeros(maxLen, modelSize)
		positionsList = torch.arange(0, maxLen, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
		divisionTerm = torch.exp(torch.arange(0, modelSize, 2).float() * (-math.log(10000.0)) / modelSize) # 1000^(2i/dim_model)
		
		# PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
		posEncoding[:, 0::2] = torch.sin(positionsList * divisionTerm)
		
		# PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
		posEncoding[:, 1::2] = torch.cos(positionsList * divisionTerm)
		
		# Saving buffer (same as parameter without gradients needed)
		posEncoding = posEncoding.unsqueeze(0).transpose(0, 1)
		self.register_buffer("posEncoding",posEncoding)
        
	def forward(self, tokenEmbedding: torch.tensor) -> torch.tensor:
		"""
    	forward pass
    	
		Parameters
			tokenEmbedding : token embedding
		"""
		# Residual connection + pos encoding
		return self.dropout(tokenEmbedding + self.posEncoding[:tokenEmbedding.size(0), :])
        
        
        