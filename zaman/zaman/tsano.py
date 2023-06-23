#!/usr/bin/python

# whakapai/zaman: time series
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
from random import randint
from datetime import datetime
from dateutil.parser import parse
import numpy as np
from sklearn import preprocessing
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.daexp import *
from matumizi.stats import *


"""
time series anomaly detection
"""

class MarkovChainAnomaly:
	"""
	anomaly detection with markov chain and conditional probability
	"""
	def __init__(self,  configFile):
		"""
		initilizers
		
		Parameters
			configFile : config file path
			callback : user defined function
		"""
		defValues = dict()
		defValues["train.data.file"] = (None, None)
		defValues["train.data.field"] = (None, None)
		defValues["train.discrete.size"] = (None, "missing discretization size")
		defValues["train.val.margin"] = (20.0, None)
		defValues["train.save.model"] = (False, None) 
		defValues["train.model.file"] = (None, None) 
		defValues["pred.data.file"] = (None, None)
		defValues["pred.data.field"] = (None, None)
		defValues["pred.ts.field"] = (None, None)
		defValues["pred.window.size"] = (5, None)
		defValues["pred.ano.threshold"] = (None, "missing cond probability threshold")
		defValues["pred.output.file"] = (None, "missing output file path")
		defValues["pred.output.prec"] = (8, None)

		self.config = Configuration(configFile, defValues)
		
	def fit(self, tsval=None):
		"""
    	buils conditional probability table
    	
		Parameters
			tsval : time series value list
		"""
		vmargin = self.config.getFloatConfig("train.val.margin")[0]
		dsize = self.config.getFloatConfig("train.discrete.size")[0]
		toSave = self.config.getBooleanConfig("train.save.model")[0]
		mfpath = self.config.getStringConfig("train.model.file")[0]
		
		if tsval is None:
			self.config.assertParams("train.data.file", "train.data.field")
			fpath = self.config.getStringConfig("train.data.file")[0]
			tscol = self.config.getIntConfig("train.data.field")[0]
			tsval = getFileColumnAsFloat(fpath, tscol)
		
		vmax = max(tsval)
		vmin = min(tsval)
		vra = vmax - vmin
		
		#increase bin range for ani=omaly
		vmin -= vra * vmargin
		vmax += vra * vmargin
		nbins = int((vmax - vmin) / dsize) + 1
		
		#state tyransition probability matrix
		stpr = np.empty(shape=(nbins,nbins))
		stpr.fill(1)
		for i in range(len(tsval) - 1):
			sb = round((tsval[i] - vmin) / dsize)
			tb = round((tsval[i+1]- vmin) / dsize)
			stpr[sb][tb] += 1
		
		#normalize rows
		stpr = preprocessing.normalize(stpr, norm="l1", axis=1)
		#for i in range(nbins):
		#	print(stpr[i])
		
		if toSave:
			self.config.assertParams("train.model.file")
			mod = {"vmin":vmin, "vmax":vmax, "nbins":nbins, "stpr":stpr}
			saveObject(mod, mfpath)
			

	def predict(self, tsval=None):
		"""
    	predicts anomaly in sub sequence
    	
		Parameters
			tsval : time series value list
		"""
		dsize = self.config.getFloatConfig("train.discrete.size")[0]
		mfpath = self.config.getStringConfig("train.model.file")[0]
		wsize = self.config.getIntConfig("pred.window.size")[0]
		thresh = self.config.getFloatConfig("pred.ano.threshold")[0]
		ofpath = self.config.getStringConfig("pred.output.file")[0]
		oprec = self.config.getIntConfig("pred.output.prec")[0]
		ts = None
		
		if tsval is None:
			#file path from config
			self.config.assertParams("pred.data.file", "pred.data.field", "pred.ts.field")
			fpath = self.config.getStringConfig("pred.data.file")[0]
			tsvcol = self.config.getIntConfig("pred.data.field")[0]
			tsval = getFileColumnAsFloat(fpath, tsvcol)
			tscol = self.config.getIntConfig("pred.ts.field")[0]
			ts = getFileColumnAsString(fpath, tscol)
		
		#restore model
		mod = restoreObject(mfpath)
		vmin = mod["vmin"]
		stpr = mod["stpr"]
		
		cprmin = 1.0
		imin = 0
		cprl = list()
		for i in range(len(tsval) - wsize):
			cpr = 1.0
			for j in range(wsize - 1):
				k = i + j
				sb = round((tsval[k] - vmin) / dsize)
				tb = round((tsval[k+1]- vmin) / dsize)
				cpr *= stpr[sb][tb]
			
			cprl.append(formatFloat(oprec, cpr))
			if cpr < cprmin:
				cprmin = cpr
				imin = i
				#print("cpr {:.6f}".format(cpr))
			if cpr < thresh:
				cprs = formatFloat(oprec, cpr)
				print("seq anomaly {}  score {}  loc index {}".format(str(tsval[i:i+wsize]), cprs, i))
		#print("min prob {:.6f}  loc {}".format(cprmin, imin))

		#output
		if ts is None:
			with open(ofpath,'w') as fi:
				for va in cprl:
					fi.write(va + '\n')
		else:
			#num of cond prob values will ve lower depending on the window size
			ts = ts[:len(cprl)]
			with open(ofpath,'w') as fi:
				for t, v in zip(ts, cprl):
					fi.write(t +  "," + v + '\n')
				
		