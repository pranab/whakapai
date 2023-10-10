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

import os
import sys
import matplotlib.pyplot as plt
from random import randint
from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import numpy as np
import joblib
from matumizi.util import *
from matumizi.mlutil import *

"""
Forecasting using simple network with decomposed time series
Ref: https://arxiv.org/abs/2205.13504
"""
class DecmpNetwork(object):
	def __init__(self, configFile):
		"""
    	In the constructor we instantiate two nn.Linear modules and assign them as
    	member variables.
    	
		Parameters
			configFile : config file path
		"""
		defValues = dict() 
		defValues["common.verbose"] = ("training", None)
		defValues["common.model.directory"] = ("model", None)
		defValues["common.model.file.trend"] = (None, None)
		defValues["common.model.file.remain"] = (None, None)
		defValues["common.trend.config.file"] = (None, "missing trend model config file")
		defValues["common.rem.config.file"] = (None, "missing remain model config file")
		defValues["decomp.maverage.window.size"] = (25, None)
		defValues["train.data.file"] = (None, "missing training data file")
		defValues["train.data.file.trend"] = (None, "missing training data file")
		defValues["train.data.file.remain"] = (None, "missing training data file")
		defValues["train.data.lookback.size"] = (16, None)
		defValues["train.data.forecast.size"] = (4, None)
		defValues["train.data.ts.col"] = (0, None)
		defValues["train.data.value.col"] = (1, None)
		defValues["train.data.split"] = (0.8, None)
		defValues["valid.data.file.trend"] = (None, "missing validation data file")
		defValues["valid.data.file.remain"] = (None, "missing validation data file")
		defValues["output.data.precision"] = (3, None)
		self.config = Configuration(configFile, defValues)
		super(DecmpNetwork, self).__init__()
		

	def decompose(self):
		"""
		decomposes TS into trend and remaining and reformats based on tookback and forecast size
		
		"""
		dfpath = self.config.getStringConfig("train.data.file")[0]
		tcol = self.config.getIntConfig("train.data.ts.col")[0]
		vcol = self.config.getIntConfig("train.data.value.col")[0]
		mawsz =  self.config.getIntConfig("decomp.maverage.window.size")[0]
		prec = self.config.getIntConfig("output.data.precision")[0]
		trfpathTr = self.config.getStringConfig("train.data.file.trend")[0]
		refpathTr = self.config.getStringConfig("train.data.file.remain")[0]
		trfpathVa = self.config.getStringConfig("valid.data.file.trend")[0]
		refpathVa = self.config.getStringConfig("valid.data.file.remain")[0]
		
		lbsize = self.config.getIntConfig("train.data.lookback.size")[0]
		fcsize = self.config.getIntConfig("train.data.forecast.size")[0]
		tsize = lbsize + fcsize
		
		ts = getFileColumnAsString(dfpath, vcol)
		tdata = getFileColumnAsFloat(dfpath, vcol)

		trsplit =  self.config.getFloatConfig("train.data.split")[0]
		dlen = len(tdata)
		trsplit = int(trsplit * dlen)
		vasplit = dlen - trsplit
		
		#trend
		mawindow = SlidingWindowAverage(mawsz)
		trmid = list(map(lambda d : mawindow.add(d), tdata))
		trb, tre = mawindow.getEnds()
		trend = trb.extend(trmid)
		trend.extend(tre)
		
		#trend training
		samples = list(map(lambda i : ternd[i:i+tsize], range(len(trend)-tsize)))
		with open(trfpathTr, "w") as ftrTr:
			for i in range(trsplit):
				ftrTr.write(floatArrayToString(samples[i], prec) + "\n")
		
		#trend validationg		
		with open(trfpathVa, "w") as ftrVa:
			for i in range(trsplit,dlen,1):
				ftrVa.write(floatArrayToString(samples[i], prec) + "\n")
			
		#remainder training
		remain = np.array(tdata) - np.array(trend)	
		remain = remain.tolist
		samples = list(map(lambda i : remain[i:i+tsize], range(len(remain)-tsize)))
		with open(refpathTr, "w") as freTr:
			for i in range(trsplit):
				freTr.write(floatArrayToString(samples[i], prec) + "\n")
		
		#remaining validationg		
		with open(refpathVa, "w") as freVa:
			for i in range(trsplit,dlen,1):
				freVa.write(floatArrayToString(samples[i], prec) + "\n")

		
	def fit(self):
		"""
		fit models for trend and remaining
		
		"""
		trcfpath = self.config.getStringConfig("common.trend.config.file")[0]
		recfpath = self.config.getStringConfig("common.rem.config.file")[0]
		
		#traun trend model
		trmod = FeedForwardNetwork(trcfpath)
		trmod.train()
		
		#train remain model
		remod = FeedForwardNetwork(recfpath)
		remod.train()
		
		
			
		
		
		
