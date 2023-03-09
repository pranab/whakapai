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
import pandas as pd
import numpy as np
from fbprophet import Prophet
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.daexp import *

"""
time series untilities
"""

def doPlot(ds):
	"""
	gets auts correlation
		
	Parameters
		ds: file name and col index or list
	"""
	expl =_initDexpl(ds)
	expl.plot("mydata")

def autoCorr(ds, plot, nlags, alpha=.05):
	"""
	gets auts correlation
		
	Parameters
		ds: file name and col index or list
		plot : True if to be plotted
		lags: num of lags
		alpha: confidence level
	"""
	expl =_initDexpl(ds)
	auc = None
	if plot:
		expl.plotAutoCorr("mydata", nlags, alpha)
	else:
		auc = expl.getAutoCorr("mydata", nlags, alpha)["autoCorr"]
	return auc
		
def appEntropy(ds, m, r):
	"""
	approximate entropy for TS forecastability ref: https://en.wikipedia.org/wiki/Approximate_entropy
		
	Parameters
		ds : data array  actual data or file name and col index
		m : m parameter 
		r : r parameter 
	"""
	def _maxdist(xi, xj):
		return max([abs(ua - va) for ua, va in zip(xi, xj)])
		
	def _phi(m):
		x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
		C = [ len([1 for xj in x if _maxdist(xi, xj) <= r]) / (N - m + 1.0) for xi in x ]
		return sum(np.log(C)) / (N - m + 1.0)
		
	if type(ds[0]) == str:
		# file name and col index
		data = list(map(lambda v : float(v), fileSelFieldValueGen(ds[0], int(ds[1]))))
	else:
		# list
		data = ds
	
	#print(data)	
	N = len(data)
	return abs(_phi(m + 1) - _phi(m))	
    
def kaboudan(cnfFpath, changepoints, holidays, bsize, shTrDataFpath, shVaDataFpath):
	"""
	kaboudan for TS foresatibility 
		
	Parameters
		cnfFpath : forecaster config file path
		changepoints : change points
		holidays : holidays
		bsize : block size
		shTrDataFpath : shuffled training data file path
		shVaDataFpath : shuffled validation data file path
	"""
	forecaster = ProphetForcaster(cnfFpath)
	
	#error from normal data
	forecaster.train()
	err = forecaster.validate()
	
	#shuffled data
	config = forecaster.getConfig()
	_shuffleData(config, bsize, "train.data.file", shTrDataFpath)
	_shuffleData(config, bsize, "validate.data.file", shVaDataFpath)
	forecaster.train()
	serr = forecaster.validate()

def components(ds, model, freq, summaryOnly, doPlot=False):
	"""
	extracts trend, cycle and residue components of time series
		
	Parameters
		ds: list containing file name and col index or list of data
		model : model type
		freq : seasnality period
		summaryOnly : True if only summary needed in output
		doPlot: true if plotting needed
		"""	
	expl =_initDexpl(ds)
	return expl.getTimeSeriesComponents("mydata", model, freq, summaryOnly, doPlot)

def _initDexpl(ds):
	"""
	initialize data explorer
		
	Parameters
		ds: file name and col index or list
	"""
	expl = DataExplorer()
	if type(ds[0]) == str:
		# file name and col index
		expl.addFileNumericData(ds[0], int(ds[1]), "mydata")
	else:
		# list
		expl.addListNumericData(ds, "mydata")
	return expl

def _shuffleData(config, bsize, dataFileConf, shDataFpath):
	"""
	block shuffles data
		
	Parameters
		config : config object
		bsize : shuffle block size
		dataFileConf : data file path config name
		shDataFpath : shuffled data file path
	"""
	dataFpath = config.getStringConfig(dataFileConf)[0]
	assert dataFpath, "missing input data file path"
	df = pd.read_csv(dataFpath, header=None, names=["ds", "y"])
	df.set_index("ds")
	dsValues = df.loc[:,"ds"].values		
	yValues = df.loc[:,"y"].values	

	# shuffle and write training data
	shValues = blockShuffle(yValues, bSize)	
	assert shDataFpath, "missing shuffled data file path"
	with open(shDataFpath, 'w') as shFile:
		for z in zip(dsValues, shValues):
			line = "%s,%.3f\n" %(z[0], z[1])
			shFile.write(line)
	
	# set config with shugffled data file path
	config.setParam(dataFileConf, shDataFpath)
	
	