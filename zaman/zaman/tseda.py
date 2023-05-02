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
from matumizi.stats import *


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

def meanVarNonStationarity(ds, wlen, doPlot=True):
	"""
	mean and variance based test for non stationarity with trend, sotachstic trend or heteroscedastic
		
	Parameters
		ds: file name and col index or list
		wlen : window length
		doPlot : plotted if True
	"""
	mlist = list()
	vlist = list()
	
	data = _getListData(ds)
	assertGreater(len(data), wlen, "data size should be larger than window size")
	rwin = SlidingWindowStat.initialize(data[:wlen])
	m, s = rwin.getStat()
	mlist.append(m)
	vlist.append(s * s)
	
	#iterate rolling window
	for i in range(wlen, len(data), 1):
		m, s = rwin.addGetStat(data[i])	
		mlist.append(m)
		vlist.append(s * s)
	
	if doPlot:
		drawLine(mlist)
		drawLine(vlist)
		 
	res = __createResult("meanValues", mlist, "varvalues", vlist)
	return res

def meanVarShift(ds, wlen, doPlot=True):
	"""
	detects mean and variance shift
		
	Parameters
		ds: file name and col index or list
		wlen : window length
		doPlot : plotted if True
	"""
	mlist = list()
	slist = list()
	data = _getListData(ds)
	assertGreater(len(data), wlen, "data size should be larger than window size")
	
	mmdiff = None
	msdiff = None
	for i in range(len(data) - wlen):
		#mean and sd of each half window
		beg = i
		half = beg + int(wlen / 2)
		end = beg + wlen
		m1, s1 = basicStat(data[beg:half])
		m2, s2 = basicStat(data[half:end])
		
		#max diff in mean and sd
		mdiff = abs(m1 - m2)
		sdiff = abs(s1 - s2)
		if mmdiff is None:
			mmdiff = mdiff
			msdiff = sdiff
		else:
			if mdiff > mmdiff:
				mmdiff = mdiff
				mi = i
			if sdiff > msdiff:
				msdiff = sdiff
				si = i
			
	res = __createResult("meanDiff", mmdiff, "meanDiffLoc", mi, "sdDiff", msdiff, "sdDiffLoc", si)
	return res
			
def fft(ds, srate):
	"""
	gets fft
		
	Parameters
		ds: list containing file name and col index or list of data
		srate : sampling rate	
	"""
	expl =_initDexpl(ds)
	res = expl.getFourierTransform("mydata", srate)
	yf = res["fourierTransform"]
	xf = res["frquency"]
	res["fourierTransform"] = np.abs(yf)
	return res
	
def _getListData(ds):
	"""
	gets lists data from file column or returns list as is
		
	Parameters
		ds: file name and col index or list
	"""
	if type(ds[0]) == str:
		# file name and col index
		data = getFileColumnAsFloat(ds[0], ds[1])
	else:
		# list
		data = ds
	return data
		
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

def __createResult(*values):
	"""
	create result map
		
	Parameters
		values : flattened kay and value pairs
	"""
	result = dict()
	assert len(values) % 2 == 0, "key value list should have even number of items"
	for i in range(0, len(values), 2):
		result[values[i]] = values[i+1]
	return result
	
	