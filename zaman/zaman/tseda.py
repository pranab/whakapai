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
	expl = __initDexpl(ds, "mydata")
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
	expl = __initDexpl(ds, "mydata")
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
	__shuffleData(config, bsize, "train.data.file", shTrDataFpath)
	__shuffleData(config, bsize, "validate.data.file", shVaDataFpath)
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
	expl = __initDexpl(ds, "mydata")
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
	
	data = __getListDatal(ds)
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

def meanStdDevShift(ds, wlen, rdata=None):
	"""
	detects mean and variance shift
		
	Parameters
		ds: file name and col index or list
		wlen : window length
		rdata : reference data
	"""
	mlist = list()
	slist = list()
	data = __getListDatal(ds)
	assertGreater(len(data), wlen, "data size should be larger than window size")
	
	mmdiff = None
	msdiff = None
	means = None
	sds = None
	
	def setMax(m1, s1, m2, s2):
		#max diff in mean and sd
		mdiff = abs(m1 - m2)
		sdiff = abs(s1 - s2)
		if mmdiff is None:
			mmdiff = mdiff
			msdiff = sdiff
			sds = (s1, s2)
			means = (m1, m2)
		else:
			if mdiff > mmdiff:
				mmdiff = mdiff
				mi = i
				sds = (s1, s2)
			if sdiff > msdiff:
				msdiff = sdiff
				si = i
				means = (m1, m2)
	
	if rdata is None:
		for i in range(len(data) - wlen):
			#use half windows
			beg = i
			half = beg + int(wlen / 2)
			end = beg + wlen
			m1, s1 = basicStat(data[beg:half])
			m2, s2 = basicStat(data[half:end])
			setMax(m1, s1, m2, s2)
	else:
		#use reference data
		rdata = __getListDatal(rdata)
		m1, s1 = basicStat(rdata)
		for i in range(len(data) - wlen):
			beg = i
			end = beg + wlen
			m2, s2 = basicStat(data[beg:end])
			setMax(m1, s1, m2, s2)

	res = __createResult("meanDiff", mmdiff, "meanDiffLoc", mi, "stdDeviations", sds, "sdDiff", msdiff, "sdDiffLoc", si, "means", means)
	return res

def twoSampleStat(ds, wlen, algo, rdata=None):
	"""
	two sample statistic
		
	Parameters
		ds: file name and col index or list
		wlen : window length
		algo : two sample stat algorithm
		rdata : reference data file name and col index or list
	"""
	maxKs = None
	maxi = None
	data = __getListDatal(ds)
	
	def setMax(res):
		ks = res["stat"]
		if maxKs is None or ks > maxKs:
			maxKs = ks
			maxi = i
	
	
	if rdata is None:
		# two half windows
		for i in range(len(data) - wlen):
			#use half windows
			beg = i
			half = beg + int(wlen / 2)
			end = beg + wlen
			__regData(data[beg:half],  "d1", expl)
			__regData(data[half:end],  "d2", expl)
			setMax(res)
	else:
		expl = __initDexpl(rdata, "d1")
		for i in range(len(data) - wlen):
			beg = i
			end = beg + wlen
			__regData(data[beg:end],  "d2", expl)
			res = expl.testTwoSampleKs("d1", "d2")
			setMax(res)
	
	res = __createResult("maxKS", maxKs, "maxLoc", maxi)
	return res
		
def fft(ds, srate):
	"""
	gets fft
		
	Parameters
		ds: list containing file name and col index or list of data
		srate : sampling rate	
	"""
	expl = __initDexpl(ds, "mydata")
	res = expl.getFourierTransform("mydata", srate)
	yf = res["fourierTransform"]
	xf = res["frquency"]
	res["fourierTransform"] = np.abs(yf)
	return res
	
def __getListDatal(ds):
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
		
def __initDexpl(ds, dsname, expl=None):
	"""
	initialize data explorer
		
	Parameters
		ds: file name and col index or list
		dsname : data sourceb name
	"""
	if expl is None:
		expl = DataExplorer()
	if type(ds[0]) == str:
		# file name and col index
		expl.addFileNumericData(ds[0], int(ds[1]), dsname)
	else:
		# list
		expl.addListNumericData(ds, dsname)
	return expl

def __regData(ds, dsname, expl):
	"""
	register data withdata explorer
		
	Parameters
		ds: file name and col index or list
		dsname : data source name
		expl ; data explorer
	"""
	if type(ds[0]) == str:
		# file name and col index
		expl.addFileNumericData(ds[0], int(ds[1]),dsname)
	else:
		# list
		expl.addListNumericData(ds, dsname)
	return expl

def __shuffleData(config, bsize, dataFileConf, shDataFpath):
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
	
	