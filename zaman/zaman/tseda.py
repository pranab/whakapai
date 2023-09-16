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
from scipy import signal
import pywt
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
	
	data = getListData(ds)
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
		 
	res = createExplResult("meanValues", mlist, "varvalues", vlist)
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
	data = getListData(ds)
	assertGreater(len(data), wlen, "data size should be larger than window size")
	
	mmdiff = None
	msdiff = None
	means = None
	sds = None
	mi = None
	si = None
	
	def setMax(m1, s1, m2, s2, i):
		#max diff in mean and sd
		nonlocal mmdiff
		nonlocal msdiff
		nonlocal means
		nonlocal sds
		nonlocal mi
		nonlocal si
		
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
			setMax(m1, s1, m2, s2, half)
	else:
		#use reference data
		rdata = getListData(rdata)
		m1, s1 = basicStat(rdata)
		for i in range(len(data) - wlen):
			beg = i
			end = beg + wlen
			m2, s2 = basicStat(data[beg:end])
			setMax(m1, s1, m2, s2, i)

	res = createExplResult("meanDiff", mmdiff, "meanDiffLoc", mi, "stdDeviations", sds, "sdDiff", msdiff, "sdDiffLoc", si, "means", means)
	return res

def twoSampleStat(ds, wlen, pstep, algo, rdata=None):
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
	maxPvalue = None
	data = getListData(ds)
	
	def setMax(res):
		nonlocal maxKs
		nonlocal maxi
		nonlocal maxPvalue
		ks = res["stat"]
		if maxKs is None or ks > maxKs:
			maxKs = ks
			maxPvalue = res["pvalue"]
			maxi = i
	
	
	if rdata is None:
		# two half windows
		expl = DataExplorer()
		for i in range(0, len(data) - wlen, pstep):
			#use half windows
			beg = i
			half = beg + int(wlen / 2)
			end = beg + wlen
			__regData(data[beg:half],  "d1", expl)
			__regData(data[half:end],  "d2", expl)
			if algo == "ks":
				res = expl.testTwoSampleKs("d1", "d2")
			else:
				exitWithMsg("invalid 2 sample statistic algo")
			setMax(res)
	else:
		expl = __initDexpl(rdata, "d1")
		for i in range(0, len(data) - wlen, pstep):
			beg = i
			end = beg + wlen
			__regData(data[beg:end],  "d2", expl)
			if algo == "ks":
				res = expl.testTwoSampleKs("d1", "d2")
			else:
				exitWithMsg("invalid 2 sample statistic algo")
			setMax(res)
	
	res = createExplResult("maxKS", maxKs, "pvalue", maxPvalue, "loc", maxi)
	return res
		
def fft(ds, srate):
	"""
	gets fft
		
	Parameters
		ds: list containing file name and col index or list of data
		srate : sampling rate	
	"""
	expl = __initDexpl(ds, "mydata")
	re = expl.getFourierTransform("mydata", srate)
	yf = re["fourierTransform"]
	xf = re["frquency"]
	res = createExplResult("frquency", xf, "fft", np.abs(yf))
	return res
	
	
def bhpassFilter(ds, cutoff, fs, order=5):
	"""
	high pass filter
		
	Parameters
		ds: list containing file name and col index or list of data
		cutoff : cut off frequency
		fs : sampling frequency
		order : order	
	"""
	data = getListData(ds)
	b, a = __bhpass(cutoff, fs, order=order)
	y = signal.filtfilt(b, a, data)
	return y

def getListData(ds):
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

def __bhpass(cutoff, fs, order=5):
	"""
	creates high pass filter
		
	Parameters
		cutoff : cut off frequency
		fs : sampling frequency
		order : filter order	
	"""
	nyq = 0.5 * fs
	ncutoff = cutoff / nyq
	b, a = signal.butter(order, ncutoff, btype='high', analog=False)
	return b, a

def createExplResult(*values):
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
	

class MeanStdShiftDetector(SlidingWindowProcessor):
	"""
	online detection of mean and std deviation shift
	"""
	def __init__(self,  wsize, pstep, rdata=None):
		"""
		initilizers
		
		Parameters
			wsize : window size
			pstep : processing step size
			rdata : reference data
		"""
		self.mmdiff = None
		self.msdiff = None
		self.mi = None
		self.si = None
		self.means = None
		self.sds = None
		self.pcount = 0
		self.rdata = None
		self.mr = None
		self.sr = None
		self.useRdata = False
		if rdata is not None:
			rdata = getListData(rdata)
			self.mr, self.sr = basicStat(rdata)
			self.useRdata = True
		self.mdiffs = list()
		self.sdiffs = list()
		
		super(MeanStdShiftDetector, self).__init__(wsize, pstep)	
		
	
	def process(self):
		"""
		processes window
		
		"""
		self.pcount += 1
		data = self.window
		wlen = self.wsize
		half = int(wlen / 2)
		if not self.useRdata:
			#use half windows
			m1, s1 = basicStat(data[:half])
			m2, s2 = basicStat(data[half:])
			self.__setMax(m1, s1, m2, s2)
		else:
			#use reference data
			m2, s2 = basicStat(data)
			self.__setMax(self.mr, self.sr, m2, s2)
	
	def getResult(self):
		"""
		get results
		"""
		res = createExplResult("meanDiff", self.mmdiff, "meanDiffLoc", self.mi, "stdDeviations", self.sds, "sdDiff", self.msdiff, 
		"sdDiffLoc", self.si, "means", self.means)
		return res
	
	def getDiffList(self):
		"""
		get lists of mean diff and std dev diff
		"""
		return (self.mdiffs, self.sdiffs)
		
	def __setMax(self, m1, s1, m2, s2):
		"""
		sets max values
		
		Parameters
			m1 : first mean
			s1 : first std dev
			m2 : second mean
			s2 : second std dev
		"""
		#max diff in mean and sd
		mdiff = abs(m1 - m2)
		sdiff = abs(s1 - s2)
		self.mdiffs.append(mdiff)
		self.sdiffs.append(sdiff)
		if self.mmdiff is None:
			self.mmdiff = mdiff
			self.msdiff = sdiff
			self.sds = (s1, s2)
			self.means = (m1, m2)
		else:
			if mdiff > self.mmdiff:
				self.mmdiff = mdiff
				self.mi = self.pcount
				self.sds = (s1, s2)
			if sdiff > self.msdiff:
				self.msdiff = sdiff
				self.si = self.pcount
				self.means = (m1, m2)

class WaveletExpl(object):
	"""
	time and freq domain exploration with wavelet
	"""
	def __init__(self,  data, wavelet, sampf, scales=None, freqs=None):
		"""
		initilizers
		
		Parameters
			data : data
			wavelet : wavelet function
			sampf : sampling frequency
			scales : scale list
			freqs ; frequency list should be specified if no scales specified
		"""
		self.data = data
		self.wavelet = wavelet
		self.sampf = sampf
		if scales is None:
			freqs = np.array(freqs) / sampf
			self.scales = pywt.frequency2scale(self.wvlet, freqs)
		else:
			self.scales = scales
		
	
	def transform(self):
		"""
		wavelet transform
		"""
		self.tcoef, self.tfreqs = pywt.cwt(self.data, self.scales, self.wvlet, sampling_period=1.0/self.sampf)
		
	def atFreq(self, iscale, doPlot=True, nparts=2):
		"""
		contrubtion of a freq at all times
		
		Parameters
			iscale : index into freq or scale list list
			doPlot : true if to be plotted
			nparts : num of plots
		"""	
		trform = self.tcoef[iscale]
		if doPlot:
			drawPlotParts(None, trform, "time", "value", nparts)
		return trform
		
	def atTime(self, itime, doPlot=True):
		"""
		contrubtion of a freq at all times
		
		Parameters
			iscale : index into freq or scale list list
			doPlot : true if to be plotted
			nparts : num of plots
		"""	
		trform = self.tcoef[:,itime]
		if doPlot:
			drawPlot(self.tfreqs, trform, "freq", "value")
		return trform
		