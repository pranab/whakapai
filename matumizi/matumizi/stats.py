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

import sys
import random 
import time
import math
import numpy as np
from scipy import stats
import statistics 
from .util import *

"""
histogram class
"""
class Histogram:
	def __init__(self, vmin, binWidth):
		"""
    	initializer
    	
		Parameters
			min : min x
			binWidth : bin width
    	"""
		self.xmin = vmin
		self.xmax = None
		self.binWidth = binWidth
		self.normalized = False
		self.bins = None
		self.numBin = None
	
	@classmethod
	def createInitialized(cls, xmin, binWidth, values):
		"""
    	create histogram instance with min domain, bin width and values
    	
		Parameters
			min : min x
			binWidth : bin width
			values : distribution values (y)
    	"""
		instance = cls(xmin, binWidth)
		instance.xmax = xmin + binWidth * len(values)
		instance.ymin = 0
		instance.bins = np.array(values)
		instance.fmax = max(values)
		instance.ymin = 0.0
		instance.ymax = instance.fmax
		return instance

	@classmethod
	def createWithNumBins(cls, dvalues, numBins=20):
		"""
    	create histogram instance values and no of bins
    	
		Parameters
			dvalues : data values
			numBins : no of bins
		"""
		xmin = min(dvalues)
		xmin = 0.999 * xmin if xmin > 0 else 1.001 * xmin
		xmax = max(dvalues)
		xmax = 1.001 * xmax if xmax > 0 else 0.999 * xmax
		binWidth = (xmax  - xmin) / numBins
		instance = cls(xmin, binWidth)
		instance.xmax = xmax
		instance.numBin = numBins
		instance.bins = np.zeros(instance.numBin)
		for v in dvalues:
			instance.add(v)
		return instance
	
	@classmethod
	def createUninitialized(cls, xmin, xmax, binWidth):
		"""
    	create histogram instance with no y values using domain min , max and bin width
    	
		Parameters
			min : min x
			max : max x
			binWidth : bin width
    	"""
		instance = cls(xmin, binWidth)
		instance.xmax = xmax
		instance.numBin = int((xmax - xmin) / binWidth) + 1
		instance.bins = np.zeros(instance.numBin)
		return instance
	
	@classmethod
	def createUninitializedWithNumBins(cls, xmin, binWidth, nbins):
		"""
    	create histogram instance with no y values using domain min, bin width and num of bins
    	
		Parameters
			xmin : min x			
			binWidth : bin width
			nbins : num of bins
    	"""
		instance = cls(xmin, binWidth)
		instance.numBin = nbins
		instance.bins = np.zeros(instance.numBin)
		return instance

	def initialize(self):
		"""
    	set y values to 0
    	"""
		self.bins = np.zeros(self.numBin)
		self.normalized = False
		
	def add(self, value):
		"""
    	adds a value to a bin
    	
		Parameters
			value : value
    	"""
		bin = int((value - self.xmin) / self.binWidth)
		if (bin < 0 or  bin > self.numBin  - 1):
			print (bin)
			raise ValueError("outside histogram range")
		self.bins[bin] += 1.0
	
	def normalize(self):
		"""
    	normalize  bin counts
    	"""
		if not self.normalized:
			total = self.bins.sum()
			self.bins = np.divide(self.bins, total)
			self.normalized = True
	
	def cumDistr(self):
		"""
    	cumulative dists
    	"""
		self.normalize()
		self.cbins = np.cumsum(self.bins)
		return self.cbins
		
	def distr(self):
		"""
    	distr
    	"""
		self.normalize()
		return self.bins

		
	def percentile(self, percent):
		"""
    	return value corresponding to a percentile
    	
		Parameters
			percent : percentile value
    	"""
		if self.cbins is None:
			raise ValueError("cumulative distribution is not available")
			
		for i,cuml in enumerate(self.cbins):
			if percent > cuml:
				value = (i * self.binWidth) - (self.binWidth / 2) + \
				(percent - self.cbins[i-1]) * self.binWidth / (self.cbins[i] - self.cbins[i-1]) 
				break
		return value

	def entropy(self):
		"""
    	return entropy
    	
    	"""
		self.normalize()
		entr = 0
		for p in self.bins:
			if p > 0:
				entr += (-p * math.log(p))
		return entr
		
	def max(self):
		"""
    	return max bin value 
    	"""
		return self.bins.max()
	
	def value(self, x):
		"""
    	return a bin value	
     	
		Parameters
			x : x value
   		"""
		bin = int((x - self.xmin) / self.binWidth)
		f = self.bins[bin]
		return f

	def bin(self, x):
		"""
    	return a bin index	
     	
		Parameters
			x : x value
   		"""
		return int((x - self.xmin) / self.binWidth)
	
	def cumValue(self, x):
		"""
    	return a cumulative bin value	
     	
		Parameters
			x : x value
   		"""
		bin = int((x - self.xmin) / self.binWidth)
		c = self.cbins[bin]
		return c
	
		
	def getMinMax(self):
		"""
    	returns x min and x max
    	"""
		return (self.xmin, self.xmax)
		
	def boundedValue(self, x):
		"""
    	return x bounde by min and max	
     	
		Parameters
			x : x value
   		"""
		if x < self.xmin:
			x = self.xmin
		elif x > self.xmax:
			x = self.xmax
		return x
		
	def getBinWidth(self):
		"""
		return bin width
		"""
		return self.binWidth
	
"""
categorical histogram class
"""
class CatHistogram:
	def __init__(self):
		"""
    	initializer
    	"""
		self.binCounts = dict()
		self.counts = 0
		self.normalized = False
	
	def add(self, value):
		"""
		adds a value to a bin
		
		Parameters
			x : x value
		"""
		addToKeyedCounter(self.binCounts, value)
		self.counts += 1	
		
	def normalize(self):
		"""
		normalize
		"""
		if not self.normalized:
			self.binCounts = dict(map(lambda r : (r[0],r[1] / self.counts), self.binCounts.items()))
			self.normalized = True
	
	def getMode(self):
		"""
		get mode
		"""
		maxk = None
		maxv = 0
		#print(self.binCounts)
		for  k,v  in  self.binCounts.items():
			if v > maxv:
				maxk = k
				maxv = v
		return (maxk, maxv)	
	
	def getEntropy(self):
		"""
		get entropy
		"""
		self.normalize()
		entr = 0 
		#print(self.binCounts)
		for  k,v  in  self.binCounts.items():
			entr -= v * math.log(v)
		return entr

	def getUniqueValues(self):
		"""
		get unique values
		"""		
		return list(self.binCounts.keys())

	def getDistr(self):
		"""
		get distribution
		"""	
		self.normalize()	
		return self.binCounts.copy()
		
class RunningStat:
	"""
	running stat class
	"""
	def __init__(self):
   		"""
    	initializer	
   		"""
   		self.sum = 0.0
   		self.sumSq = 0.0
   		self.count = 0
	
	@staticmethod
	def create(count, sum, sumSq):
		"""
    	creates iinstance	
     	
		Parameters
			sum : sum of values
			sumSq : sum of valure squared
		"""
		rs = RunningStat()
		rs.sum = sum
		rs.sumSq = sumSq
		rs.count = count
		return rs
		
	def add(self, value):
		"""
		adds new value

		Parameters
			value : value to add
		"""
		self.sum += value
		self.sumSq += (value * value)
		self.count += 1

	def getStat(self):
		"""
		return mean and std deviation 
		"""
		mean = self.sum /self. count
		t = self.sumSq / (self.count - 1) - mean * mean * self.count / (self.count - 1)
		sd = math.sqrt(t)
		re = (mean, sd)
		return re

	def addGetStat(self,value):
		"""
		calculate mean and std deviation with new value added

		Parameters
			value : value to add
		"""
		self.add(value)
		re = self.getStat()
		return re
	
	def getCount(self):
		"""
		return count
		"""
		return self.count
	
	def getState(self):
		"""
		return state
		"""
		s = (self.count, self.sum, self.sumSq)
		return s
		
class SlidingWindowStat:
	"""
	sliding window stats
	"""
	def __init__(self):
		"""
		initializer
		"""
		self.sum = 0.0
		self.sumSq = 0.0
		self.count = 0
		self.values = None
	
	@staticmethod
	def create(values, sum, sumSq):
		"""
    	creates iinstance	
     	
		Parameters
			sum : sum of values
			sumSq : sum of valure squared
		"""
		sws = SlidingWindowStat()
		sws.sum = sum
		sws.sumSq = sumSq
		self.values = values.copy()
		sws.count = len(self.values)
		return sws
		
	@staticmethod
	def initialize(values):
		"""
    	creates iinstance	
     	
		Parameters
			values : list of values
		"""
		sws = SlidingWindowStat()
		sws.values = values.copy()
		for v in sws.values:
			sws.sum += v
			sws.sumSq += v * v		
		sws.count = len(sws.values)
		return sws

	@staticmethod
	def createEmpty(count):
		"""
    	creates iinstance	
     	
		Parameters
			count : count of values
		"""
		sws = SlidingWindowStat()
		sws.count = count
		sws.values = list()
		return sws

	def add(self, value):
		"""
		adds new value
		
		Parameters
			value : value to add
		"""
		self.values.append(value)		
		if len(self.values) > self.count:
			self.sum += value - self.values[0]
			self.sumSq += (value * value) - (self.values[0] * self.values[0])
			self.values.pop(0)
		else:
			self.sum += value
			self.sumSq += (value * value)
		

	def getStat(self):
		"""
		calculate mean and std deviation 
		"""
		mean = self.sum /self. count
		t = self.sumSq / (self.count - 1) - mean * mean * self.count / (self.count - 1)
		sd = math.sqrt(t)
		re = (mean, sd)
		return re

	def addGetStat(self,value):
		"""
		calculate mean and std deviation with new value added
		"""
		self.add(value)
		re = self.getStat()
		return re
	
	def getCount(self):
		"""
		return count
		"""
		return self.count
	
	def getCurSize(self):
		"""
		return count
		"""
		return len(self.values)
		
	def getState(self):
		"""
		return state
		"""
		s = (self.count, self.sum, self.sumSq)
		return s

class SlidingWindowAverage:
	"""
	sliding window stats
	"""
	def __init__(self, wsize):
		"""
		initializer
		
		Parameters
			wsize : window size
		"""
		self.mean = None
		self.wsize = wsize
		self.values = list()
		self.beg = None
	
	def extendEnds(self, data):
		"""
		extends at 2 ends by hal;f window
		
		Parameters
			data : data array
		"""
		hsize = int(self.wsize/2)
		bav = statistics.mean(data[:hsize])
		beg = [bav] * hsize
		eav = statistics.mean(data[-hsize:])
		end = [eav] * hsize
		edata = beg
		edata.extend(data.copy())
		edata.extend(end)
		return edata
		
	def add(self, value):
		"""
		adds new value
		
		Parameters
			value : value to add
		"""
		self.values.append(value)		
		if len(self.values) == self.wsize:
			self.mean = sum(self.values) / self.wsize
			hsize = int(self.wsize / 2) + 1
			self.beg = self.values[:hsize].copy()
		elif len(self.values) > self.wsize:
			self.mean += (self.values[-1] - self.values[0]) / self.wsize
			self.values.pop(0)
		return self.mean

	def getEnds(self):
		"""
		gets 2 half windows at ends with linear regression
		"""
		xd = list(range(len(self.beg)))
		slope, intercept, rvalue, pvalue, stderr = stats.linregress(xd, self.beg)	
		rsize = int(self.wsize / 2)
		bvalues = list(map(lambda x : x * slope + intercept, list(range(rsize))))
		
		hsize = int(self.wsize / 2)
		end = self.values[hsize:].copy()
		slope, intercept, rvalue, pvalue, stderr = stats.linregress(xd, end)
		evalues = list(map(lambda x : x * slope + intercept, list(range(1,rsize+1,1))))
		
		return bvalues, evalues
		
def basicStat(ldata):
	"""
	mean and std dev

	Parameters
		ldata : list of values
	"""
	m = statistics.mean(ldata)
	s = statistics.stdev(ldata, xbar=m)
	r = (m, s)
	return r

def getFileColumnStat(filePath, col, delem=","):
	"""
	gets stats for a file column
	
	Parameters
		filePath : file path
		col : col index
		delem : field delemter
	"""
	rs = RunningStat()
	for rec in fileRecGen(filePath, delem):
		va = float(rec[col])
		rs.add(va)
		
	return rs.getStat()
