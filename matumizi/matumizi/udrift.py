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
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.neighbors import KDTree
import random
import jprops
import statistics as stat
from matplotlib import pyplot
from sklearn.decomposition import PCA
from .util import *
from .mlutil import *
from .sampler import *
from .stats import *

"""
data or label drift detection
"""
class UnSupConceptDrift(object):
	"""
	drift detection without label feedback
	"""
	def __init__(self, rdata, wsize, wpsize):
		"""
		initializer
	
		Parameters
			rdata : reference data
			wsize : window size
			wpsize : window processing step size
		"""
		if type(rdata) == list:
			rdata = np.array(rdata)
		self.rdata = rdata
		self.nfeat = rdata.shape[1]
		self.rsize = rdata.shape[0]
		self.scount = 0
		self.wsize = wsize
		self.wpsize = wpsize
		self.vcount = 0
		self.cdata = list()
	
	def add(self, flist):
		"""
		detects drift online
	
		Parameters
			flist : feature value list
		"""
		assertGreaterEqual(self.wsize, 50, "minimum window size is 50")
		self.scount += 1
		res = None
		if self.vcount < self.wsize:
			#fill window
			self.cdata.append(value)
			self.vcount += 1
		else:
			#detect and then flush window
			res = self.detect()
			for _ in range(self.wpsize):
				self.cdata.pop(0)
			self.vcount -= self.wpsize
		return res

	def createResult(self, *values):
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

class DimRedDataDrift(UnSupConceptDrift):
	"""
	data drift detection with PCA dim reduction
	"""
	def __init__(self, rdata,  wsize, wpsize, ncomp, algo, sigLev=.05):
		"""
		initializer
	
		Parameters
			rdata : reference data
			wsize : window size
			wpsize : window processing step size
			ncomp ; num of PCA components
			algo : two sample stat technique
		"""
		self.ncomp = ncomp
		self.algo = algo
		self.sigLev = sigLev
		self.nhmsg = "probably same distribution" 
		self.ahmsg = "probably not same distribution"
		super(DimRedDataDrift, self).__init__(rdata,  wsize, wpsize)
		
	def detect(self):
		"""
		detects data drift in batch
		"""
		cdata = np.array(self.cdata)
		assertEqual(self.rdata.shape[1], cdata.shape[1], "reference data and current data don't have same number of columns")
		
		#consolidation and column wise mean substraction
		td = np.vstack((self.rdata,cdata)).copy()
		tsize = td.shape[0]
		mean = td.mean(axis=0)
		td = td - mean
			
		#PCA components
		pca = PCA(n_components=self.ncomp)
		pca.fit(td)
		comps = pca.components_
		
		#independent transformed data
		comps = comps.transpose()
		td = np.matmul(td, comps)
		assertEqual(td.shape, (tsize, self.ncomp), "invalid shape of PCA components")
		
		#two sample stat
		expl = DataExplorer()
		maxStat = None
		maxPvalue = None
		maxComp = None
		for i in range(self.ncomp):
			d = td[:, i]
			d1 = d[:self.rsize]
			d2 = d[self.rsize:]
			expl.addListNumericData(d1, "d1")
			expl.addListNumericData(d2, "d2")
			if algo == "ks":
				res = expl.testTwoSampleKs("d1", "d2", self.sigLev)
			else:
				exitWithMsg("invalid 2 sample statistic algo")
			
			stat = res["stat"]
			if maxStat is None or stat > maxStat:
				maxStat = stat
				maxPvalue = res["pvalue"]
				maxComp = i
		
		msg = self.nhMsg if maxPvalue > self.sigLev else self.ahMsg
		res = self.createResult("two sample stat", maxStat, "pvalue", maxPvalue, "maxIndComp", maxComp, "msg", msg)
		return res
			

