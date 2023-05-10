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
from .util import *
from .mlutil import *
from .sampler import *
from .stats import *

class SupConceptDrift(object):
	"""
	supervised cpncept drift detection 
	"""
	def __init__(self, threshold, warmUp, wsize, wpsize):
		"""
		initializer
	
		Parameters
			threshold : threshold
			warmUp : warmup length
			wsize : window size
			wpsize : window processing step size
		"""
		assertGreaterEqual(warmUp, 50, "minimum warmup size is 50")
		self.threshold = threshold
		self.warmUp = warmUp
		self.wsize = wsize
		self.wpsize = wpsize
		self.warmedUp = False
		self.prMin = None
		self.sdMin = None
		self.count = 0
		self.ecount = 0
		self.sum = 0.0
		self.sumSq = 0.0		
		self.diMeanMax = None
		self.diSdMax = None
		self.maxAccRate = None
		self.evalues = list()
		self.evcount = 0
		self.scount = 0
		
		#ECDD
		self.pr = 0.0
		self.sd = 0.0
		self.expf = 0.0
		self.z = 0.0
		self.sdz = 0.0

	
	def add(self, evalue):
		"""
		detects drift online
	
		Parameters
			evalue : error value
		"""
		assertGreaterEqual(self.wsize, 50, "minimum window size is 50")
		self.scount += 1
		res = None
		if self.evcount < self.wsize:
			#fill window
			#print("filling window ", self.scount, self.evcount)
			self.evalues.append(evalue)
			self.evcount += 1
		else:
			#detect and then flush window
			#print("full window ", self.scount, self.evcount)
			res = self.detect(self.evalues)
			for _ in range(self.wpsize):
				self.evalues.pop(0)
			self.evcount -= self.wpsize
		return res

class DDM(SupConceptDrift):
	"""
	Drift Detection Method (DDM)
	"""
	def __init__(self, threshold, warmUp, wsize, wpsize):
		"""
		initializer
	
		Parameters
			threshold : threshold
			warmUp : warmup length
			wsize : window size
			wpsize : window processing step size
		"""
		super(DDM, self).__init__(threshold, warmUp, wsize, wpsize)

	def reset(self):
		"""
		resets counts
		"""
		self.count = 0
		self.ecount = 0
		
	def detect(self, evalues):
		"""
		detects drift in batch
	
		Parameters
			evalues : error value list
		"""
		self.reset()
		warmed = False
		if not self.warmedUp:
			assertGreaterEqual(self.wsize, self.warmUp, "window size should be greater than or equal to warmup size")
			for i in range(self.warmUp):
				if (evalues[i] == 1):
					self.ecount += 1
				self.count += 1
			
			self.prMin = self.ecount / self.count
			self.sdMin = math.sqrt(self.prMin * (1 - self.prMin) / self.count )
			self.warmedUp = True
			warmed = True
			#print("min {:.6f},{:.6f}".format(self.prMin, self.sdMin))
		
		result = None
		beg = self.warmUp if  warmed else 0
		remain = len(evalues) - beg
		if remain > 20:
			for i in range(beg, len(evalues), 1):
				if (evalues[i] == 1):
					self.ecount += 1
				self.count += 1
			
			pr = self.ecount / self.count
			sd = math.sqrt(pr * (1 - pr) / self.count)
			dr = 1 if (pr + sd) > (self.prMin + self.threshold * self.sdMin) else 0
			result = (pr, sd, pr + sd, dr)
			
			if (pr + sd) < (self.prMin +  self.sdMin):
				self.prMin = pr
				self.sdMin = sd
				#print("counts {},{}".format(self.count, self.ecount))
				#print("min {:.6f},{:.6f}".format(self.prMin, self.sdMin))
			
		return result
	
	
	def save(self, fpath):
		"""
		save DDM algorithm state
		
		Parameters
			fpath : file apth
		"""
		ws = dict()
		ws["warmUp"] = self.warmUp
		ws["warmedUp"] = self.warmedUp
		ws["wsize"] = self.wsize
		ws["prMin"] = self.prMin
		ws["sdMin"] = self.sdMin
		ws["threshold"] = self.threshold
		saveObject(ws, fpath)
			
	def restore(self, fpath):
		"""
		restore DDM algorithm state
		
		Parameters
			fpath : file apth
		"""
		ws = restoreObject(fpath)
		self.warmUp = ws["warmUp"]
		self.warmedUp = ws["warmedUp"]
		self.wsize = ws["wsize"]
		self.prMin = ws["prMin"]
		self.sdMin = ws["sdMin"]
		self.threshold = ws["threshold"]

class EDDM(SupConceptDrift):
	"""
	Early Drift Detection Method (EDDM)
	"""
	def __init__(self, threshold, warmUp, wsize, wpsize):
		"""
		initializer
	
		Parameters
			threshold : threshold
			warmUp : warmup length
			wsize : window size
			wpsize : window processing step size
		"""
		self.maxLim = None
		super(EDDM, self).__init__(threshold, warmUp, wsize, wpsize)

	def reset(self):
		"""
		resets counts
		"""
		self.count = 0
		self.sum = 0
		self.sumSq = 0
		
	def detect(self, evalues):
		"""
		detects drift in batch
	
		Parameters
			evalues : error value list
		"""
		self.reset()
		warmed = False
		rstat = RunningStat()
		lastEr = None
		if not self.warmedUp:
			for i in range(self.warmUp):
				if (evalues[i] == 1):
					if lastEr is not None:
						dist = i - lastEr
						rstat.add(dist)
					lastEr = i
			assertGreater(rstat.getCount(), 10, "not enough samples for warm up")
			re = rstat.getStat()
			
			self.diMeanMax = re[0]
			self.diSdMax = re[1]
			self.maxLim = self.diMeanMax + 2.0 * self.diSdMax	
			self.warmedUp = True
			warmed = True

		result = None
		beg = self.warmUp if  warmed else 0
		remain = len(evalues) - beg
		if remain > 20:
			lastEr = None
			for i in range(beg, len(evalues), 1):
				if (evalues[i] == 1):
					if lastEr is not  None:
						dist = i - lastEr
						re = rstat.add(dist)
					lastEr = i	
		
			if rstat.getCount() > 5:
				re = rstat.getStat()			
				cur = re[0] + 2.0 * re[1]
				dr = 1 if (cur / self.maxLim < self.threshold)  else 0
				result = (re[0],re[1], cur, dr)
				if cur > self.maxLim:
					self.diMeanMax = re[0]
					self.diSdMax = re[1]
					self.maxLim = cur
		return result

	def save(self, fpath):
		"""
		save EDDM algorithm state
		"""
		ws = dict()
		ws["warmUp"] = self.warmUp
		ws["warmedUp"] = self.warmedUp
		ws["wsize"] = self.wsize
		ws["diMeanMax"] = self.diMeanMax
		ws["diSdMax"] = self.diSdMax
		saveObject(ws, fpath)

	def restore(self, fpath):
		"""
		restore DDM algorithm state
		"""
		ws = restoreObject(fpath)
		self.warmUp = ws["warmUp"]
		self.warmedUp = ws["warmedUp"]
		self.wsize = ws["wsize"]
		self.diMeanMax = ws["diMeanMax"]
		self.diSdMax = ws["diSdMax"]

			