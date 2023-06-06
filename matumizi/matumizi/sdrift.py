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
	
	@classmethod
	def create(cls, fpath):
		"""
		factory method to create based on restored model
	
		Parameters
			fpath : file path for saved model
		"""
		ws = restoreObject(fpath)
		obj = cls(ws["threshold"], ws["warmUp"], ws["wsize"], ws["wpsize"])
		obj.warmedUp = ws["warmedUp"]
		obj.prMin = ws["prMin"]
		obj.sdMin = ws["sdMin"]
		return obj

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
			fpath : file path for model checkpointing
		"""
		ws = dict()
		ws["warmUp"] = self.warmUp
		ws["warmedUp"] = self.warmedUp
		ws["wsize"] = self.wsize
		ws["wpsize"] = self.wpsize
		ws["prMin"] = self.prMin
		ws["sdMin"] = self.sdMin
		ws["threshold"] = self.threshold
		saveObject(ws, fpath)
			
	def restore(self, fpath):
		"""
		restore DDM algorithm state
		
		Parameters
			fpath : file path for model checkpointing
		"""
		ws = restoreObject(fpath)
		self.warmUp = ws["warmUp"]
		self.warmedUp = ws["warmedUp"]
		self.wsize = ws["wsize"]
		self.wpsize = ws["wpsize"]
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

	@classmethod
	def create(cls, fpath):
		"""
		factory method to create based on restored model
	
		Parameters
			fpath : file path for saved model
		"""
		ws = restoreObject(fpath)
		obj = cls(ws["threshold"], ws["warmUp"], ws["wsize"], ws["wpsize"])
		obj.warmedUp = ws["warmedUp"]
		obj.diMeanMax = ws["diMeanMax"]
		obj.diSdMax = ws["diSdMax"]
		obj.maxLim = ws["maxLim"]
		return obj
			
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
		
		Parameters
			fpath : file path for model checkpointing
		"""
		ws = dict()
		ws["threshold"] = self.threshold
		ws["warmUp"] = self.warmUp
		ws["warmedUp"] = self.warmedUp
		ws["wsize"] = self.wsize
		ws["wpsize"] = self.wpsize
		ws["diMeanMax"] = self.diMeanMax
		ws["diSdMax"] = self.diSdMax
		ws["maxLim"] = self.maxLim
		saveObject(ws, fpath)

	def restore(self, fpath):
		"""
		restore DDM algorithm state
		
		Parameters
			fpath : file path for model checkpointing
		"""
		ws = restoreObject(fpath)
		self.threshold = ws["threshold"]
		self.warmUp = ws["warmUp"]
		self.warmedUp = ws["warmedUp"]
		self.wsize = ws["wsize"]
		self.wpsize = ws["wpsize"]
		self.diMeanMax = ws["diMeanMax"]
		self.diSdMax = ws["diSdMax"]
		self.maxLim = ws["maxLim"]

class FHDDM(SupConceptDrift):
	"""
	Fast Hoeffding Drift Detection Method (FHDDM)
	"""
	def __init__(self, confLevel, warmUp, wsize, wpsize):
		"""
		initializer
	
		Parameters
			confLevel : confidence level
			warmUp : warmup length
			wsize : window size
			wpsize : window processing step size
		"""
		self.maxAccRate = None
		self.confLevel = confLevel
		threshold = math.sqrt(0.5 * math.log(1 / confLevel) / wsize )
		super(FHDDM, self).__init__(threshold, warmUp, wsize, wpsize)

	@classmethod
	def create(cls, fpath):
		"""
		factory method to create based on restored model
	
		Parameters
			fpath : file path for saved model
		"""
		ws = restoreObject(fpath)
		obj = cls(ws["confLevel"], ws["warmUp"], ws["wsize"], ws["wpsize"])
		obj.warmedUp = ws["warmedUp"]
		obj.maxAccRate = ws["maxAccRate"]
		return obj
		
	def detect(self, evalues):
		"""
		detects drift in batch
	
		Parameters
			evalues : error value list
		"""
		warmed = False
		if not self.warmedUp:
			assertGreaterEqual(self.wsize, self.warmUp, "window size should be greater than or equal to warmup size")
			accCount = 0
			for i in range(self.warmUp):
				if evalues[i] == 0:
					accCount += 1
			self.maxAccRate = accCount / self.warmUp
			self.warmedUp = True
			warmed = True

		result = None
		beg = self.warmUp if  warmed else 0
		remain = len(evalues) - beg
		if remain > 20:
			accCount = 0
			for i in range(beg, len(evalues), 1):
				if (evalues[i] == 0):
					accCount += 1 
			accRate = accCount / remain
			if accRate > self.maxAccRate:
				self.maxAccRate = accRate
			dr = 1 if (self.maxAccRate - accRate) > self.threshold else 0
			result = (accRate, dr)

		return result

	def save(self, fpath):
		"""
		save FHDDM algorithm state
		
		Parameters
			fpath : file path for model checkpointing
		"""
		ws = dict()
		ws["confLevel"] = self.confLevel
		ws["warmUp"] = self.warmUp
		ws["warmedUp"] = self.warmedUp
		ws["wsize"] = self.wsize
		ws["wpsize"] = self.wpsize
		ws["maxAccRate"] = self.maxAccRate
		saveObject(ws, fpath)
			
	def restore(self, fpath):
		"""
		restore FHDDM algorithm state

		Parameters
			fpath : file path for model checkpointing
		"""
		ws = restoreObject(fpath)
		self.confLevel = ws["confLevel"]
		self.warmUp = ws["warmUp"]
		self.warmedUp = ws["warmedUp"]
		self.wsize = ws["wsize"]
		self.wpsize = ws["wpsize"]
		self.maxAccRate = ws["maxAccRate"]
		
class ECDD():
	"""
	EWMA for Concept Drift Detection (ECDD)
	"""
	def __init__(self, expf, fprate, warmUp):
		"""
		initializer
	
		Parameters
			expf : exponential factor
			fprate : false positive rate for drift
			warmUp : warmup length
		"""
		self.count = 0
		self.expf = expf
		self.fprate = fprate
		self.warmUp = warmUp
		self.cl = None
		self.pr = 0
		self.sd = 0
		self.sdz = 0
		self.z = 0

	def add(self, evalue):
		"""
		detects drift online
	
		Parameters
			evalue : error value
		"""
		self.count += 1
		t = self.count
		self.pr = self.pr * t / (t + 1) + evalue / (t + 1)
		self.sd = self.pr * (1.0 - self.pr)
		e = 1.0 - self.expf
		self.sdz = math.sqrt(self.sd * self.expf * (1.0 - e ** (2 * t)) / (2.0 - self.expf))
		self.z = (1 - self.expf) * self.z + self.expf * evalue
		
		res = None
		if self.count > self.warmUp:
			if self.fprate == 100:
				self.cl = 2.76 - 6.23 * self.pr + 18.12 * self.pr ** 3 - 312.45 * self.pr ** 5 + 1002.18 + self.pr ** 7
			elif self.fprate == 400:
				self.cl = 3.97 - 6.56 * self.pr + 48.73 * self.pr ** 3 - 330.13 * self.pr ** 5 + 848.18 + self.pr ** 7
			else:
				exitWithMsg("invalid false positive rate")	
			dr = 1 if self.z > self.pr + self.cl * self.sdz else 0
			res = (self.z, dr)
			
		return res
		
	def save(self, fpath):
		"""
		save ECDD algorithm state
		"""
		ws = dict()
		ws["warmUp"] = self.warmUp
		ws["count"] = self.count
		ws["pr"] = self.pr
		ws["expf"] = self.expf
		ws["z"] = self.z
		ws["fprate"] = self.fprate
		saveObject(ws, fpath)

	def restore(self, fpath):
		"""
		restore ECDD algorithm state
		"""
		ws = restoreObject(fpath)
		self.warmUp = ws["warmUp"]
		self.count = ws["count"]
		self.pr = ws["pr"]
		self.expf = ws["expf"]
		self.z = s["z"]
		self.fprate = ws["fprate"]
		

class MultiSupConceptDrift(object):
	"""
	supervised cpncept drift detection multi label classification 
	"""
	def __init__(self, labels, detype):
		"""
		initializer
	
		Parameters
			labels : list of label names
			detype : detector type
		"""
		self.labels = labels
		self.detype = detype
		self.detectors = dict()
	
	def create(self,  warmUp, wsize=300, wpsize=30, threshold=0.8, confLevel=0.2, expf=0.7, fprate=100):
		"""
		creates all  detectors for all labels
		
		Parameters
			warmUp : warmup length
			wsize : window size
			wpsize : window processing step size
			threshold : threshold
			confLevel : confidence level for FHDDM
			expf : exponential factor for ECDD
			fprate : false positive rate for drift for ECDD
		"""
		for la in self.labels:
			if self.detype == "ddm":
				self.detectors[la] = DDM(threshold, warmUp, wsize, wpsize)
			elif self.detype == "eddm":
				self.detectors[la] = EDDM(threshold, warmUp, wsize, wpsize)
			elif self.detype == "fhddm":
				self.detectors[la] = FHDDM(confLevel, warmUp, wsize, wpsize)
			elif self.detype == "ecdd":
				self.detectors[la] = ECDD(expf, fprate, warmUp)
			else:
				exitWithMsg("invalid drift detector type")
		
	
	def add(self, evalues):
		"""
		detects drift online for all labels
	
		Parameters
			evalues : error values for all labels
		"""
		res = dict()
		for k in evalues.keys():
			re = self.detectors[k].add(evalues[k])
			res[k] = re
		return res
		
	def save(self, fpath):
		"""
		save EDDM algorithm state
		
		Parameters
			fpath : file path for model checkpointing
		"""
		agfpath = fpath + "_aggr.mod"
		agdet = dict()
		agdet["labels"] = self.labels
		agdet["detype"] = self.detype
		saveObject(agdet, agfpath)
		for k in self.detectors.keys():
			dfpath = fpath + "_" + k + ".mod"
			self.detectors[k].save(dfpath)

	@classmethod
	def restore(cls, fpath):
		"""
		factory method to restore all detectors for all labels
	
		Parameters
			fpath : file path for saved model
		"""
		agfpath = fpath + "_aggr.mod"
		agdet = restoreObject(agfpath)
		labels = agdet["labels"]
		detype = agdet["detype"]
		mdet = MultiSupConceptDrift(labels, detype)
		for la in mdet.labels:
			dfpath = fpath + "_" + la + ".mod"
			if mdet.detype == "ddm":
				mdet.detectors[la] = DDM.create(dfpath)
			elif mdet.detype == "eddm":
				mdet.detectors[la] = EDDM.create(dfpath)
			elif mdet.detype == "fhddm":
				mdet.detectors[la] = FHDDM.create(dfpath)
			elif mdet.detype == "ecdd":
				mdet.detectors[la] = ECDD.create(dfpath)
			else:
				exitWithMsg("invalid drift detector type")
		return mdet

			
		