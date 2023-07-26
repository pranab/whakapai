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

import os
import sys
import random 
import math
import numpy as np
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *

"""
Contextual multi arm bandit
"""
class LinUpperConfBound(object):
	"""
	linear upper conf bound multi arm bandit (lin ucb1)
	"""
	
	def __init__(self, actions, nfeat, horizon, totPlays=0, reg=None, pthresh=None, a=None, b=None, alpha=None, logFilePath=None, logLevName=None):
		"""
		initializer
		
		Parameters
			actions : action names
			nfeat : feature size
			reg ; regularizing param (lambda)
			horizon : num of plays
			pthresh: probability threshold (delta)
			a : regression matrix
			b : regression vector
			alpha : constant
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
		"""
		self.actions = actions
		self.naction = len(actions)
		self.totPlays = totPlays
		self.nfeat = nfeat
		self.reg = reg
		self.horizon = horizon
		self.pthresh = pthresh
		self.a = np.identity(nfeat) * reg if a is None else a
		self.b = np.zeros(nfeat) if b is None else b
		self.horizon = horizon
		self.alpha = 1 + sqrt(0.5 * math.log(2 * horizon * self.naction / pthresh)) if alpha is None else alpha
		self.sactions = dict()
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + "LinUpperConfBound")
		
	def getAction(self, features):
		"""
		next play return selected action

		Parameters
			features : features for all actions with 1 row per action
		"""
		inva = np.linalg.inv(self.a)
		theta = np.matmul(inva, self.b)
		
		sact = None
		smax = None
		saf = None
		for i in range(np.shape(features)[0]):
			af = features[i]
			af = np.array(af)
			aft = np.transpose(af)
			t = np.matmul(aft, inva)
			t = np.matmul(t, af)
			s = np.dot(aft, theta) + self.alpha * sqrt(t)
			
			if smax is None or s > smax:
				smax = s
				sact = self.actions[i]
				saf = af	
				
		self.sactions[sact] = saf
		if self.logger is not None:
			self.logger.info("play count {}  action {}  reward upper bound {:.3f}".format(self.totPlays + 1, sact,  smax))

		self.totPlays += 1
		return sact
		

	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			act : action
			reward : reward value
		"""
		af = self.sactions[aname]
		aft = np.transpose(af)
		self.a = np.add(self.a, np.matmul(af, aft))
		self.b = np.add(self.b, af * reward)
		if self.logger is not None:
			self.logger.info("action {}  feature {}  actual reward {:.3f}".format(aname, floatArrayToString(af, delem=None), reward))
		self.sactions.pop(aname)
	
	def save(self, fpath):
		"""
		saves object
				
		Parameters
			fpath : file path
		"""
		mod = dict()
		mod["actions"] = self.actions
		mod["nfeat"] = self.nfeat
		mod["horizon"] = self.horizon
		mod["a"] = self.a
		mod["b"] = self.b
		mod["alpha"] = self.alpha
		mod["totPlays"] = self.totPlays
		saveObject(mod, fpath)
		
		if self.logger is not None:
			self.logger.info("cherckpointed model")
		
	@staticmethod
	def create(fpath, logFilePath=None, logLevName=None):
		"""
		restores object
				
		Parameters
			fpath : file path
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
		"""
		mod = restoreObject(fpath)
		actions = mod["actions"]
		nfeat = mod["nfeat"]
		horizon = mod["horizon"]
		a = mod["a"]
		b = mod["b"]
		alpha = mod["alpha"]
		totPlays = mod["totPlays"]
		linUcb = LinUpperConfBound(actions, nfeat, horizon, totPlays=totPlays, a=a, b=b, alpha=alpha, logFilePath=logFilePath, logLevName=logLevName)
		
		if linUcb.logger is not None:
			linUcb.logger.info("restored model from cherckpoint")
		return linUcb

class LinThompsonSampling(object):
	"""
	linear thompson sampling multi arm bandit (lin ts)
	"""
	
	def __init__(self, actions, nfeat, transientAction, subgaus, horizon, eps, pthresh, logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			actions : action names
			nfeat : feature size
			transientAction ; if decision involves some tied up resource it should be set False
			subgaus ; sub gaussian (R)
			horizon : num of plays
			eps : parameter (epsilon)
			pthresh: probability threshold (delta)
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
		"""
		self.actions = actions
		self.naction = len(actions)
		self.totPlays = 0
		self.transientAction = transientAction
		self.nfeat = nfeat
		self.subgaus = subgaus
		self.horizon = horizon
		self.eps = eps
		self.pthresh = pthresh
		self.b = np.identity(nfeat)
		self.mean = np.zeros(nfeat)
		self.f = np.zeros(nfeat)
		
		self.sactions = dict()
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + clname)
			
	def getAction(self, features):
		"""
		next play return selected action

		Parameters
			features : features for all actions with 1 row per action
		"""
		self.totPlays += 1
		v = self.subgaus *  sqrt(24 * self.nfeat * math.log(self.totPlays  / pthresh) / self.eps)
		v2 = v * v
		invb = np.linalg.inv(self.b)
		sd = invb * v2
		mu = np.random.multivariate_normal(self.mean, sd)
		sact = None
		smax = None
		saf = None
		for i in range(np.shape(features)[0]):
			af = features[i]
			af = np.array(af)
			aft = np.transpose(af)
			s = np.dot(aft, mu)
			
			if smax is None or s > smax:
				smax = s
				sact = self.actions[i]
				saf = af	

		self.sactions[sact] = saf
		return sact

	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			aname : action name
			reward : reward value
		"""
		af = self.sactions[aname]
		taf = np.transpose(af)
		self.b = np.add(self.b, np.matmul(taf, af))
		self.f = np.add(self.f, taf * reward)
		invb = np.linalg.inv(self.b)
		self.mean = np.matmul(invb, f)
		if self.logger is not None:
			self.logger.info("action {}  feature {}  reward {:.3f}".format(aname, floatArrayToString(af, delem=None), reward))
		self.sactions.pop(aname)
		
		
		