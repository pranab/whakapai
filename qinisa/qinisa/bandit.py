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

import os
import sys
import random 
import math
import numpy as np
from matumizi.util import *
from matumizi.mlutil import *

class Action(object):
	"""
	action class for  multi arm bandit 
	"""
	
	def __init__(self, name, wsize):
		"""
		initializer
		
		Parameters
			naction : no of actions
			rwindow : reward window size
		"""
		self.name = name
		self.available = True
		self.rwindow = RollingStat(wsize)
		self.nplay = 0
		self.nreward = 0
		self.treward = 0

	def makeAvailable(self, available):
		"""
		sets available flag
		
		Parameters
			available : available flag
		"""
		self.available = available
	
	def addReward(self, reward):
		"""
		add reward
		
		Parameters
			reward : reward value
		"""
		self.rwindow.add(reward)
		self.nreward += 1
		self.treward += reward	
		
	def getRewardStat(self):
		"""
		get average reward
		"""
		rs = self.rwindow.getStat() if self.rwindow.getSize() > 0 else None
		return rs
		
	def getRewardCount(self):
		"""
		get reward count
		"""
		return self.rwindow.getSize()
	
	def isRewarded(self):
		"""
		if rewarded return true
		"""
		return self.rwindow.getSize() > 0

	def __str__(self):
		"""
		content
		"""
		desc = "name {}  available {}  window size {}  no of play {}".format(self.name, self.available, self.rwindow.getSize(), self.nplay)
		return desc
	

class MultiArmBandit:
	"""
	multi arm bandit base
	"""
	
	def __init__(self, actions, wsize, transientAction,logFilePath, logLevName, mname, clname):
		"""
		initializer
		
		Parameters
			actions : action names
			wsize : reward window size
			transientAction ; if decision involves some tied up resource it should be set False
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
			mname : module name
			clname : class name
		"""
		assertGreater(wsize, 9, "window size should be at least 10")
		self.actions = list(map(lambda aname : Action(aname, wsize), actions))
		self .totPlays = 0
		self.transientAction = transientAction
		self.raction = None
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(mname, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + clname)
	
			
	def getAction(self):
		"""
		next play return selected action
		"""
		sact, scmax = self.getBestAction()
			
		if not self.transientAction:
			sact.makeAvailable(False)
		sact.nplay += 1
		self .totPlays += 1
		self.logger.info("action selected {}  score {}".format(str(sact), scmax))
		re = (sact.name, scmax)	
		return re
	
	def getActionScore(self, act):
		"""
		return action average reward 
		
		Parameters
			act : action 
		"""
		s = act.getRewardStat()
		sc = s[0] if s is not None else None
		return sc
		
	def getBestAction(self):
		"""
		return action with best average reward
		
		"""
		sact = None
		scmax = 0

		for act in self.actions:
			#any untried
			if act.nplay == 0:
				sact = act
				break
		
		if sact is None:
			for act in self.actions:
				if self.transientAction or act.available:
					if act.isRewarded():
						sc  = self.getActionScore(act)
					
						self.logger.info("action {}  plays total {}  this action {}".format(act.name, self.totPlays, act.nplay))
						if sc > scmax:
							scmax = sc
							sact = act
		r = (sact, scmax)
		return r
							
	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			act : action
			reward : reward value
		"""
		acts = list(filter(lambda act : act.name == aname, self.actions))
		assertEqual(len(acts), 1, "invalid action name")
		act = acts[0]
		self.raction = act
		act.addReward(reward)
		if not self.transientAction:
			act.makeAvailable(True)
		self.logger.info("action {}  reward {:.3f}".format(act, reward))

	def getRegret(self):
		"""
		gets regret
		"""
		#actual reward
		nreward = 0
		treward = 0
		avrmax = 0
		for act in self.actions:
			treward += act.treward
			nreward += act.nreward
			avreward = act.treward / act.nreward if act.nreward > 0 else 0
			if avreward > avrmax:
				avrmax = avreward
		
		avr =  treward / nreward
		return (avrmax, avr, avrmax - avr)
		
	@staticmethod
	def save(model, filePath):
		"""
		saves object
				
		Parameters
			model : model object
			filePath : file path
		"""
		saveObject(model, filePath)
			
	@staticmethod
	def restore(filePath):
		"""
		restores object
				
		Parameters
			filePath : file path
		"""
		model = restoreObject(filePath)
		return model
		
	def actFinalize(self, sact, sc):
		"""
		
		"""
		if not self.transientAction:
			sact.makeAvailable(False)
		sact.nplay += 1
		self.totPlays += 1
		self.logger.info("action selected {}  score {}".format(str(sact), sc))
		
	