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
from matumizi.sampler import *
from .bandit import *

class UpperConfBound(MultiArmBandit):
	"""
	upper conf bound multi arm bandit (ucb1)
	"""
	
	def __init__(self, actions, wsize, transientAction,logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			actions : action names
			wsize : reward window size
			transientAction ; if decision involves some tied up resource it should be set False
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
		"""
		self.tuned = False
		super(UpperConfBound, self).__init__(actions, wsize, transientAction,logFilePath, logLevName, __name__, "UpperConfBound")

	def useTuned(self):
		"""
		set to use tuned UCB model
		"""
		self.tuned =  True
			
	def act(self):
		"""
		next play return selected action
		"""
		sact = None
		scmax = 0
		for act in self.actions:
			#print(str(act))
			if act.nplay == 0:
				sact = act
				break
			else:		
				if self.transientAction or act.available:
					if act.isRewarded():
						s1 = act.getRewardStat()
						if self.tuned:
							v = s1[1] * s1[1] + sqrt(2 * math.log(self.totPlays) / act.nplay)
							s2 = sqrt(math.log(self .totPlays) / act.nplay) + min(.25, v)
						else:
							s2 = sqrt(2 * math.log(self.totPlays) / act.nplay)
						sc  = s1[0] + s2
					
						self.logger.info("action {}  plays total {}  this action {}".format(act.name, self.totPlays, act.nplay))
						self.logger.info("ucb score components {:.3f}  {:.3f}".format(s1[0],s2))
						if sc > scmax:
							scmax = sc
							sact = act
		if sact is None:
			sact = selectRandomFromList(self.actions)
			scmax = self.getActionScore(sact)
			
		if not self.transientAction:
			sact.makeAvailable(False)
		sact.nplay += 1
		self .totPlays += 1
		self.logger.info("action selected {}  score {}".format(str(sact), scmax))
		re = (sact.name, scmax)	
		return re
			
		
		
class RandomGreedy(MultiArmBandit):
	"""
	random greedy multi arm bandit (epsilon greedy)
	"""
	
	def __init__(self, actions, wsize, transientAction,logFilePath, logLevName, epsilon, redPolicy="linear"):
		"""
		initializer
		
		Parameters
			actions : action names
			wsize : reward window size
			transientAction ; if decision involves some tied up resource it should be set False
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
			epsilon : random selection probability
			redPolicy : reduction policy for epsilon
		"""
		self.epsilon = epsilon
		self.redPolicy = redPolicy
		super(RandomGreedy, self).__init__(actions, wsize, transientAction,logFilePath, logLevName, __name__, "RandomGreedy")	
		
		
	def act(self):
		"""
		next play return selected action
		"""
		sact, sc = self.getBestAction()
		if sact is None or sact.nplay > 0: 
			if sact is not None:
				redFact = 1.0 / self.totPlays if self.redPolicy == "linear" else math.log(self.totPlays) / self.totPlays
				eps = self.epsilon * redFact
			if sact is None or random.random() < eps:
				sact = selectRandomFromList(self.actions)
				sc = self.getActionScore(sact)
				
		if not self.transientAction:
			sact.makeAvailable(False)
		sact.nplay += 1
		self.totPlays += 1
		self.logger.info("action selected {}  score {}".format(str(sact), sc))
		re = (sact.name, sc)	
		return re
		
	