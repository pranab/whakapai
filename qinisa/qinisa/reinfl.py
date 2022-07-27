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
from .rlba import *

class TempDifferenceValue:
	"""
	temporal difference TD(0) learning
	"""
	def __init__(self, policy, lrate, lrdecay, dfactor, istate, logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			policy : deterministic or probabilistic policy
			lrate : learning rate
			lrdecay ; learning rate decay
			dfactor : discount factor
			istate : initial state
		"""
		self.policy = policy
		self.states = policy.getStates()
		self.values = dict(list(map(lambda s : (s, 0), self.states)))
		self.lrate = lrate
		self.lrdecay = lrdecay
		self.dfactor = dfactor
		self.state = istate
		self.count = 0
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("\n******** stating new  session of " + "TempDifferenceValue")
		
	def getAction(self):
		"""
		get action for current state
		"""
		act =  self.policy.getAction(self.state)
		if self.logger is not None:
			self.logger.info("state {}  action {}".format(self.state, act))
		return act
		
	def setReward(self, reward, nstate):
		"""
		initializer
		
		Parameters
			rwarde : reward
			nstate : next state
		"""
		lrate = self.lrate / (1 + self.count * self.lrdecay)
		delta = lrate * (reward + self.dfactor * self.values[nstate] - self.values[self.state])
		self.values[self.state] += delta
		if self.logger is not None:
			self.logger.info("state {}  incr value {:.3f}  cur value {:.3f} reward {:.3f}  new state {} ".
		format(self.state, delta, self.values[self.state], reward, nstate))
		self.state = nstate
		self.count += 1
	
	def getValues(self):
		"""
		return state values
		"""	
		return self.values
		
	def getTotValue(self):
		"""
		return total value
		"""	
		tval = 0
		for k in self.values.keys():
			tval += self.values[k]
		return tval
			
		
class TempDifferenceControl:
	"""
	temporal difference control Q learning
	"""
	def __init__(self, states, actions, banditAlgo, banditParams, lrate, dfactor, istate, logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			banditAlgo : bandit algo (rg, ucb)
			banditParams : bandit algo params
			lrate : learning rate
			dfactor : discount factor
			istate : initial state
		"""
		self.states = states
		avalues = list(map(lambda a : [a, 0], actions))
		self.qvalues = dict(list(map(lambda s : (s, avalues.copy()), states)))
		if banditAlgo == "rg":
			self.policy = RandomGreedyPolicy(qvalues, banditParams["epsilon"], banditParams["redPolicy"])
		elif banditAlgo == "ucb":
			self.policy = UpperConfBoundPolicy(qvalues)
		else:
			exitWithMsg("invalid bandit algo")
			
		self.lrate = lrate
		self.dfactor = dfactor
		self.state = istate
		self.action = None
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + "TempDifferenceControl")
		
	def getAction(self):
		"""
		get action for current state
		"""
		self.action =  self.policy.getAction(self.state)
		return self.action

	def setReward(self, reward, nstate):
		"""
		initializer
		
		Parameters
			rwarde : reward
			nstate : next state
		"""
		cv = None
		for a in self.qvalues[self.state]:
			if a[0] == self.action:
				cv = a[1]
		nmv = 0	
		for a in self.qvalues[nstate]:
			if a[1] > nmv:
				nmv = a[1]

		delta = self.lrate * (reward + self.dfactor * nmv - cv)
		for a in self.qvalues[self.state]:
			if a[0] == self.action:
				a[1] += delta
		self.logger.info("state {}  action {} incr value {:.3f}  cur value {:.3f}".format(self.state, self.action, delta, self.values[self.state]))
		self.state = nstate
