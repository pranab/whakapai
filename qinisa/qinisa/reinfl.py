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

class TempDifference:
	"""
	temporal difference TD(0)
	"""
	def __init__(self, policy, lrate, dfactor, istate, logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			policy : deterministic or probabilistic policy
			lrate : learning rate
			dfactor : discount factor
			istate : initial state
		"""
		self.states = policy.getStates()
		self.values = dict(list(map(lambda s : (s, 0), self.states)))
		self.lrate = lrate
		self.dfactor = dfactor
		self.state = istate
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(mname, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + "TempDifference")
		
	def getAction(self):
		"""
		get action for current state
		"""
		return policy.getAction(self.state)
	
	def setReward(self, reward, nstate):
		"""
		initializer
		
		Parameters
			rwarde : reward
			nstate : next state
		"""
		delta = self.lrate * (reward + self.dfactor * self.values[nstate] - self.values[self.state])
		self.values[self.state] += delta
		self.logger.info("state {}  incr value {:.3f}  cur value {:.3f}".format(self.state, delta, self.values[self.state]))
		self.state = nstate
	
	def getValues(self):
		"""
		return state values
		"""	
		return self.values
		