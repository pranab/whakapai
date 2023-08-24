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
import statistics
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from .rlba import *
from .mab import *

class TempDifferenceValue:
	"""
	temporal difference TD(0) learning for policy evaluation
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
			logFilePath : log file path
			logLevName : log level
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
		
	def setReward(self, reward, nstate, terminal=False):
		"""
		initializer
		
		Parameters
			reward : reward
			nstate : next state
			terminal : true if terninal state
		"""
		if not terminal:
			lrate = self.lrate / (1 + self.count * self.lrdecay)
			delta = lrate * (reward + self.dfactor * self.values[nstate] - self.values[self.state])
			self.values[self.state] += delta
			if self.logger is not None:
				self.logger.info("state {}  incr value {:.3f}  cur value {:.3f} reward {:.3f}  new state {} ".
			format(self.state, delta, self.values[self.state], reward, nstate))
			self.count += 1
		else:
			self.values[self.state] = 0

		self.state = nstate
		
	
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

class FirstVisitMonteCarlo:
	"""
	first visit Monte Carlo learning for policy evaluation
	"""
	def __init__(self, policy, dfactor, istate, logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			policy : deterministic or probabilistic policy
			dfactor : discount factor
			istate : initial state
			logFilePath : log file path
			logLevName : log level
		"""
		self.policy = policy
		self.states = policy.getStates()
		self.values = dict(list(map(lambda s : (s, 0), self.states)))
		self.valist = dict()
		self.rewards = list()
		self.dfactor = dfactor
		self.state = istate
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("\n******** stating new  session of " + "FirstVisitMonteCarlo")


	def getAction(self):
		"""
		get action for current state
		"""
		act =  self.policy.getAction(self.state)
		if self.logger is not None:
			self.logger.info("state {}  action {}".format(self.state, act))
		return act

	def setReward(self, reward, nstate, terminal=False):
		"""
		initializer
		
		Parameters
			reward : reward
			nstate : next state
			terminal : true if terninal state
		"""
		if not terminal:
			re = (self.state, reward)
			self.rewards.append(re)
		self.state = nstate
		
	def endEpisode(self):
		"""
		process episode end
		"""
		le = len(self.rewards)
		for i in range(le):
			rs = 0
			dfactor = 1
			for j in range(i, le, 1):
				#accumulate reward
				rs += dfactor * self.rewards[j][1]
				dfactor *= self.dfactor
			if self.logger is not None:
				self.logger.info("state {}  accumulated reward {:.3f}".format(self.rewards[i][0], rs)) 
			appendKeyedList(self.valist, self.rewards[i][0], rs)
		self.rewards.clear()
				
	def endIter(self):
		"""
		process iteration end
		"""
		for k in self.valist.keys():
			#average values from multiple episodes
			mv = statistics.mean(self.valist[k])
			self.values[k] = mv
	
	def getValues(self):
		"""
		return state values
		"""	
		return self.values
		
class PolicyImprovement:
	"""
	compares 2 policies by using actions from the second policy
	"""
	def __init__(self, policyOne, policyTwo, svaluesOne, svaluesTwo, dfactor,  logFilePath, logLevName):
		"""
		initializer
		
		Parameters
			policyOne : deterministic or probabilistic policy
			policyTwo : deterministic or probabilistic policy
			svaluesOne : state vaues for policyOne
			svaluesTwo : state vaues for policyTwo
			dfactor : discount factor
			logFilePath : log file path
			logLevName : log level
		"""
		self.policyOne = policyOne
		self.states = policyOne.getStates()
		self.policyTwo = policyTwo
		self.svaluesOne = svaluesOne
		self.svaluesTwo = svaluesTwo
		self.dfactor = dfactor
		self.values = list()
		self.svalues = dict()
		self.state = None

		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("\n******** stating new  session of " + "PolicyImprovement")
		

	def startStateIter(self, state):
		"""
		starts value iteration for value expectation using second policy action
		
		Parameters
			state : state
		"""	
		if self.state is not None:
			mvalue = statistics.mean(self.values)
			self.svalues[self.state] = mvalue
			self.values.clear()
			if self.logger is not None:
				self.logger.info("state {}  value {:.3f}".format(self.state, mvalue))
		self.state = state
	
	def endStateIter(self):
		"""
		ends value iteration for value expectation using second policy action
		"""	
		mvalue = statistics.mean(self.values)
		self.svalues[self.state] = mvalue
		self.values.clear()
		if self.logger is not None:
			self.logger.info("state {}  value {:.3f}".format(self.state, mvalue))

	def getAction(self):
		"""
		get action for current state from second policy
		"""
		act =  self.policyTwo.getAction(self.state)
		return act

	def setReward(self, reward, nstate):
		"""
		initializer
		
		Parameters
			reward : reward
			nstate : next state
		"""
		self.values.append(reward + self.dfactor * self.svaluesOne[nstate])


	def compare(self, states=None):
		"""
		compare first policy state values with  values based on second policy 
		"""
		re = dict() 
		states = self.states if states is None else states
		for st in states:
			nv = self.svalues[st]
			ev = self.svaluesOne[st]
			gr = 1 if nv >= ev else 0
			re[st] = (nv, ev, gr)

		return re

	
class TempDifferenceControl:
	"""
	temporal difference control Q learning
	"""
	def __init__(self, states, actions, banditAlgo, banditParams, lrate, dfactor, istate, qvPath=None, policy=None, onPolicy=False, logFilePath=None, logLevName=None):
		"""
		initializer
		
		Parameters
			states : all states
			actions : all actions
			banditAlgo : bandit algo (rg, ucb)
			banditParams : bandit algo params
			lrate : learning rate
			dfactor : discount factor
			istate : initial state
			qvPath : state action  values file path
			policy : current policy (optional)
			onPolicy : True if on policy
			logFilePath : log file path
			logLevName : log level
		"""
		self.states = states
		if qvPath is None:
			self.qvalues = dict()
			for s in states:
				avalues = list(map(lambda a : [a, randomFloat(.01, .10)], actions))
				self.qvalues[s] = avalues
		else:
			self.qvalues = restoreObject(qvPath)
		
		if banditAlgo == "rg":
			#random greedy
			qvalues = self.qvalues if policy is None else None
			pol = policy if policy is not None else None
			self.policy = RandomGreedyPolicy(states, actions, banditParams["epsilon"], qvalues=qvalues, policy=pol, 
			redPolicy=banditParams["redPolicy"], redParam=banditParams["redParam"], nonGreedyActions=banditParams["nonGreedyActions"])
		if banditAlgo == "boltz":
			#boltzman
			qvalues = self.qvalues 
			self.policy = BoltzmanPolicy(states, actions, banditParams["epsilon"], qvalues=qvalues, 
			redPolicy=banditParams["redPolicy"], redParam=banditParams["redParam"])
		elif banditAlgo == "ucb":
			#ucb
			self.policy = UpperConfBoundPolicy(qvalues)
		else:
			exitWithMsg("invalid bandit algo")
			
		self.lrate = lrate
		self.dfactor = dfactor
		self.state = istate
		self.action = None
		self.onPolicy = onPolicy
		
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
		sets reward
		
		Parameters
			rwarde : reward
			nstate : next state
		"""
		#current q value
		cv = None
		for a in self.qvalues[self.state]:
			if a[0] == self.action:
				cv = a[1]
				break
		
		
		if self.onPolicy:
			#on policy with action per policy
			naction = self.policy.getAction(nstate)
			nmv = 0	
			for a in self.qvalues[nstate]:
				if a[0] == naction:
					nmv = a[1]
					break
		else:
			#off policy with action for max q value
			nmv = 0	
			for a in self.qvalues[nstate]:
				if a[1] > nmv:
					nmv = a[1]

		delta = self.lrate * (reward + self.dfactor * nmv - cv)
		qval = 0
		for a in self.qvalues[self.state]:
			if a[0] == self.action:
				a[1] += delta
				qval = a[1]
				break
		self.logger.info("state {}  action {} incr value {:.3f}  cur qvalue {:.3f}".format(self.state, self.action, delta, qval))
		self.state = nstate
		
	def getPolicy(self):
		"""
		get policy from qvaluese
		"""
		policy = dict()
		for st in self.states:
			actions = self.qvalues[st]
			self.logger.info("state {}   actions {}".format(st, str(actions)))
			vmax = 0
			sact = None
			for a in actions:
				if a[1] > vmax:
					sact = a[0]
					vmax = a[1]
			policy[st] = sact
		
		return policy
		
	def save(self, fpath):
		"""
		saves object
				
		Parameters
			fpath : file path
		"""
		saveObject(self.qvalues, fpath)
