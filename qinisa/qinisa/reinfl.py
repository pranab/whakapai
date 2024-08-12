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
	def __init__(self, states, actions, env, banditAlgo, banditParams, lrate, dfactor, istate, gstate=None, qvPath=None, 
	policy=None, onPolicy=False, invalidStateActiins=None, logFilePath=None, logLevName=None):
		"""
		initializer
		
		Parameters
			states : all states
			actions : all actions
			env : environment
			banditAlgo : bandit algo (rg, ucb)
			banditParams : bandit algo params
			lrate : learning rate
			dfactor : discount factor
			istate : initial state
			gstate : goa; state
			qvPath : state action  values file path
			policy : current policy (optional)
			onPolicy : True if on policy
			invalidStateActiins : list of invalid state action tuples
			logFilePath : log file path
			logLevName : log level
		"""
		self.states = states
		if qvPath is None:
			self.qvalues = dict()
			for s in states:
				avalues = list(map(lambda a : [a, randomFloat(.001, .002)], actions))
				self.qvalues[s] = avalues
			
			#in=valid state actions	
			if invalidStateActiins is not None:
				for (s,a) in invalidStateActiins:
					for i in range(len(self.qvalues[s])):
						if self.qvalues[s][i][0] == a:
							self.qvalues[s][i][1] = -sys.float_info.max
							break
				
		else:
			self.qvalues = restoreObject(qvPath)
		
		
		self.invalidStateActiins = invalidStateActiins
		
		if banditAlgo == "rg":
			#random greedy
			qvalues = self.qvalues if policy is None else None
			pol = policy if policy is not None else None
			self.policy = RandomGreedyPolicy(states, actions, env, banditParams["epsilon"], qvalues=qvalues, policy=pol, 
			redPolicy=banditParams["redPolicy"], redParam=banditParams["redParam"], nonGreedyActions=banditParams["nonGreedyActions"])
		elif banditAlgo == "boltz":
			#boltzman
			qvalues = self.qvalues 
			self.policy = BoltzmanPolicy(states, actions, env, banditParams["epsilon"], qvalues=qvalues, 
			redPolicy=banditParams["redPolicy"], redParam=banditParams["redParam"])
		elif banditAlgo == "ucb":
			#ucb
			qvalues = self.qvalues 
			self.policy = UpperConfBoundPolicy(env, qvalues)
		else:
			exitWithMsg("invalid bandit algo " + banditAlgo)
			
		self.lrate = lrate
		self.dfactor = dfactor
		self.state = istate
		self.istate = istate
		self.gstate = gstate
		self.action = None
		self.onPolicy = onPolicy
		self.qvalUpdates = list()
		
		self.logger = None
		if logFilePath is not None: 		
			self.logger = createLogger(__name__, logFilePath, logLevName)
			self.logger.info("******** stating new  session of " + "TempDifferenceControl")
		
	def getAction(self, env):
		"""
		get action for current state
			
		Parameters
			env : environment
		"""
		acCnt = 0
		while True:
			self.action =  self.policy.getAction(self.state)
			sa = (self.state, self.action)
			if self.invalidStateActiins is not None and sa in self.invalidStateActiins:
				acCnt += 1
				if acCnt == 100:
					exitWithMsg("failed to select valid actions after 100 tries")
				continue
			else:
				break

		#appendKeyedList(self.stateActions, self.state, self.action)
		env.track(self.state, self.action)
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
			nmv = None
			for a in self.qvalues[nstate]:
				if nmv is None or a[1] > nmv:
					nmv = a[1]

		#update Q value
		delta = self.lrate * (reward + self.dfactor * nmv - cv)
		qval = 0
		for a in self.qvalues[self.state]:
			if a[0] == self.action:
				a[1] += delta
				qval = a[1]
				break
		
		#qvalue update history
		self.qvalUpdates.append(delta)
		
		if self.logger is not None:
			self.logger.info("state {}  action {} incr value {:.3f}  cur qvalue {:.3f}".format(self.state, self.action, delta, qval))
		self.state = nstate
		
	def getPolicy(self, env):
		"""
		get policy from qvaluese
		
		Parameters
			env : environment
		"""
		#log Qtable
		if self.logger is not None:
			for st in self.states:
				self.logger.info("Qtable state {}".format(st))
				actions = self.qvalues[st]
				for ac, va in actions:
					va = 0.0 if va < -1000000 else va
					self.logger.info("action {} value {:.6f}".format(ac,va))
				
		policy = dict()
		policyPath = list()
		if self.gstate is None:
			#generic task
			for st in self.states:
				actions = self.qvalues[st]
				if self.logger is not None:
					self.logger.info("state {}   actions {}".format(st, str(actions)))
				vmax = None
				sact = None
				for a in actions:
					if vmax is None or a[1] > vmax:
						sact = a[0]
						vmax = a[1]
				policy[st] = sact
		else:
			#goal state based task
			states = list()
			st = self.istate
			states.append(st)
			stcnt = 0
			
			while st != self.gstate:
				actions = self.qvalues[st]
				sactions = sorted(actions, key=takeSecond, reverse=True)
				if self.logger is not None:
					self.logger.info("Qtable state {}".format(st))
					self.logger.info("sorted action {}  value {:.3f}".format(sactions[0][0], sactions[0][1]))
					self.logger.info("sorted action {}  value {:.3f}".format(sactions[1][0], sactions[1][1]))
				
				ac = sactions[0][0]
				policy[st] = ac
				sa = (st, ac)
				policyPath.append(sa)
				
				if self.logger is not None:
					self.logger.info("policy state {}  action {}".format(st,ac))
				st = env.getNextState(st, ac)
				
				stcnt += 1
				if stcnt == 100:
					if self.logger is not None:
						self.logger.info("policy:")
						for sa in policyPath:
							self.logger.info("state action {}".format(str(sa)))
					exitWithMsg("failed to find policy with goal state defined")
					
		return policy

	def train(self, niter, env):	
		"""
		train model
		
		Parameters
			niter : num of iterations
			env : environment
		"""
		for i in range(niter):
			ac = self.getAction(env)
			nst, re = env.getReward(self.state, self.action)
			self.setReward(re, nst)
			
			#back to intial state if goal state vreached
			if self.gstate is not None and  nst == self.gstate:
				self.state = istate
				if self.logger is not None:
					self.logger.info("reset to intial state")
		
	def getQvalUpdates(self):
		"""
		return qvalue update history
		
		"""
		return self.qvalUpdates
	
	def save(self, fpath):
		"""
		saves object
				
		Parameters
			fpath : file path
		"""
		saveObject(self.qvalues, fpath)
		
class DynaQvalue(TempDifferenceControl):
	"""
	Dyna Q
	"""
	
	def __init__(self, states, actions, env, banditAlgo, banditParams, lrate, dfactor, istate, model, gstate=None, qvPath=None, 
	invalidStateActiins=None, logFilePath=None, logLevName=None):
		"""
		initializer
		
		Parameters
			states : all states
			actions : all actions
			env : environment
			banditAlgo : bandit algo (rg, ucb)
			banditParams : bandit algo params
			lrate : learning rate
			dfactor : discount factor
			istate : initial state
			model : environment model
			qvPath : state action  values file path
			policy : current policy (optional)
			onPolicy : True if on policy
			invalidStateActiins : list of invalid state action tuples
			logFilePath : log file path
			logLevName : log level
		"""
		super(DynaQvalue, self).__init__(states, actions, env, banditAlgo, banditParams, lrate, dfactor, istate, gstate, qvPath=qvPath, 
		invalidStateActiins=invalidStateActiins, logFilePath=logFilePath, logLevName=logLevName)	
		self.model = model

	def setReward(self, reward, nstate):
		"""
		sets reward
		
		Parameters
			rwarde : reward
			nstate : next state
		"""
		self.model.train(self.state, self.action, nstate, reward)
		super().setReward(reward, nstate)
		
	def simulate(self, env):
		"""
		initializer
		
		Parameters
			env : environment
		"""
		#some state visited earlier
		st = selectRandomFromList(env.statesVisited())
		if self.gstate is not None:
			while st == self.gstate:
				st = selectRandomFromList(env.statesVisited())
		
		#some action from that state
		ac = selectRandomFromList(env.actionsForVisitedState(st))
		
		#next state and reward
		ns, re = self.model.predict(st, ac)
		
		#current q value
		cv = None
		for a in self.qvalues[st]:
			if a[0] == ac:
				cv = a[1]
				break

		#off policy with action for max q value
		nmv = None	
		for a in self.qvalues[ns]:
			if nmv is None or a[1] > nmv:
				nmv = a[1]
		
		#update
		delta = self.lrate * (re + self.dfactor * nmv - cv)
		for a in self.qvalues[st]:
			if a[0] == ac:
				a[1] += delta
				qval = a[1]
				break
		
		"""
		if self.logger is not None:
			self.logger.info("model simulation state {}  action {} incr value {:.3f}  cur qvalue {:.3f}".format(st, ac, delta, qval))
		"""
		
	def train(self, niter, siter, env):	
		"""
		train model
		
		Parameters
			niter : num of iterations
			siter : num of simulation iteration
			env : environment
		"""
		#iteration count for boot strapping
		biter = int(.1 * niter)
		
		for i in range(niter):
			ac = self.getAction(env)
			nst, re = env.getReward(self.state, self.action)
			self.setReward(re, nst)
			
			#back to intial state if goal state vreached
			if self.gstate is not None and nst == self.gstate:
				self.state = self.istate
				if self.logger is not None:
					self.logger.info("reset to intial state")
				
			
			#model based simulation
			if i > biter:
				for j in range(siter):
					self.simulate(env)
			
