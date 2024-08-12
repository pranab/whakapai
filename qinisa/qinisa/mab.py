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
			
	def getAction(self):
		"""
		next play return selected action
		"""
		sact = None
		scmax = 0
		for act in self.actions:
			#print(str(act))
			if act.nplay == 0:
				sact = act
				self.logger.info("untried action found")
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
			self.logger.info("all actions not rewarded yet")
			sact = selectRandomFromList(self.actions)
			scmax = self.getActionScore(sact)
			
		self.actFinalize(sact, scmax)
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
		
		
	def getAction(self):
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
				
		self.actFinalize(sact, sc)
		re = (sact.name, sc)	
		return re
		

class ThompsonSampling(MultiArmBandit):
	"""
	thompson sampling multi arm bandit (ts)
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
		self.shapeFac = dict(map(lambda a : (a, [1,1]), actions))
		super(ThompsonSampling, self).__init__(actions, wsize, transientAction,logFilePath, logLevName, __name__, "ThompsonSampling")
		
		
	def getAction(self):
		"""
		next play return selected action
		"""
		sact = None
		sc = 0
		for act in self.actions:
			#any action not tried yet
			if act.nplay == 0:
				sact = act
				self.logger.info("untried action found")
				break
		
		if sact is None:
			#sample reward
			for act in self.actions:
				if act.isRewarded():
					sf = self.shapeFac[act.name]
					p = np.random.beta(sf[0], sf[1])
					if p > sc:
						sc = p
						sact = act
				else:
					sact = None
					sc = 0
					break
					
		if sact is None:
			#select random
			self.logger.info("all actions not rewarded yet")
			sact = selectRandomFromList(self.actions)

		self.actFinalize(sact, sc)
		re = (sact.name, sc)	
		return re

	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			aname : action name
			reward : reward value
		"""
		super().setReward(aname, reward)
		sf = self.shapeFac[aname]
		if random.random() < reward:
			sf[0] += 1
		else:
			sf[1] += 1
		self.logger.info("action {} beta distr factors {} {}".format(aname, sf[0], sf[1]))	
		
class ExponentialWeight(MultiArmBandit):
	"""
	exponential weight multi arm bandit (exp3)
	"""
	
	def __init__(self, actions, wsize, transientAction,logFilePath, logLevName, gama=0.2):
		"""
		initializer
		
		Parameters
			actions : action names
			wsize : reward window size
			transientAction ; if decision involves some tied up resource it should be set False
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
			gama : uniform distr factor
		"""
		super(ExponentialWeight, self).__init__(actions, wsize, transientAction,logFilePath, logLevName, __name__, "ExponentialWeight")
		assertWithinRange(gama, 0, 0.5, "gama should not be greater that 0.5")
		self.weights = list(map(lambda a : [a, 1.0], actions))
		self.gama = gama
		self.distr = None
		self.sampler = None
		self.__getActionDistr()

	def getAction(self):
		"""
		next play return selected action
		"""
		sact = self.getUntriedAction()
		sc = 0

		
		if sact is None:
			#sample reward
			aname = self.sampler.sample()
			self.logger.info("sampled action " + aname)
			sact = self.getActionByName(aname)
		
		self.actFinalize(sact, sc)
		re = (sact.name, sc)	
		return re

	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			aname : action name
			reward : reward value
		"""
		super().setReward(aname, reward)
		for d in self.distr:
			if d[0] == aname:
				rn = reward / d[1]
				break
				
		for w in self.weights:
			if w[0] == aname:
				w[1] *= math.exp(self.gama * rn / self.naction)
				break
		self.__getActionDistr()
	
	def __getActionDistr(self):
		"""
		action probability distribution
		"""	
		tw = 0
		for w in self.weights:
			tw += w[1]
		self.distr = list(map(lambda w : (w[0], (1.0 - self.gama) * w[1] / tw + self.gama / self.naction), self.weights))
		self.sampler = CategoricalRejectSampler(self.distr)

class SoftMix(MultiArmBandit):
	"""
	softmix multi arm bandit (smix)
	"""
	
	def __init__(self, actions, wsize, transientAction,logFilePath, logLevName, d=0.5):
		"""
		initializer
		
		Parameters
			actions : action names
			wsize : reward window size
			transientAction ; if decision involves some tied up resource it should be set False
			logFilePath : log file path set None for no logging
			logLevName : log level e.g. info, debug
			d : factor
		"""
		super(SoftMix, self).__init__(actions, wsize, transientAction,logFilePath, logLevName, __name__, "SoftMix")
		assertWithinRange(d, 0, 1.0, "d should be between 0 and 1.0")
		self.weights = list(map(lambda a : [a, 0], actions))
		self.d = d
		self.ds = d * d
		self.distr = None
		self.sampler = None		
		self.__getActionDistr()
		
	def getAction(self):
		"""
		next play return selected action
		"""
		sact = self.getUntriedAction()
		sc = 0
		
		if sact is None:
			aname = self.sampler.sample()
			self.logger.info("sampled action " + aname)
			sact = self.getActionByName(aname)
		
		self.actFinalize(sact, sc)
		re = (sact.name, sc)	
		return re	
	
	def setReward(self, aname, reward):
		"""
		reward feedback for action
			
		Parameters
			aname : action name
			reward : reward value
		"""
		super().setReward(aname, reward)
		for d in self.distr:
			if d[0] == aname:
				rn = reward / d[1]
				break	
		self.logger.info("weight correction {:.3f}".format(rn))			
		
		for w in self.weights:
			if w[0] == aname:
				w[1] += rn
				break
		self.logger.info("action weights " + str(self.weights))
		
		self.__getActionDistr()
									
	def __getActionDistr(self):
		"""
		action probability distribution
		"""					
		gama = min(1, 5 * self.naction * math.log(self.totPlays) / (self.ds * self.totPlays)) if self.totPlays > 1 else 1
		eta = math.log(1 + self.d * (self.naction / gama + 1) / (2 * self.naction / gama - self.ds)) / (self.naction / gama + 1)
		self.logger.info("gama {:.3f}  eta {:.3f}".format(gama, eta))
			
		mweights = list(map(lambda w : (w[0], w[1] * eta), self.weights))
		z = sum(list(map(lambda mw : math.exp(mw[1]), mweights)))
		self.distr = list(map(lambda w : (w[0], (1.0 - gama) * math.exp(w[1]) / z + gama / self.naction), mweights))
		self.logger.info("action distr " + str(self.distr))
		self.sampler = CategoricalRejectSampler(self.distr)			


class RandomGreedyPolicy:
	"""
	random greedy multi arm bandit policy (epsilon greedy)
	"""
	def __init__(self, states, actions, env, epsilon, qvalues=None, policy=None, redPolicy="linear", redParam=None, nonGreedyActions=None):
		"""
		initializer
		
		Parameters
			states : all states
			actions : all actions
			env : environment
			epsilon : random selection probability 
			qvalues : q values
			policy : list of state,action tuple
			redPolicy : epsilon reduction policy
			redParam : epsilon reduction parameter
			nonGreedyActions ; hints for non greedy actions 
		""" 
		self.epsilon = epsilon
		self.redPolicy = redPolicy
		self.redParam = redParam
		self.qvalues = None
		self.policy = None
		self.states = states
		self.actions = actions
		self.env = env
		self.nonGreedyActions = nonGreedyActions
		if qvalues is not None:
			#qvalues
			self.qvalues = qvalues
		elif policy is not None:
			#policy
			self.policy = policy
		else:
			exitWithMsg("either qvalues or policy needs to be provided")	
		self.totPlays = dict(map(lambda s : (s, 0), self.states))
		
	def getAction(self, state):
		"""
		next play return selected action
		
		Parameters
			state : state
		"""
		#greedy action
		if self.qvalues is not None:
			actions = self.qvalues[state]
			vmax = 0
			sact = None
			for a in actions:
				if a[1] > vmax:
					sact = a[0]
					vmax = a[1]
			if sact is None:
				sact = selectRandomFromList(actions)[0]
		else:
			sact = self.policy[state]
			
		tp = self.totPlays[state]
		eps = self.epsilon
		if tp > 0:
			if self.redPolicy == "stepred":
				eps = self.epsilon - tp * self.redParam
				eps = max(0, eps)
			elif self.redPolicy == "linear":
				redFact = 1.0 / tp  
				eps = self.epsilon * redFact
			elif self.redPolicy == "loglinear":
				redFact = math.log(tp) / tp
				eps = self.epsilon * redFact
			else:
				exitWithMsg("invalid epsilon reduction strategy")
			
		
		#random action
		if random.random() < eps:
			if self.policy is not None and self.nonGreedyActions is not None:
				sact = selectRandomFromList(self.nonGreedyActions[sact])
			else:
				sact = selectRandomFromList(self.env.getActionsForState(state))
		incrKeyedCounter(self.totPlays, state)		
		return sact

class BoltzmanPolicy:
	"""
	boltzman multi arm bandit policy (epsilon greedy)
	"""
	def __init__(self, states, actions, env, epsilon, qvalues, redPolicy="linear", redParam=None):
		"""
		initializer
		
		Parameters
			states : all states
			actions : all actions
			env : environment
			epsilon : random selection probability 
			qvalues : q values
			redPolicy : epsilon reduction policy
			redParam : epsilon reduction parameter
		""" 
		self.epsilon = epsilon
		self.redPolicy = redPolicy
		self.redParam = redParam
		self.qvalues = None
		self.states = states
		self.actions = actions
		self.env = env
		self.qvalues = qvalues
		self.totPlays = dict(map(lambda s : (s, 1), self.states))
		
	def getAction(self, state):
		"""
		next play return selected action
		
		Parameters
			state : state
		"""
		tp = self.totPlays[state]
		eps = self.epsilon
		if tp > 0:
			if self.redPolicy == "stepred":
				eps = self.epsilon - tp * self.redParam
				eps = max(0, eps)
			elif self.redPolicy == "linear":
				redFact = 1.0 / tp  
				eps = self.epsilon * redFact
			elif self.redPolicy == "loglinear":
				redFact = math.log(tp+1) / tp
				eps = self.epsilon * redFact
			else:
				exitWithMsg("invalid epsilon reduction strategy")
			
		
		#actions for the state
		actions = self.env.getActionsForState(state)
		
		#values and values distribution
		actValues = dict(self.qvalues[state])
		bvalues = list(map(lambda a : actValues[a] ,actions))
		bvalues = list(map(lambda v : math.exp(eps * v), bvalues))
		bvalues = norm(bvalues, 1)
		
		#sample action
		sampler = CategoricalRejectSampler(list(zip(actions, bvalues)))
		sact = sampler.sample()
		
		incrKeyedCounter(self.totPlays, state)		
		return sact

class UpperConfBoundPolicy:		
	"""
	upper confidence bound multi arm bandit policy (ucb)
	"""
	def __init__(self, env, qvalues):
		"""
		initializer
		
		Parameters
			env : environment
			qvalues : q values
		""" 
		#qvalues
		self.env = env
		self.qvalues = qvalues
		self.states = list(self.qvalues.keys())
		self.actions = list(map(lambda a : a[0], self.qvalues[self.states[0]]))
		
		self.totPlays = dict(map(lambda s : (s, 0), self.states))
		self.actPlays = dict()
		for s in self.states:
			self.actPlays[s] = dict(map(lambda a : (a, 0), self.actions))

	def getAction(self, state):
		"""
		next play return selected action
		
		Parameters
			state : state
		"""
		sact = None
		vmax = None

		#actions and action values for the state
		actions = self.env.getActionsForState(state)
		actValues = dict(self.qvalues[state])
		for ac in actions:
			#if first time
			if self.actPlays[state][ac] == 0:
				sact = ac
				break
					
			v = actValues[ac] + sqrt(2 * math.log(self.totPlays[state]) / self.actPlays[state][ac])
			if vmax is None or v > vmax:
				vmax = v
				sact = ac
		
		incrKeyedCounter(self.totPlays, state)
		incrKeyedCounter(self.actPlays[state], sact)
		return sact
