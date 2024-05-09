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
from matumizi.stats import *

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
		self.naction = len(actions)
		self.totPlays = 0
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
		self.totPlays += 1
		self.logger.info("action selected {}  score {}".format(str(sact), scmax))
		re = (sact.name, scmax)	
		return re
	
	def getUntriedAction(self):
		"""
		next untried action
		"""
		sact = None
		for act in self.actions:
			#any action not tried yet
			if act.nplay == 0:
				sact = act
				self.logger.info("untried action found")
				break
		return sact
		
	def getActionByName(self, aname):
		"""
		get action by name
		
		Parameters
			aname : action name
		"""
		sact = None
		for act in self.actions:
			if act.name == aname:
				sact = act
				break
		return sact
		
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
		
class Policy:
	"""
	deterministicor probabilistic policy
	"""
	def __init__(self, deterministic,  *stateActions):
		"""
		initializer
		
		Parameters
			deterministic : True if policy deterministic
			stateActions : state and action tuple list
		"""
		self.deterministic = deterministic
		if (len(stateActions) == 1):
			self.stateActions = dict(self.stateActions[0])
		else:
			self.stateActions = dict(stateActions)
				
	def getAction(self, state):
		"""
		get action
		
		Parameters
			state : state
		"""
		if state in self.stateActions:
			action = self.stateActions[state] if self.deterministic else self.stateActions[state].sample()
		else:
			exitWithMessage("invalid state " + state)
		return action
		
	def getStates(self):
		"""
		get all states
		"""
		return self.stateActions.keys()

class Environment:
	"""
	Environment base class
	"""
	def __init__(self, trackStates=False, trackActions=False, implReward=None, implRewardFactor=None):
		"""l
		initializer
		
		Parameters
			trackStates :True if states vneed tracking
			trackActions : Tri=ue if actions for each state need tracking
			implReward : implicit reward, bayesian exploration bonus (beb or beblog), recency(rec) or default (None)
			implRewardFactor : implicit reward factor
		"""
		self.trackStates = trackStates
		self.trackActions = trackActions
		self.implReward = implReward
		self.implRewardFactor = implRewardFactor
		self.visitedStates = list()
		self.stateActions = dict()
		self.iterCnt = 0
		self.stateActionLast = dict()
		
	def getReward(self, state, action):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
		"""
		return None
	
	def track(self, state, action):
		"""
		track state and action
		
		Parameters
			state : next state
			action : action
		"""
		if self.trackActions:
			appendKeyedList(self.stateActions, state, action)
		if self.trackStates:
			self.visitedStates.append(state)
		
		self.iterCnt += 1	
		
		
	def actionsForVisitedState(self, state):	
		"""
		returns actions for a state
		
		Parameters
			state : next state
		"""
		return self.stateActions[state]  if self.trackActions else None
		
	def statesVisited(self):
		"""
		returns states visited
		
		"""
		return self.visitedStates if self.trackStates else None
	
	def implicitReward(self, state, action):
		"""
		returns implicit reward
		
		Parameters
			state : next state
			action : action
		"""
		imptReward = 0
		if self.implReward is not None:
			if self.implReward == "beb":
				acnt = self.stateActions[state].count(action)
				imptReward =  1 / acnt if acnt > 0 else 1
			
			if self.implReward == "beblog":
				acnt = self.stateActions[state].count(action)
				imptReward =  1 / (1 + math.log(acnt)) if acnt > 1 else 1

			elif self.implReward == "rec":
				k = (state,action)
				lastIt = self.stateActionLast[k] if k in self.stateActionLast  else 0
				imptReward = math.sqrt(self.iterCnt - lastIt) / math.sqrt(self.iterCnt)
				self.stateActionLast[k] = self.iterCnt
			
			else:
				exitWithMsg("invalid implicit reward function " + self.implReward)	
			
			imptReward *= self.implRewardFactor
		
		return imptReward
		
class EnvModel:
	"""
	Environment model base class
	"""
	def __init__(self):
		"""
		initializer
		
		"""
		pass
		
	def train(self, state, action, nstate, reward):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
			nstate : next state
			reward : reward
		"""
		pass

	def predict(self, state, action):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
		"""
		pass
		
		
class DetEnvModel(EnvModel):
	"""
	detrministic environment model 
	"""
	def __init__(self):
		"""
		initializer
		
		"""
		self.model = dict()
		super(DetEnvModel, self).__init__()
		
	def train(self, state, action, nstate, reward):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
			nstate : next state
			reward : reward
		"""
		k = (state,action)
		if k not in self.model:
			v = (nstate, reward)
			self.model[k] = v
			
	def predict(self, state, action):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
		"""
		k = (state,action)
		return self.model[k]

class StochEnvModel(EnvModel):
	"""
	stochastic environment model 
	"""
	def __init__(self, sampUn, nbins=None):
		"""
		initializer
		
		Parameters
			sampleUn :True if next state and reward to be sampled uniformly
			nbins : num of bins for distrinution based sampling of reward
		"""
		self.smodel = dict()
		self.rmodel = dict()
		self.needUpdate = dict()
		self.uniqueStates = dict()
		self.rewardRange = dict()
		self.stateDistr = dict()
		self.rewardDistr = dict()
		self.sampUn = sampUn
		self.nbins = nbins
		if not sampUn:
			assertNotNone(nbins, "missing num of bins for distrinution based sampling of reward")
		super(StochEnvModel, self).__init__()
		
	def train(self, state, action, nstate, reward):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
			nstate : next state
			reward : reward
		"""
		k = (state,action)
		appendKeyedList(self.smodel, k, nstate)
		appendKeyedList(self.rmodel, k, reward)
		self.needUpdate[k] = True
	
	def predict(self, state, action):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
		"""
		k = (state,action)
		if self.sampUn:
			#uniform sampling
			if self.needUpdate[k]:
				#states
				self.uniqueStates[k] = list(set(self.smodel[k]))
				
				#rewards
				rmin = min(self.rmodel[k])
				rmax = max(self.rmodel[k])
				rrange = (rmion, rmax)
				self.rewardRange[k] = rrange
				self.needUpdate[k] = False
			
			nstate = selectRandomFromList(self.uniqueStates[k])
			reward = randomFloat(self.rewardRange[k][0], self.rewardRange[k][1])
			
		else:
			#distribution based sampling
			if self.needUpdate[k]:
				#states
				stCounter = toKeyedCount(self.smodel[k])
				stCounter = list(stCounter.items())
				stSampler = CategoricalRejectSampler(stCounter)
				self.stateDistr[k] = stSampler
				
				#rewards
				hgram = Histogram.createWithNumBins(self.rmodel[k], self.nbins)
				bvalues = hgram.distr()
				bwidth = hgram.getBinWidth()
				rmin, rmax = hgram.getMinMax()
				reSampler = NonParamRejectSampler(rmin, bwidth, bvalues)
				self.rewardDistr[k] = reSampler
				self.needUpdate[k] = False
		
			nstate = self.stateDistr[k].sample()
			reward = self.rewardDistr[k].sample()
			
		return (nstate, reward)
		
		

