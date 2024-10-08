#!/usr/local/bin/python3

# qinisa : Machine Learning
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
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from qinisa.reinfl import *
from qinisa.rlba import *

"""
navigation with DynaQ RL
"""

class MapEnv(Environment):
	def __init__(self, states, actions, allStateActions, rewards, trackStates=False, trackActions=False):
		"""
		initializer

		Parameters
			trackStates :True if states vneed tracking
			trackActions : Tri=ue if actions for each state need tracking
			implReward : implicit reward, bayesian exploration bonus (beb or beblog), recency(rec) or default (None)
			implRewardFactor : implicit reward factor
		"""
		self.states = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]
		
		#path distances
		self.dist = dict()
		self.dist[("A", "B")] = 1.3
		self.dist[("A", "C")] = 1.2
		self.dist[("A", "I")] = 1.4
		self.dist[("B", "D")] = 1.8
		self.dist[("B", "F")] = 1.6
		self.dist[("C", "D")] = 1.7
		self.dist[("C", "E")] = 1.5
		self.dist[("D", "E")] = 1.1
		self.dist[("D", "G")] = 1.0
		self.dist[("E", "H")] = 1.7
		self.dist[("E", "M")] = 0.9
		self.dist[("F", "G")] = 1.4
		self.dist[("F", "J")] = 1.9
		self.dist[("H", "G")] = 1.6
		self.dist[("H", "M")] = 1.6
		self.dist[("I", "J")] = 1.6
		self.dist[("I", "K")] = 1.2
		self.dist[("I", "L")] = 2.1
		self.dist[("L", "M")] = 0.9
		
		
		#straight line distances
		self.sldist = dict()
		self.sldist[("A", "G")] = 1.70
		self.sldist[("B", "G")] = 1.50
		self.sldist[("C", "G")] = 1.65
		self.sldist[("D", "G")] = 1.00
		self.sldist[("E", "G")] = 1.75
		self.sldist[("F", "G")] = 1.40
		self.sldist[("H", "G")] = 1.50
		self.sldist[("I", "G")] = 3.10
		self.sldist[("J", "G")] = 2.75
		self.sldist[("K", "G")] = 4.10
		self.sldist[("L", "G")] = 2.65
		self.sldist[("M", "G")] = 1.70
		
		self.stsldist = self.sldist[("A", "G")]
		
		#next state and reward based on state and action
		self.stre = dict()
		self.stre[("A", "B")] = ["B"]
		self.stre[("A", "C")] = ["C"]
		self.stre[("A", "I")] = ["I"]
		self.stre[("B", "D")] = ["D"]
		self.stre[("B", "F")] = ["F"]
		self.stre[("C", "D")] = ["D"]
		self.stre[("C", "E")] = ["E"]
		self.stre[("C", "L")] = ["L"]
		self.stre[("D", "E")] = ["E"]
		self.stre[("D", "G")] = ["G"]
		self.stre[("E", "H")] = ["H"]
		self.stre[("E", "M")] = ["M"]
		self.stre[("F", "G")] = ["G"]
		self.stre[("F", "J")] = ["J"]
		self.stre[("H", "M")] = ["M"]
		self.stre[("H", "G")] = ["G"]
		self.stre[("I", "J")] = ["J"]
		self.stre[("I", "K")] = ["K"]
		self.stre[("I", "L")] = ["L"]
		self.stre[("L", "M")] = ["M"]
		
		#add reverse paths
		revstre = dict()
		for k in self.stre.keys():
			if k[1] != "G":
				nk = (k[1], k[0])
				revstre[nk] = [k[0]]
		self.stre.update(revstre)
		
		#reward
		print("reward and next state")
		for k in self.stre.keys():
			#print(k)
			cs = k[0]
			ac = k[1]
			print(cs, ac)
			re = self.reward(cs, ac)
			self.stre[k].append(re)
			print(self.stre[k])
			
		self.findInvalidActions()
		
		super(MapEnv, self).__init__(states, actions, allStateActions, rewards, trackStates=trackStates, trackActions=trackActions)

	def reward(self, cs, ac):
		"""
		return reward  based on current and next state
		
		Parameters
			cs : current state
			ns : next state
		"""
		ns = ac
		#print(cs, ns)
		if ns == "G":
			re = 1.0
		else:
			dnt = self.findDist(self.sldist, ns, "G")
			re = 0.3 * (self.stsldist - dnt) / self.stsldist
			
		print("cs {}  ns {}  re {:.3f}".format(cs, ns, re))
		return re
		
	def findDist(self, distances, cs, ns):
		"""
		return reward  based on current and next state
		
		Parameters
			distances : distance map
			cs : current state
			ns : next state
		"""
		d = distances[(cs,ns)] if (cs,ns) in distances else distances[(ns, cs)]
		return d
		
	def findInvalidActions(self):
		"""
		find invalid actions
		
		"""
		self.invalidActions = list()
		nums = len(self.states)
		for i in range(nums):
			for j in range(i, nums, 1):
				k = (self.states[i], self.states[j])
				if k not in self.dist:
					self.invalidActions.append(k)
					
					rk = (self.states[j], self.states[i])
					if i != j and not(self.states[i] == "G" and rk in self.dist):
						self.invalidActions.append(rk)
		
		print("invalid actions")
		for inv in self.invalidActions:
			print(inv)	
				
	def getReward(self, state, action):
		"""
		get next state and reward
		
		Parameters
			state : state
			action : action
		"""
		print(state, action)
		k = (state, action)
		reward = self.stre[k][1] + self.implicitReward(state, action)
		re = (self.stre[k][0], reward)
		return re

class WraehouseEnv(Environment):
	def __init__(self, states, actions, allStateActions, rewards, defaultReward=-0.1, trackStates=False, 
	trackActions=False,implReward="beb", implRewardFactor=.5):
		"""
		initializer		
		
		"""
		super(WraehouseEnv, self).__init__(states, actions, allStateActions, rewards, defaultReward=defaultReward, 
		trackStates=trackStates, trackActions=trackActions, implReward=implReward, implRewardFactor=implRewardFactor)
		

		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--niter', type=int, default = 100, help = "num of of iteration")
	parser.add_argument('--siter', type=int, default = 10, help = "num of of model simulatiom iteration")
	parser.add_argument('--cs', type=str, default = "none", help = "current state")
	parser.add_argument('--ac', type=str, default = "none", help = "action")
	parser.add_argument('--lrate', type=float, default = 0.1, help = "learning rate")
	parser.add_argument('--dfactor', type=float, default = 0.9, help = "decay rate")
	parser.add_argument('--bandit', type=str, default = "rg", help = "bandit algorithm")
	parser.add_argument('--eps', type=float, default = 0.1, help = "bandit algo epsilon")
	parser.add_argument('--eprpol', type=str, default = "linear", help = "bandit algo eps reduction policy")
	parser.add_argument('--eprp', type=float, default = .001, help = "bandit algo epsilon reduction parameter")
	parser.add_argument('--implrwd', type=str, default = "none", help = "impilcit reward")
	parser.add_argument('--irfactor', type=float, default = 0.3, help = "implicit reawrd factor")
	parser.add_argument('--savefp', type=str, default = "none", help = "model save file")
	parser.add_argument('--restorefp', type=str, default = "none", help = "model restore file path")
	parser.add_argument('--logfp', type=str, default = "none", help = "log file path")
	parser.add_argument('--loglev', type=str, default = "none", help = "log level")
	args = parser.parse_args()
	op = args.op
	
	
	if op == "reward":
		menv = MapEnv(trackStates=True, trackActions=True)
		r = menv.reward(args.cs, args.ac)
		print("reward {:.3f}".format(r))
		
	elif op == "train":
		""" train navigation model """
		menv = MapEnv(None, None, None, None, trackStates=True, trackActions=True)
		banditParams = dict()
		banditParams["epsilon"] = args.eps
		banditParams["redPolicy"] = args.eprpol
		banditParams["redParam"] = args.eprp if args.eprpol == "stepred" else None
		banditParams["nonGreedyActions"] = None
		
		qvPath = args.restorefp if args.restorefp != "none" else None	
		envModel = DetEnvModel()
		model = DynaQvalue(menv.states, menv.states, menv, args.bandit, banditParams, args.lrate, args.dfactor, "A", envModel,  "G",
		qvPath=qvPath, invalidStateActiins = menv.invalidActions)
		model.train(args.niter, args.siter, menv)
		policy = model.getPolicy(menv)
		print("policy")
		for s in policy.keys():
			print(s, policy[s])
			
		qvs = model.getQvalUpdates()
		drawPlot(None, qvs, "iteration", "Q Value Update")
		
		if args.savefp != "none":
			model.save(args.savefp)

	elif op == "whtrain":
		""" train warehouse navigation model """
		states = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"]
		actions = list()
		allStateActions = dict()
		allStateActions["A"] = {"AB" : "B"}
		allStateActions["B"] = {"BA" : "A", "BC" : "C"}
		allStateActions["C"] = {"CB" : "B", "CD" : "D",  "CH" : "H"}
		allStateActions["D"] = {"DE" : "E", "DC" : "C"}
		allStateActions["E"] = {"ED" : "D"}
		allStateActions["F"] = {"FG" : "G"}
		allStateActions["G"] = {"GF" : "F", "GH" : "H"}
		allStateActions["H"] = {"HC" : "C", "HG" : "G",  "HI" : "I", "HM" : "M"}
		allStateActions["I"] = {"IH" : "H", "IJ" : "J"}
		allStateActions["J"] = {"JI" : "I"}
		allStateActions["K"] = {"KL" : "L"}
		allStateActions["L"] = {"LK" : "K", "LM" : "M"}
		allStateActions["M"] = {"MH" : "H", "ML" : "L",  "MN" : "N"}
		allStateActions["N"] = {"NM" : "M", "NO" : "O"}
		allStateActions["O"] = {"ON" : "N"}
		
		for acSt in allStateActions.values():
			actions.extend(list(acSt.keys()))
		print(actions)	
		
		rewards = dict()
		rewards[("N", "NO")] = 1.0
		implReward = args.implrwd if args.implrwd != "none" else None
		wenv = WraehouseEnv(states, actions, allStateActions, rewards, trackStates=True, trackActions=True, implReward=implReward, 
		implRewardFactor=args.irfactor)
		if implReward == "empow":
			wenv.getAdjacentStatesCount(2)
		
		invalidStateActiins = wenv.getInvalidStateActions()
		banditParams = dict()
		banditParams["epsilon"] = args.eps
		banditParams["redPolicy"] = args.eprpol
		banditParams["redParam"] = args.eprp if args.eprpol == "stepred" else None
		banditParams["nonGreedyActions"] = None
		qvPath = args.restorefp if args.restorefp != "none" else None	
		envModel = DetEnvModel()
		logFilePath = args.logfp if args.logfp != "none" else None
		logLevName = args.loglev if args.loglev != "none" else None
		
		model = DynaQvalue(states, actions, wenv, args.bandit, banditParams, args.lrate, args.dfactor, "D", envModel,  "O",
		qvPath=qvPath, invalidStateActiins=invalidStateActiins,logFilePath=logFilePath, logLevName=logLevName)
		model.train(args.niter, args.siter, wenv)
				
		policy = model.getPolicy(wenv)
		print("policy")
		for s in policy.keys():
			print(s, policy[s])
		
		if args.savefp != "none":
			model.save(args.savefp)
	
	else:
		exitWithMsg("invalid command")
