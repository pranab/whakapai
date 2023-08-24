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
import enquiries
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from qinisa.reinfl import *

"""
store inventory replenishment
"""

def reward(inv):
	""" calculates  reward from left over inventory """
	
	#max at 10 falls sharply as inventory reaches 0
	if inv <= 5:
		r = 0.2 * inv
	elif inv <= 10:
		r = 1.0
	else:
		r = 1.0 - .1 * (inv - 10)
	
	return r

def printPolicy(policy, states):
	""" print policy """	
	for st in states:
		print("{}\t\t{}".format(st, policy[st]))
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--ndays', type=int, default = 100, help = "num of of days")
	parser.add_argument('--bandit', type=str, default = "rg", help = "bandit algorithm")
	parser.add_argument('--demstat', type=str, default = "80,6,50,8", help = "deman distr parameters")
	parser.add_argument('--lrate', type=float, default = 0.1, help = "learning rate")
	parser.add_argument('--dfactor', type=float, default = 0.9, help = "decay rate")
	parser.add_argument('--eps', type=float, default = 0.1, help = "bandit algo epsilon")
	parser.add_argument('--eprpol', type=str, default = "linear", help = "bandit algo eps reduction policy")
	parser.add_argument('--eprp', type=float, default = 0, help = "bandit algo epsilon reduction parameter")
	parser.add_argument('--savefp', type=str, default = "none", help = "model save file")
	parser.add_argument('--restorefp', type=str, default = "none", help = "model restore file")
	args = parser.parse_args()

	csize = 10
	ncase = 16
	dparams = strToFloatArray(args.demstat)
	wkndDem = NormalSampler(dparams[0], dparams[1])
	wkndDem.sampleAsIntValue()
	wkdDem = NormalSampler(dparams[2], dparams[3])
	wkdDem.sampleAsIntValue()
	
	states = list()
	for dw in range(2):
		for inv in range(ncase):
			states.append((dw, inv))
	actions = list(range(ncase))
	print(states)
	print(actions)
	
	
	if args.algo == "sarsa":
		policy = dict()
		for s in states:
			if s[0] == 0:
				#week days
				a = 6 - s[1] if s[1] <= 6 else 0
			else:
				#week end
				a = 8 - s[1] if s[1] <= 8 else 0
			policy[s] = a		
		print("current policy")
		printPolicy(policy, states)
		
		#non greedy actions  for exploration around policy only
		ngacts = dict()
		for a in range(ncase):
			if a == 0:
				ngacts[a] = [a + 1, a + 2]
			elif a == ncase - 1:
				ngacts[a] = [a - 2, a - 1]
			else:
				ngacts[a] = [a - 1, a + 1]
		
		banditParams = {"epsilon" : args.eps, "redPolicy":args.eprpol, "redParam":args.eprp, "nonGreedyActions":ngacts}
		istate = (0, 1)
		inv = randomInt(1, 9)
		model = TempDifferenceControl(states, actions, "rg", banditParams, args.lrate, args.dfactor, istate, policy=policy, 
		onPolicy=True, logFilePath="./log/reifl.log", logLevName="info")
		
		for i in range(args.ndays):
			d = i % 7
			dem = wkdDem.sample() if d < 5 else wkndDem.sample()
			supl = model.getAction() * csize
			print("day {} inventory {}  supply {} demand {}".format(d, inv, supl, dem))
			inv = inv + supl - dem
			inv = max(0, inv)
			re = reward(inv)
			print("new inventory {}  reward {:.3f}".format(inv, re))
			
			nd = (i+1) % 7
			dw = 0 if nd < 5 else 1
			invc = round(inv / csize)
			ns = (dw, invc)
			model.setReward(re, ns)
			
		print("updated policy")
		npolicy = model.getPolicy()
		for st in states:
			ac = npolicy[st]
			oac = policy[st]
			msg = "no change"
			if ac is None:
				ac = oac
				msg = "action default to existing policy"
			elif ac != oac:
				msg = "change to eixting policy action {}".format(oac)
			print("{}\t{}\t{}".format(st, ac, msg))


	elif args.algo == "qlearn":
		banditParams = {"epsilon" : args.eps, "redPolicy":args.eprpol, "redParam":args.eprp, "nonGreedyActions":None}
		istate = (0, 1)
		inv = randomInt(1, 9)
		
		qvPath = args.restorefp if args.restorefp != "none" else None
		model = TempDifferenceControl(states, actions, args.bandit, banditParams, args.lrate, args.dfactor, istate, qvPath=qvPath, 
		logFilePath="./log/reifl.log", logLevName="info")
		
		if qvPath is not None:
			cpolicy = model.getPolicy()
			
		nbounds = 0
		for i in range(args.ndays):
			d = i % 7
			dem = wkdDem.sample() if d < 5 else wkndDem.sample()
			supl = model.getAction() * csize
			print("day {} inventory {}  supply {} demand {}".format(d, inv, supl, dem))
			inv = inv + supl - dem
			inv = max(0, inv)
			re = reward(inv)
			print("new inventory {}  reward {:.3f}".format(inv, re))
			
			nd = (i+1) % 7
			dw = 0 if nd < 5 else 1
			invc = round(inv / csize)
			if invc >= ncase - 1:
				invc = ncase - 1
				nbounds += 1
			ns = (dw, invc)
			model.setReward(re, ns)
		
		print("outside bounds {}".format(nbounds))
		
		if qvPath is not None:
			print("current policy")
			printPolicy(cpolicy, states)
		
		npolicy = model.getPolicy()
		print("new policy")
		printPolicy(npolicy, states)
			
		if args.savefp != "none":
			model.save(args.savefp)
		
	
