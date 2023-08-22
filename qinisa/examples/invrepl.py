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
	r = 0.1 * inv if inv <= 10 else 1.0 - .005 * (inv - 10) 
	return r
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--ndays', type=int, default = 100, help = "num of of days")
	parser.add_argument('--lrate', type=float, default = 0.1, help = "learning rate")
	parser.add_argument('--dfactor', type=float, default = 0.9, help = "decay rate")
	parser.add_argument('--eps', type=float, default = 0.1, help = "bandit algo epsilon")
	parser.add_argument('--eprpol', type=str, default = "linear", help = "bandit algo eps reduction policy")
	parser.add_argument('--eprp', type=float, default = 0, help = "bandit algo epsilon reduction parameter")
	args = parser.parse_args()
	
	if args.algo == "sarsa":
		csize = 10
		wkndDem = NormalSampler(90, 8)
		wkndDem.sampleAsIntValue()
		wkdDem = NormalSampler(60, 10)
		wkdDem.sampleAsIntValue()
		
		states = list()
		for d in range(7):
			dw = 0 if d < 5 else 1
			for inv in range(12):
				states.append((dw, inv))
		actions = list(range(12))
		
		policy = dict()
		for s in states:
			if s[0] == 0:
				#week days
				a = 7 - s[1] if s[1] <= 7 else 0
			else:
				#week end
				a = 10 - s[1] if s[1] <= 10 else 0
			policy[s] = a		
		
		banditParams = {"epsilon" : args.eps, "redPolicy":args.eprpol, "redParam":args.eprp}
		istate = (0, 1)
		inv = randomInt(10, 19)
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
			
			
		
		
	
