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
from qinisa.mab import *

"""
Email campaign optimization  using MAB
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--nemails', type=int, default = 1000, help = "nom of emails")
	args = parser.parse_args()
	
	emtempl = ["d1", "d2", "d3"]
	if args.algo == "rg":
		model = RandomGreedy(emtempl, 20, True, "./log/rg.log", "info", 0.8, "loglin")
	elif args.algo == "ucb":
		model = UpperConfBound(emtempl, 20, True, "./log/ucb.log", "info")
	elif args.algo == "ts":
		model = ThompsonSampling(emtempl, 20, True, "./log/ts.log", "info")
	elif args.algo == "exp3":
		model = ExponentialWeight(emtempl, 20, True, "./log/ts.log", "info", 0.2)
	elif args.algo == "smix":
		model = SoftMix(emtempl, 20, True, "./log/smix.log", "info", 0.6)
	
	evsamplers = dict()	
	evsamplers["d1"] = CategoricalRejectSampler(("op", 80), ("cl", 20))
	evsamplers["d2"] = CategoricalRejectSampler(("op", 50), ("cl", 50))
	evsamplers["d3"] = CategoricalRejectSampler(("op", 30), ("cl", 70))
	rewards = dict()
	rewards["op"] = 0.3
	rewards["cl"] = 1.0
	emsent = list()
	
	for i in range(args.nemails):
		cid = genID(8)
		sel = model.getAction()
		camp = (cid, sel[0])
		print("next campaign email ", camp)
		emsent.append(camp)
		
		if i > 10 and isEventSampled(80):
			#email event and reward
			ai = randomInt(0, int(len(emsent) / 2))
			camp = emsent[ai]
			act  = camp[1]
			ev = evsamplers[act].sample()
			reward = rewards[ev]
			model.setReward(act, reward)
			emsent.remove(camp)
		
	reg = model.getRegret()
	print("max reward {:.3f}  actual reawrd {:.3f}  regret {:.3f}".format(reg[0], reg[1], reg[2]))
		
		
		
