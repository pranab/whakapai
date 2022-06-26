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
import enquiries
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from qinisa.reinfl import *
from qinisa.rlba import *


"""
Email campaign optimization  using MAB
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "eval", help = "operation")
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--policy', type=str, default = "p1", help = "policy")
	parser.add_argument('--nepisode', type=int, default = 20, help = "no of episodes")
	args = parser.parse_args()
	
		
	if args.op == "evalv": 
		#period before flight
		periods = dict()
		periods["P1"] = (60, 41)
		periods["P2"] = (40, 21)
		periods["P3"] = (20, 11)
		periods["P4"] = (10, 0)
		
		#occupancy percentage
		occupancy =  dict()
		occupancy["O1"] = (250, 299)
		occupancy["O2"] = (300, 349)
		occupancy["O3"] = (350, 399)
		occupancy["O4"] = (400, 449)
		occupancy["O5"] = (450, 500)
		
		#first policy
		pol1 = dict()
		pol1["P1O1"] = "d0"
		pol1["P2O1"] = "d5"
		pol1["P3O1"] = "d10"
		pol1["P4O1"] = "d15"
		pol1["P1O2"] = "d0"
		pol1["P2O2"] = "d5"
		pol1["P3O2"] = "d8"
		pol1["P4O2"] = "d12"
		pol1["P1O3"] = "d0"
		pol1["P2O3"] = "d0"
		pol1["P3O3"] = "d3"
		pol1["P4O3"] = "d8"
		pol1["P1O4"] = "d0"
		pol1["P2O4"] = "d0"
		pol1["P3O4"] = "d0"
		pol1["P4O4"] = "d3"
		pol1["P1O5"] = "d0"
		pol1["P2O5"] = "d0"
		pol1["P3O5"] = "e3"
		pol1["P4O5"] = "e8"
		
		#second plociy more aggressive
		pol2 = dict()
		pol2["P1O1"] = "d0"
		pol2["P2O1"] = "d3"
		pol2["P3O1"] = "d8"
		pol2["P4O1"] = "d10"
		pol2["P1O2"] = "d0"
		pol2["P2O2"] = "d0"
		pol2["P3O2"] = "d3"
		pol2["P4O2"] = "d8"
		pol2["P1O3"] = "d0"
		pol2["P2O3"] = "d0"
		pol2["P3O3"] = "d0"
		pol2["P4O3"] = "d3"
		pol2["P1O4"] = "d0"
		pol2["P2O4"] = "d0"
		pol2["P3O4"] = "d0"
		pol2["P4O4"] = "e3"
		pol2["P1O5"] = "d0"
		pol2["P2O5"] = "e3"
		pol2["P3O5"] = "e8"
		pol2["P4O5"] = "e12"
		
		#reward distr
		discounts = [0, -3, -8, -10, -12, -15, 3, 8, 12]
		mean = dict()
		mean["P1"] = 150
		mean["P2"] = 180
		mean["P3"] = 130
		mean["P4"] = 140
		 
		std = dict()
		std["P1"] = 10
		std["P2"] = 6
		std["P3"] = 4
		std["P4"] = 3
		
		pers = list(mean.keys())
		rdistr = dict()
		for d in discounts:
			for p in pers:
				m = mean[p] + 2 * (-d - 3) 
				s = std[p] - 0.2 * (-d - 3) 
				ds = "d" + str(-d) if d <= 0 else "e" + str(d)
				print(p, ds)
				key = p + ds
				rdistr[key] = NormalSampler(m,s)
				 
		
		pol = pol1 if args.policy == "p1" else pol2
		pol = Policy(True, pol)
		
		osampler = CategoricalRejectSampler(("O1", 70), ("O2", 30))
		oc = osampler.sample()
		st = "P1" + oc
		td = TempDifferenceValue(pol, 0.1, 0.95, st, "./log/tdl.log", "info")
		
		for _ in range(args.nepisode):
			print("next episode")
			ocu = random.randint(250, 299) if oc == "O1" else random.randint(300, 349)
			episode = True
			while episode:
				ac = td.getAction()
				key = st[:2] + ac
				dem = int(rdistr[key].sample())
				di = int(ac[1:])
				di = -di if ac[0] == 'd' else di
				re = dem * (1 + di)
				ocu += dem
				ocu = 500 if ocu > 500 else ocu
				for k in occupancy.keys():
					if ocu >= occupancy[k][0] and ocu <= occupancy[k][1]:
						oc = k
						break
				cp = st[:2]
				p = int(cp[1:])
				if p == 4:
					episode = False
					p = 1
					vac = 500 - ocu
					re -= vac
					oc = osampler.sample()
					nst = "P1" + oc
				else:
					p += 1
					nst = "P" + str(p) + oc
				td.setReward(re, nst)
				st = nst
		
		vals = td.getValues()
		for k in vals.keys():
			print("state {}  value {:.3f}".format(k, vals[k]))
						
				
				
				
		
		
		
		
