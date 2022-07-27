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
sys.path.append(os.path.abspath("../reinf"))
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from qinisa.reinfl import *
from qinisa.rlba import *


"""
Air fare pricing policy evaluation
"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "evalst", help = "operation")
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--policy', type=str, default = "p1", help = "policy")
	parser.add_argument('--nepisode', type=int, default = 20, help = "no of episodes")
	args = parser.parse_args()
	
		
	if args.op == "evalst": 
		""" evaluate state value """
		#period before flight in days range
		periods = dict()
		periods["P1"] = (60, 41)
		periods["P2"] = (40, 21)
		periods["P3"] = (20, 11)
		periods["P4"] = (10, 0)
		
		#occupancy range
		occupancy =  dict()
		occupancy["O1"] = (250, 299)
		occupancy["O2"] = (300, 349)
		occupancy["O3"] = (350, 399)
		occupancy["O4"] = (400, 449)
		occupancy["O5"] = (450, 500)
		
		#first policy state: period + occcupancy action: d = discount e = extra + amount
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
		pol1["P1O4"] = "e3"
		pol1["P2O4"] = "d0"
		pol1["P3O4"] = "d0"
		pol1["P4O4"] = "d3"
		pol1["P1O5"] = "e8"
		pol1["P2O5"] = "e3"
		pol1["P3O5"] = "d0"
		pol1["P4O5"] = "d0"
		
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
		pol2["P1O3"] = "e3"
		pol2["P2O3"] = "d0"
		pol2["P3O3"] = "d0"
		pol2["P4O3"] = "d3"
		pol2["P1O4"] = "e8"
		pol2["P2O4"] = "e3"
		pol2["P3O4"] = "d0"
		pol2["P4O4"] = "d0"
		pol2["P1O5"] = "e12"
		pol2["P2O5"] = "e8"
		pol2["P3O5"] = "e3"
		pol2["P4O5"] = "d0"
		
		#reward distr mean and std dev
		discounts = [0, -3, -5, -8, -10, -12, -15, 3, 8, 12]
		mean = dict()
		mean["P1"] = 60
		mean["P2"] = 70
		mean["P3"] = 75
		mean["P4"] = 80
		 
		std = dict()
		std["P1"] = 9
		std["P2"] = 6
		std["P3"] = 4
		std["P4"] = 3
		
		#reward sampler based on period and discount
		pers = list(mean.keys())
		rdistr = dict()
		for d in discounts:
			for p in pers:
				m = mean[p] + 2 * (-d - 3) 
				s = std[p] - 0.2 * (-d - 3) 
				ds = "d" + str(-d) if d <= 0 else "e" + str(d)
				key = p + ds
				print("demand key {}  mean {:.3f}  std dev {:.3f}".format(key, m, s))
				rdistr[key] = NormalSampler(m,s)
				 
		
		pol = pol1 if args.policy == "p1" else pol2
		pol = Policy(True, pol)
		
		osampler = CategoricalRejectSampler(("O1", 70), ("O2", 30))
		oc = osampler.sample()
		st = "P1" + oc
		fp = "./log/tdl.log"
		td = TempDifferenceValue(pol, 0.2, 0.2, 0.95, st, fp, "info")
		
		values = list()
		for i in range(args.nepisode):
			print("**next episode " + str(i))
			ocu = random.randint(250, 299) if oc == "O1" else random.randint(300, 349)
			episode = True
			while episode:
				ac = td.getAction()
				print("state {}  action {}".format(st, ac))
				key = st[:2] + ac
				#print("demand key {}".format(key))
				dem = int(rdistr[key].sample())
				if (ocu + dem) > 500:
					dem = 500 - ocu
					
				dis = int(ac[1:])
				dis = -dis if ac[0] == 'd' else dis
				price = (1 + dis / 100)
				re = dem * price
				print("demand {:.3f}  price {:.3f} occupation {}".format(dem, price, ocu))
				for k in occupancy.keys():
					if ocu >= occupancy[k][0] and ocu <= occupancy[k][1]:
						oc = k
						break
				
				#next period
				cp = st[:2]
				pe = int(cp[1:])
				if pe == 4:
					#next episode
					episode = False
					pe = 1
					oc = osampler.sample()
					nst = "P1" + oc
				else:
					#current episode
					pe += 1
					ocu += dem
					ocu = 500 if ocu > 500 else ocu
					nst = "P" + str(pe) + oc
				
				print("reward {:.3f}  next state {}".format(re, nst))
				td.setReward(re, nst)
				st = nst
				if not episode:
					tval = td.getTotValue()				
					print("episode end total value {:.3f}".format(tval))
					values.append(tval)
					
		
		print("state values")
		vals = td.getValues()
		for k in vals.keys():
			print("state {}  value {:.3f}".format(k, vals[k]))
			
		drawLine(values)
		
				
				
				
		
		
		
		
