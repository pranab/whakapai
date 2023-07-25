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
from qinsia.cmab import *

"""
Adc placement using CMAB
"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--algo', type=str, default = "rg", help = "bandit algo")
	parser.add_argument('--nads', type=int, default = 5, help = "num of ads")
	parser.add_argument('--nsites', type=int, default = 10, help = "num of sites")
	parser.add_argument('--ntplay', type=int, default = 100, help = "total num of plays")
	parser.add_argument('--nplay', type=int, default = 20, help = "num of plays")
	parser.add_argument('--reg', type=float, default = 1.0, help = "regression regularizer")
	parser.add_argument('--pthresh', type=float, default = 0.05, help = "probablity threshold")
	parser.add_argument('--savefp', type=str, default = "none", help = "model save file")
	parser.add_argument('--restorefp', type=str, default = "none", help = "model restore file")
	args = parser.parse_args()
	
	if args.algo == "linucb":
		ads = list(map(lambda i : "adv_" + str(i+1), range(args.nads)))
		sites = list(map(lambda i : "site_" + str(i+1), range(args.nsites)))
		features = dict()
		
		#features for all sites
		for s in sites:
			afs = list()
			for _ in ads:
				af = list(map(lambda _ : randomFloat(0.4, 1.0), range(4)))
				afs.append(af)
			features[s] = afs
		
		noise = NormalSampler(0, 0.06)
		
		if args.restorefp == "none":
			model = LinUpperConfBound(ads, 4, args.ntplay, reg=args.reg, pthresh=args.pthresh, 
			logFilePath="./log/linucb.log", logLevName="info")
		else:
			model = LinUpperConfBound.create(args.restorefp, logFilePath="./log/linucb.log", logLevName="info")
			
		for _ in range(args.nplay):
			s = selectRandomFromList(sites)
			afs = features[s]
			adv = model.getAction(afs)
			
			ai = ads.index(adv)
			af = afs[ai]
			sf = sum(af) / 4
			re = .9 * sf + noise.sample()
			re = rangeLimit(re, 0, 1.0)
			print("site {}   adv {}  features {}  reward {:.3f}".format(s, adv, floatArrayToString(af, delem=None), re))
			model.setReward(adv, re)
		
		if args.savefp != "none":
			model.save(args.savefp)
			
			
			