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
	def __init__(self):
		"""
		initializer
		"""
		#path distances
		self.dist = dict()
		self.dist[("A", "B")] = 1.2
		self.dist[("A", "C")] = 1.4
		self.dist[("A", "I")] = 1.4
		self.dist[("B", "D")] = 1.3
		self.dist[("B", "F")] = 1.6
		self.dist[("C", "D")] = 1.7
		self.dist[("C", "L")] = 1.5
		self.dist[("D", "E")] = 1.1
		self.dist[("D", "G")] = 1.0
		self.dist[("E", "H")] = 1.7
		self.dist[("E", "M")] = 0.9
		self.dist[("F", "G")] = 1.4
		self.dist[("F", "J")] = 1.9
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
		
		
		#next state and reward based on state and action
		self.stre = dict()
		self.stre[("A", "B")] = ["B"]
		self.stre[("A", "C")] = ["C"]
		self.stre[("A", "I")] = ["I"]
		self.stre[("B", "D")] = ["D"]
		self.stre[("B", "F")] = ["F"]
		self.stre[("C", "D")] = ["D"]
		self.stre[("C", "L")] = ["L"]
		self.stre[("D", "E")] = ["E"]
		self.stre[("D", "G")] = ["G"]
		self.stre[("E", "H")] = ["H"]
		self.stre[("E", "M")] = ["M"]
		self.stre[("F", "G")] = ["G"]
		self.stre[("F", "J")] = ["J"]
		self.stre[("H", "M")] = ["M"]
		self.stre[("I", "J")] = ["J"]
		self.stre[("I", "K")] = ["K"]
		self.stre[("I", "L")] = ["L"]
		self.stre[("L", "M")] = ["M"]
		
		#reward
		for k in self.stre.keys():
			#print(k)
			cs = k[0]
			ac = k[1]
			re = self.reward(cs, ac)
			self.stre[k].append(re)
			
		super(Environment, self).__init__()

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
			dns = self.findDist(self.dist, cs, ns)
			dct = self.findDist(self.sldist, cs, "G")
			dnt = self.findDist(self.sldist, ns, "G")
			re = 0.3 * dct / (dns + dnt)
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
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--niter', type=int, default = 100, help = "num of of iteration")
	parser.add_argument('--nmiter', type=int, default = 10, help = "num of of model iteration")
	parser.add_argument('--cs', type=str, default = "none", help = "current state")
	parser.add_argument('--ac', type=str, default = "none", help = "action")
	args = parser.parse_args()
	op = args.op
	
	menv = MapEnv()
	if op == "reward":
		r = menv.reward(args.cs, args.ac)
		print("reward {:.3f}".format(r))
		
	
