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

# Package imports
import os
import sys
import random
import jprops
import numpy as np
import statistics 
import argparse
from matumizi.util import *
from matumizi.sampler import *
from arotau.opti import *
from arotau.optsolo import *

class ApptCost():
	""""
	
	"""
	def __init__(self, pids):
		"""
		intialize
		"""
		self.pdetails = dict()
		tsampler = DiscreteRejectSampler(1, 2, 1, .9, .1)
		print("patient details")
		for pid in pids:
			ptype = tsampler.sample()
			
			#delay compared to desired day
			delay = randomInt(0, 1) if ptype == 2  else randomInt(0, 5)
			
			#days since request
			rdelay = 0 if ptype == 2 else randomInt(0, 10)
			
			self.pdetails[pid] = (ptype, delay, rdelay)
			print("ID {}  type {}  delay {} request {}".format(pid, ptype, delay, rdelay))

	def isValid(self, args):
		"""
		validation
		"""
		valid = True
		for pid in self.pdetails.keys():
			if pid not in args:
				det = self.pdetails[pid]
			
				#normal patient with 5 days delay will have to be included
				if det[0] == 1 and det[1] == 5:
					valid = False
					break
		
		return valid
		
	def evaluate(self, args):
		"""
		cost
		"""
		tcost = 0
		tcount = 0
		for pid in self.pdetails.keys():
			if pid not in args:
				det = self.pdetails[pid]
				
				#cost due to delay
				efact = 0.1 if det[0] == 1 else 0.5
				cost = 1.0 - 1.0 / math.exp(efact * det[1])
				
				#cost based time gap between request date and aappt date
				rcost = 1.0 - 1.0 / math.exp(0.2 * det[2])
				
				tcost += (cost + rcost)
				tcount += 1
		return tcost / tcount


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--npat', type=int, default = 10, help = "num of patients")
	parser.add_argument('--pidfpath', type=str, default = "", help = "patient id file path")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	args = parser.parse_args()
	
	op = args.op
	
	if op == "gpid":
		""" pat=tient id  """
		for _ in range(args.npat):
			pid = genNumID(8)
			print(pid)
			
	elif op == "opt":
		""" patient details  """
		pids = getFileLines(args.pidfpath, None)
		apptc = ApptCost(pids)
		optimizer = SimulatedAnnealingOptimizer(args.cfpath, apptc)
		optimizer.run()
		print("optimizer started, check log file for output details...")
	
		#best soln
		print("\nbest solution found")
		best = optimizer.getBest()
		#print("cost {:.3f}".format(best.cost))
		#print(best.soln)
		print(str(best))

		
		
