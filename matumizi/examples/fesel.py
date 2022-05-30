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
import statistics 
import matplotlib.pyplot as plt 
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.daexp import *
from matumizi.sampler import *

NFEAT = 11
NFEAT_EXT = 14

class LoanApprove:
	def __init__(self, numLoans=None):
		self.numLoans = numLoans
		self.marStatus = ["married", "single", "divorced"]
		self.loanTerm = ["7", "15", "30"]
		self.addExtra = False


	def initTwo(self):
		"""
		initialize samplers
		"""
		self.approvDistr = CategoricalRejectSampler(("1", 60), ("0", 40))
		self.featCondDister = {}
		
		#marital status
		key = ("1", 0)
		distr = CategoricalRejectSampler(("married", 100), ("single", 60), ("divorced", 40))
		self.featCondDister[key] = distr
		key = ("0", 0)
		distr = CategoricalRejectSampler(("married", 40), ("single", 100), ("divorced", 40))
		self.featCondDister[key] = distr
	
		
		# num of children
		key = ("1", 1)
		distr = CategoricalRejectSampler(("1", 100), ("2", 90), ("3", 40))
		self.featCondDister[key] = distr
		key = ("0", 1)
		distr = CategoricalRejectSampler(("1", 50), ("2", 70), ("3", 100))
		self.featCondDister[key] = distr

		# education
		key = ("1", 2)
		distr = CategoricalRejectSampler(("1", 30), ("2", 80), ("3", 100))
		self.featCondDister[key] = distr
		key = ("0", 2)
		distr = CategoricalRejectSampler(("1", 100), ("2", 40), ("3", 30))
		self.featCondDister[key] = distr

		#self employed
		key = ("1", 3)
		distr = CategoricalRejectSampler(("1", 40), ("0", 100))
		self.featCondDister[key] = distr
		key = ("0", 3)
		distr = CategoricalRejectSampler(("1", 100), ("0", 30))
		self.featCondDister[key] = distr
		
		# income
		key = ("1", 4)
		distr = GaussianRejectSampler(120,15)
		self.featCondDister[key] = distr
		key = ("0", 4)
		distr = GaussianRejectSampler(50,10)
		self.featCondDister[key] = distr

		# years of experience
		key = ("1", 5)
		distr = GaussianRejectSampler(15,3)
		self.featCondDister[key] = distr
		key = ("0", 5)
		distr = GaussianRejectSampler(5,1)
		self.featCondDister[key] = distr

		# number of years in current job
		key = ("1", 6)
		distr = GaussianRejectSampler(3,.5)
		self.featCondDister[key] = distr
		key = ("0", 6)
		distr = GaussianRejectSampler(1,.2)
		self.featCondDister[key] = distr

		# outstanding debt
		key = ("1", 7)
		distr = GaussianRejectSampler(20,5)
		self.featCondDister[key] = distr
		key = ("0", 7)
		distr = GaussianRejectSampler(60,10)
		self.featCondDister[key] = distr
		
		# loan amount
		key = ("1", 8)
		distr = GaussianRejectSampler(300,50)
		self.featCondDister[key] = distr
		key = ("0", 8)
		distr = GaussianRejectSampler(600,50)
		self.featCondDister[key] = distr
		
		# loan term 
		key = ("1", 9)
		distr = CategoricalRejectSampler(("7", 100), ("15", 40), ("30", 60))
		self.featCondDister[key] = distr
		key = ("0", 9)
		distr = CategoricalRejectSampler(("7", 30), ("15", 100), ("30", 60))
		self.featCondDister[key] = distr
		
		# credit score
		key = ("1", 10)
		distr = GaussianRejectSampler(700,20)
		self.featCondDister[key] = distr
		key = ("0", 10)
		distr = GaussianRejectSampler(500,50)
		self.featCondDister[key] = distr
		
		if self.addExtra:
			# saving
			key = ("1", 11)
			distr = NormalSampler(80,10)
			self.featCondDister[key] = distr
			key = ("0", 11)
			distr = NormalSampler(60,8)
			self.featCondDister[key] = distr
			
			# retirement
			zDistr = NormalSampler(0, 0)
			key = ("1", 12)
			sDistr = DiscreteRejectSampler(0,1,1,20,80)
			nzDistr = NormalSampler(100,20)
			distr = DistrMixtureSampler(sDistr, zDistr, nzDistr)
			self.featCondDister[key] = distr
			key = ("0", 12)
			sDistr = DiscreteRejectSampler(0,1,1,50,50)
			nzDistr = NormalSampler(40,10)
			distr = DistrMixtureSampler(sDistr, zDistr, nzDistr)
			self.featCondDister[key] = distr
		
			#num of prior mortgae loans
			key = ("1", 13)
			distr = DiscreteRejectSampler(0,3,1,20,60,40,15)
			self.featCondDister[key] = distr
			key = ("0", 13)
			distr = DiscreteRejectSampler(0,1,1,70,30)
			self.featCondDister[key] = distr
			
		
	def generateTwo(self, noise, keyLen, addExtra):
		"""
		ancestral sampling
		"""
		self.addExtra = addExtra
		self.initTwo()
		
		#error
		erDistr = GaussianRejectSampler(0, noise)
	
		#sampler
		numChildren = NFEAT_EXT if self.addExtra else NFEAT
		sampler = AncestralSampler(self.approvDistr, self.featCondDister, numChildren)

		for i in range(self.numLoans):
			(claz, features) = sampler.sample()
		
			# add noise
			features[4] = int(features[4])
			features[7] = int(features[7])
			features[8] = int(features[8])
			features[10] = int(features[10])
			if self.addExtra:
				features[11] = int(features[11])
				features[12] = int(features[12])

			claz = addNoiseCat(claz, ["0", "1"], noise)

			strFeatures = list(map(lambda f: toStr(f, 2), features))
			rec =  genID(keyLen) + "," + ",".join(strFeatures) + "," + claz
			print (rec)

	def encodeDummy(self, fileName, extra):
		"""
		dummy var encoding
		"""
		catVars = {}
		catVars[1] = self.marStatus
		catVars[10] = self.loanTerm
		rSize = NFEAT_EXT if extra else NFEAT
		rSize += 2
		dummyVarGen = DummyVarGenerator(rSize, catVars, "1", "0", ",")
		for row in fileRecGen(fileName, None):
			newRow = dummyVarGen.processRow(row)
			print (newRow)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--nloan', type=int, default = 1000, help = "nom of loans")
	parser.add_argument('--noise', type=float, default = 0.1, help = "nom of loans")
	parser.add_argument('--klen', type=int, default = 1000, help = "key length")
	parser.add_argument('--fpath', type=str, default = "none", help = "source file path")
	parser.add_argument('--algo', type=str, default = "none", help = "source file path")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":
		"""  generate data """
		numLoans = args.nloan
		loan = LoanApprove(numLoans)
		noise = args.noise
		keyLen = args.klen
		addExtra = True 
		loan.generateTwo(noise, keyLen, addExtra)

	elif op == "encd":
		""" encode binary """
		fileName = args.fpath
		extra = True
		loan = LoanApprove()
		loan.encodeDummy(fileName, extra)
	
	
	elif op == "fsel":
		""" feature select  """
		fpath = args.fpath
		algo = args.algo
		expl = DataExplorer(False)
		expl.addFileNumericData(fpath, 5, 8, 11, 12, "income", "debt", "crscore", "saving")
		expl.addFileCatData(fpath, 3, 4, 15, "education", "selfemp", "target")
		
		fdt = ["education", "cat", "selfemp", "cat", "income", "num",  "debt", "num", "crscore", "num"]
		tdt = ["target", "cat"]
		if args.algo == "mrmr":
			res = expl.getMaxRelMinRedFeatures(fdt, tdt, 3)
		elif args.algo == "jmi":
			res = expl.getJointMutInfoFeatures(fdt, tdt, 3)
		elif args.algo == "cmim":
			res = expl.getCondMutInfoMaxFeatures(fdt, tdt, 3)
		elif args.algo == "icap":
			res = expl.getInteractCapFeatures(fdt, tdt, 3)

		print(res)
	else:
		exitWithMsg("invalid command")		