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
import argparse
import statistics
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from zaman.nntuner import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "model config file path")
	parser.add_argument('--tcfpath', type=str, default = "", help = "auto tuner config file path")
	parser.add_argument('--prec', type=int, default = 3, help = "floating point precision")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--tvsplit', type=float, default = .8, help = "training validation split")
	parser.add_argument('--dfpath', type=str, default = "none", help = "data file path")
	parser.add_argument('--trdfpath', type=str, default = "none", help = "training data file path")
	parser.add_argument('--vadfpath', type=str, default = "none", help = "validation data file path")
	parser.add_argument('--ntrials', type=int, default = 100, help = "num of trails for auto tuner")
	
	args = parser.parse_args()
	op = args.op
	
	if op == "tvsplit":
		"""" split train and validation """
		trdata = list()
		vadata = list()
		threshold = int(100 * args.tvsplit)
		for rec in fileRecGen(args.dfpath, None):
			if isEventSampled(threshold):
				trdata.append(rec)
			else:
				vadata.append(rec)
		
		with open(args.trdfpath, "w") as ftr:
			for d in trdata:
				ftr.write(d + "\n")
		
		with open(args.vadfpath, "w") as fva:
			for d in vadata:
				fva.write(d + "\n")
	
	elif op == "train":
		"""" train """
		mod = FeedForwardNetwork(args.cfpath)
		mod.buildModel()
		score = mod.fit()
				
	elif op == "autotrain":
		"""" auto train """
		re = NeuralNetworkTuner.tune(args.cfpath, args.tcfpath, args,ntrials)
	
	else:
		exitWithMsg("invalid command")		
