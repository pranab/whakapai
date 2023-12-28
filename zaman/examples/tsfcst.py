#!/usr/bin/python

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

# Package imports
import os
import sys
import argparse
import statistics
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from dcmpnet import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	parser.add_argument('--prec', type=int, default = 3, help = "floating point precision")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--findex', type=int, default = -1, help = "forecast woindow index")
	parser.add_argument('--tibeg', type=int, default = -1, help = "time begin index")
	parser.add_argument('--tiend', type=int, default = -1, help = "time end index")
	parser.add_argument('--trvafile', type=str, default = "none", help = "trend validation file")
	parser.add_argument('--revafile', type=str, default = "none", help = "remian validation file")
	parser.add_argument('--dfpath', type=str, default = "none", help = "data file path")
	parser.add_argument('--dtype', type=str, default = "all", help = "data type")

	args = parser.parse_args()
	op = args.op
	
	if op !=  "norm":
		dn = DecmpNetwork(args.cfpath)
	
	if op == "norm":
		""" normalize by removing mean """
		col = getFileColumnAsFloat(args.dfpath, 1)
		mean = statistics.mean(col)
		mfpath = args.dtype + "_mean.txt"
		with open(mfpath, "w") as fh:
			fh.write(formatFloat(args.prec, mean))
			
		for r in fileRemMeanRecGen(args.dfpath, 1, prec=2):
			print(r)
	
	elif op == "decomp":
		""" decompose into trend and remaining """
		if args.dfpath != "none":
			dn.config.setParam("train.data.file", args.dfpath)
			
		if args.dtype == "all":
			dn.decompose()
		elif args.dtype == "training":
			dn.decomposeOne(True)
		else:
			dn.decomposeOne(False)
		
	elif op == "train":
		""" train trend and remaining models """
		dn.fit()

	elif op == "validate":
		""" validate forecast """
		trVaFpath = args.trvafile if args.trvafile != "none" else None
		reVaFpath = args.revafile if args.revafile != "none" else None
		dn.validate(args.findex, args.tibeg, args.tiend, trVaFpath, reVaFpath)
		
	else:
		exitWithMsg("invalid command")
