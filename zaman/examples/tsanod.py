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
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from zaman.tsano import *
from tsgend import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file path")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--szplots', type=int, default = -1, help = "size of plots")
	parser.add_argument('--pbeg', type=int, default = -1, help = "plot begin offset")
	parser.add_argument('--pend', type=int, default = -1, help = "plot end offset")
	parser.add_argument('--ylabel', type=str, default = "", help = "plot y label")
	parser.add_argument('--anvalue', type=int, default = -1, help = "anomaly value column")
	args = parser.parse_args()
	op = args.op

	if op == "mcm":
		""" build markov chanin model """
		ad = MarkovChainAnomaly(args.cfpath)
		ad.fit()
		
	elif op == "mcp":
		""" predict using  markov chanin model """
		ad = MarkovChainAnomaly(args.cfpath)
		ad.predict()
		
	elif op == "hfe":
		""" fit and predict using histogram based feature """
		ad = FeatureBasedAnomaly(args.cfpath)
		ad.fit()
		res = ad.predict()
		

	elif op == "plot":
		""" plot  """
		ts = getFileColumnAsInt(args.dfpath, 0)
		data = getFileColumnAsFloat(args.dfpath, args.anvalue)
		pdata, nplots = getNumPlot(data, args)
		tss = ts[args.pbeg:args.pend] if args.pbeg >= 0 and args.pend > 0 else ts
		drawPlotParts(tss, pdata, "Time", args.ylabel, nplots)
		
	else:
		exitWithMsg("invalid time series anomaly detection command")

		