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
from tseda import *

"""
driver  for time series data exploration
"""

def getNumPlot(data, args):
	"""
	get num of plots
	
	Parameters
		data : data
		args : command line args
	"""
	if args.szplots > 0:
		nplots = int(len(data) / args.szplots)
	else:
		nplots = args.nplots
	return nplots

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file file path")
	parser.add_argument('--dfcol', type=int, default = 0, help = "data file column index")
	parser.add_argument('--srate', type=float, default = 0, help = "sampling rate")
	parser.add_argument('--period', type=int, default = 0, help = "seasonal period")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--szplots', type=int, default = -1, help = "sizeplots")
	parser.add_argument('--wlen', type=int, default = -1, help = "window length")
	parser.add_argument('--pstep', type=int, default = -1, help = "window processing step size")
	args = parser.parse_args()
	op = args.op
	
	if op == "fft":
		""" fft """
		res = fft([args.dfpath, args.dfcol], args.srate)
		drawPlotParts(res["frquency"], res["fft"], "frequency", "fft", 5)
	
	elif op == "comp":
		""" get trend, cycle and remainder """
		res = components([args.dfpath, args.dfcol], "additive", args.period, False, True)	
		seas = res["seasonal"]	
		nplots = getNumPlot(seas, args)
		drawPlotParts(None, seas, "time", "value", nplots)
	
	elif op == "tsstat":
		""" two sample statistic  """
		res = twoSampleStat([args.dfpath, args.dfcol], args.wlen, args.pstep, "ks")
		print(res)
	
	elif op == "msshift":
		""" mean and std deviation shift """
		data = getListData([args.dfpath, args.dfcol])
		detector = MeanStdShiftDetector(args.wlen, args.pstep)
		for d in data:
			detector.add(d)
		res = detector.getResult()
		print(res)
		
		diffs = detector.getDiffList()
		nplots = getNumPlot(diffs[0], args)
		drawPlotParts(None, diffs[0], "time", "mean diff", nplots)
	
	else:
		exitWithMsg("invalid command")	
			
