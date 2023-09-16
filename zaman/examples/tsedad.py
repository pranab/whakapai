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
from zaman.tseda import *
from tsgend import getNumPlot

"""
driver  for time series data exploration
"""


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file file path")
	parser.add_argument('--dfcol', type=int, default = 0, help = "data file column index")
	parser.add_argument('--srate', type=float, default = 0, help = "sampling rate")
	parser.add_argument('--period', type=int, default = 0, help = "seasonal period")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--szplots', type=int, default = -1, help = "sizeplots")
	parser.add_argument('--pbeg', type=int, default = -1, help = "plot begin offset")
	parser.add_argument('--pend', type=int, default = -1, help = "plot end offset")
	parser.add_argument('--yscale', type=int, default = -1, help = "plot yscsale")
	parser.add_argument('--wlen', type=int, default = -1, help = "window length")
	parser.add_argument('--pstep', type=int, default = -1, help = "window processing step size")
	parser.add_argument('--wvlet', type=str, default = "morl", help = "wavelet function")
	parser.add_argument('--wscales', type=str, default = "none", help = "wavelet transform scales")
	parser.add_argument('--wfreqs', type=str, default = "none", help = "wavelet transform frequencies")
	parser.add_argument('--sampf', type=float, default = 100, help = "samplig freq")
	parser.add_argument('--cutoff', type=float, default = -1, help = "cut off frequency")
	parser.add_argument('--forder', type=float, default = 5, help = "filter order")
	args = parser.parse_args()
	op = args.op

	yscale = args.yscale if args.yscale > 0 else None
	
	if op == "fft":
		""" fft """
		res = fft([args.dfpath, args.dfcol], args.srate)
		drawPlotParts(res["frquency"], res["fft"], "frequency", "fft", 5)
	
	elif op == "comp":
		""" get trend, cycle and remainder """
		res = components([args.dfpath, args.dfcol], "additive", args.period, False, True)	
		seas = res["seasonal"]	
		pdata, nplots = getNumPlot(seas, args)
		drawPlotParts(None, pdata, "time", "value", nplots)
	
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
		pdata, nplots = getNumPlot(diffs[0], args)
		drawPlotParts(None, pdata, "time", "mean diff", nplots)

	elif op == "hpfilt":
		""" high pass filter """
		ts = getListData([args.dfpath, 0])
		data = getListData([args.dfpath, args.dfcol])
		fdata = bhpassFilter(data, args.cutoff, args.srate, args.forder)		
		
		pdata, nplots = getNumPlot(fdata, args)
		if nplots > 0:
			drawLineParts(pdata, nplots, yscale)
	
	elif op == "wlet":
		""" wavelet transform  """
		data = getListData([args.dfpath, args.dfcol])
		wtrans = None
		if args.wscales != "none":
			scales = strToFloatArray(args.wscales)
			wtrans = WaveletExpl(data, args.wvlet, args.sampf, scales=scales)
		elif args.wfreqs != "none":
			freqs = strToFloatArray(args.wfreqs)
			wtrans = WaveletExpl(data, args.wvlet, args.sampf, freqs=freqs)
		else:
			exitWithmsg("eithre scales or frequencies should be provided")
			
		wtrans.transform()
		while true:
			cmd = input()
			cmds = cmd.split()
			if cmds[0] == "freq":
				iscale = int(cmds[1])
				doPlot = cmds[2] == "true"
				nparts = int(cmds[3])
				wtrans.atFreq(iscale, doPlot, nparts)
			elif cmds[0] == "time":
				itime = int(cmds[1])
				doPlot = cmds[2] == "true"
				wtrans.atTime(itime, doPlot)
			elif cmds[0] == "quit":
				break
	else:
		exitWithMsg("invalid command")	
			
