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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from tseda import *
from tsgend import getNumPlot

"""
driver  for time series data exploration
"""
def plot3D(fr,tm,va):
	ax = plt.figure().add_subplot(projection='3d')
	ax.plot3D(fr, tm, va, cmap=cm.coolwarm)
	ax.set_xlabel('frequency')
	ax.set_ylabel('time')
	ax.set_zlabel('value')
	plt.show()
	
def surface3D(fr,tm,va):
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	tm,fr = np.meshgrid(tm,fr)
	surf = ax.plot_surface(fr, tm, va, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter('{x:.03f}')
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('frequency')
	ax.set_ylabel('time')
	ax.set_zlabel('value')
	plt.show()
	

		
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
		ts = getFileColumnAsInt(args.dfpath, 0)
		data = getListData([args.dfpath, args.dfcol])
		#print("data", len(data))
		wtrans = None
		if args.wscales != "none":
			scales = strToFloatArray(args.wscales)
			wtrans = WaveletExpl(data, args.wvlet, args.srate, scales=scales)
		elif args.wfreqs != "none":
			freqs = strToFloatArray(args.wfreqs)
			wtrans = WaveletExpl(data, args.wvlet, args.srate, freqs=freqs)
		else:
			exitWithmsg("eithre scales or frequencies should be provided")
			
		wtrans.transform()
		print("entering command loop")
		while  True:
			print("command:")
			cmd = input()
			cmds = cmd.split()
			if cmds[0] == "freq":
				#time domain for given frequency usage: freq frequecy_index false nparts xlabel ylabel
				iscale = int(cmds[1])
				doPlot = cmds[2] == "true"
				nparts = int(cmds[3])
				wdata = wtrans.atFreq(iscale, doPlot, nparts)
				
				if not doPlot:
					xlabel = None
					if len(cmds) > 4:
						xlabel = cmds[4]
						ylabel = cmds[5]
					pdata, nplots = getNumPlot(wdata, args)
					#print("wdata", len(wdata))
					if nplots > 0:
						if xlabel is None:
							drawLineParts(pdata, nplots, yscale)
						else:
							pts = ts[args.pbeg:args.pend] if args.pbeg >= 0 and args.pend > 0 else ts
							drawPlotParts(pts, pdata, xlabel, ylabel, nplots)
						
				
			elif cmds[0] == "time":
				#frequency domain for given time usage: time time_index true
				itime = int(cmds[1])
				doPlot = cmds[2] == "true"
				wdata = wtrans.atTime(itime, doPlot)
			
			elif cmds[0] == "all":
				#frequency and time domain
				tbeg =  int(cmds[1])
				tend =  int(cmds[2])
				fr,tm,va = wtrans.atSection(tbeg, tend)
				surfacePlot(fr,tm,va,'frequency', 'time', 'value')
			
			elif cmds[0] == "wlfun":
				#transform for a wavelet function
				wavelet = cmds[1]
				wtrans.transform(wavelet)
				
			elif cmds[0] == "quit":
				#quit
				break
		
		print("exiting command loop")
	
	else:
		exitWithMsg("invalid command")	
			
