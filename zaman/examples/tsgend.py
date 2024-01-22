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
from zaman.tsgen import *
from zaman.tsfeat import *


"""
driver  for time series data generation
"""

def getNumPlot(data, args):
	"""
	get num of plots
	
	Parameters
		data : data
		args : command line args
	"""
	pdata = data[args.pbeg:args.pend] if args.pbeg >= 0 and args.pend > 0 else data
	if args.szplots > 0:
		nplots = int(len(pdata) / args.szplots)
		#print("num plots", nplots)
	else:
		nplots = args.nplots
	return pdata, nplots
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	parser.add_argument('--ovcfpath', type=str, default = "none", help = "overriding config file path")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file path")
	parser.add_argument('--prec', type=int, default = 3, help = "floating point precision")
	parser.add_argument('--nplots', type=int, default = -1, help = "num of plots")
	parser.add_argument('--yscale', type=int, default = -1, help = "plot yscsale")
	parser.add_argument('--szplots', type=int, default = -1, help = "size of plots")
	parser.add_argument('--pbeg', type=int, default = -1, help = "plot begin offset")
	parser.add_argument('--pend', type=int, default = -1, help = "plot end offset")
	parser.add_argument('--exscomp', type=str, default = "none", help = "additional sine components")
	parser.add_argument('--xlabel', type=str, default = "none", help = "plot x label")
	parser.add_argument('--ylabel', type=str, default = "none", help = "plot y label")
	parser.add_argument('--oconfig', type=str, default = "none", help = "overirde config")
	parser.add_argument('--siparams', type=str, default = "none", help = "sine wave parameters")
	parser.add_argument('--clabels', type=str, default = "none", help = "class labels")
	parser.add_argument('--nintervals', type=int, default = 3, help = "num of intervals")
	parser.add_argument('--intvmin', type=int, default = 100, help = "min interval length")
	parser.add_argument('--intvmax', type=int, default = 500, help = "max interval length")
	args = parser.parse_args()
	op = args.op
	
	if op != "intvfe":
		ovcfpath = None if args.ovcfpath == "none" else args.ovcfpath
		generator = TimeSeriesGenerator(args.cfpath, ovcfpath)
		yscale = args.yscale if args.yscale > 0 else None
		if args.oconfig != "none":
			parts = args.oconfig.split("=")
			generator.config.setParam(parts[0], parts[1])
	
	if op == "tsn":
		""" trend, cycle and noise based generation """
		da = list()
		for rec in generator.trendCycleNoiseGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		pdata, nplots = getNumPlot(da, args)
		if args.nplots > 0:
			drawLineParts(pdata, nplots, yscale)
	
	if op == "triang":
		""" triangular cyclic  based generation """
		da = list()
		for rec in generator.triangGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		if args.nplots > 0:
			drawLineParts(da, args.nplots, yscale)

	if op == "step":
		""" step based generation """
		da = list()
		for rec in generator.stepGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		if args.nplots > 0:
			drawLineParts(da, args.nplots, yscale)

	if op == "motif":
		""" motif based generation """
		da = list()
		for rec in generator.motifGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		if args.nplots > 0:
			drawLineParts(da, args.nplots, yscale)

	if op == "sine":
		"""multiple sine function based generation """
		da = list()
		exscomp = args.exscomp if args.exscomp != "none" else None
		#print(exscomp)
		for rec in generator.multSineGen(exscomp):
			print(rec)
			da.append(float(rec.split(",")[1]))
		pdata, nplots = getNumPlot(da, args)
		if nplots > 0:
			drawLineParts(pdata, nplots, yscale)

	if op == "sineclf":
		"""multiple sine function based generation for classification """
		da = list()
		
		#num of classes
		siparams = args.siparams.split(":")
		ncl = len(siparams) + 1
		
		#no of samples per class
		nsamples = int(generator.config.getIntConfig("output.value.nsamples")[0] / ncl)
		generator.config.setParam("output.value.nsamples", str(nsamples))
		
		clabels = args.clabels.split(",")
		
		for c in range(ncl):
			pcnt = 0
			if c > 0:
				generator.config.setParam("si.params", siparams[c-1])
				
			for rec in generator.multSineGen():
				lrec = rec + "," + clabels[c]
				print(lrec)
				if args.szplots > 0 and pcnt < 2:
					pdata = toFloatList(rec)[:args.szplots]
					drawLine(pdata)
					pcnt += 1
					
	elif op == "insan":
		""" insert sequence anomaly """
		da = list()
		for rec in generator.insertAnomalySeqGen(args.dfpath, args.prec):			
			print(rec)
			da.append(float(rec.split(",")[1]))
		pdata, nplots = getNumPlot(da, args)
		if nplots > 0:
			drawLineParts(pdata, nplots, yscale)
	
	elif op == "insanp":
		""" insert point anomaly """
		da = list()
		for rec in generator.insertAnomalyPointGen(args.dfpath, args.prec):			
			print(rec)
			da.append(float(rec.split(",")[1]))
		pdata, nplots = getNumPlot(da, args)
		if nplots > 0:
			drawLineParts(pdata, args.nplots, yscale)
			
	elif op == "plot":
		""" plot """
		ts = getFileColumnAsInt(args.dfpath, 0)
		da = getFileColumnAsFloat(args.dfpath, 1)
		pdata, nplots = getNumPlot(da, args)
		if nplots > 0:
			if args.xlabel == "none":
				drawLineParts(pdata, nplots, yscale)
			else:
				pts = ts[args.pbeg:args.pend] if args.pbeg >= 0 and args.pend > 0 else ts
				drawPlotParts(pts, pdata, args.xlabel, args.ylabel, nplots)
				
	elif op == "intvfe":
		""" interval based feature extraction """
		intvFeat = IntervalFeatureExtractor()
		for frec in intvFeat.featGen(args.dfpath, args.nintervals, args.intvmin, args.intvmax, prec=args.prec):
			print(frec)
		
		
		
	
