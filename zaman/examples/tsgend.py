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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	parser.add_argument('--ovcfpath', type=str, default = "none", help = "overriding config file path")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file path")
	parser.add_argument('--prec', type=int, default = 3, help = "floating point precision")
	parser.add_argument('--nplots', type=int, default = 3, help = "num of plots")
	parser.add_argument('--yscale', type=int, default = 3, help = "plot yscsale")
	args = parser.parse_args()
	op = args.op
	
	ovcfpath = None if args.ovcfpath == "none" else args.ovcfpath
	generator = TimeSeriesGenerator(args.cfpath, ovcfpath)
	if op == "tsn":
		""" trend, cycle and noise based generation """
		da = list()
		for rec in generator.trendCycleNoiseGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		drawLineParts(da, args.nplots, args.yscale)
	
	if op == "triang":
		""" triangular cyclic  based generation """
		da = list()
		for rec in generator.triangGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		drawLineParts(da, args.nplots, args.yscale)

	if op == "step":
		""" step based generation """
		da = list()
		for rec in generator.stepGen():
			print(rec)
			da.append(float(rec.split(",")[1]))
		drawLineParts(da, args.nplots, args.yscale)

	elif op == "insan":
		""" insert sequence anomaly """
		da = list()
		for rec in generator.insertAnomalySeqGen(args.dfpath, args.prec):			
			print(rec)
			da.append(float(rec.split(",")[1]))
		drawLineParts(da, args.nplots, args.yscale)
	
	elif op == "insanp":
		""" insert point anomaly """
		da = list()
		for rec in generator.insertAnomalyPointGen(args.dfpath, args.prec):			
			print(rec)
			da.append(float(rec.split(",")[1]))
		drawLineParts(da, args.nplots, args.yscale)
