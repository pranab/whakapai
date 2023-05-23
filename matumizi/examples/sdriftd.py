#!/usr/local/bin/python3

# beymani-python: Machine Learning
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
import argparse
import statistics 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KDTree
from matumizi.util import *
from matumizi.sampler import *
from matumizi.sdrift import *

"""
concept drift data generation and detection
"""
def linTrans(val, scale, shift):
	return val * scale + shift

def addDrift(fpath, dLoc=None, outPath=None):
	""" modify for dtift """
	i = 0
	wrFile =  outPath is not None
	if wrFile:
		ofile = open(outPath, "w")

	for rec in fileRecGen(fpath):
		if dLoc is not None and i > dLoc:
			fi = 1
			tran = linTrans(float(rec[fi]), 1.1, 30)
			rec[fi] = tran
			fi += 1
			ga = int(linTrans(int(rec[fi]), 0.95, -6))
			rec[fi] = ga
			fi += 1
			du = int(linTrans(int(rec[fi]), 1.2, 120))
			rec[fi] = du
			fi += 1
			srch = int(linTrans(int(rec[fi]), 1.3, 2))
			rec[fi] = srch
			fi += 1
			issue = int(linTrans(int(rec[fi]), 1, 1))
			rec[fi] = issue
			fi += 2
			pissue = int(linTrans(int(rec[fi]), 1, 1))
			rec[fi] = pissue
			r = toStrFromList(rec, 2)
		else:
			r = ",".join(rec)
			
		if wrFile:
			ofile.write(r + "\n")
		else:
			print(r)
		i += 1
			
	if wrFile:
		ofile.close()
					
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--nsamp', type=int, default = 1000, help = "num of samples")
	parser.add_argument('--oerate', type=float, default = 0.1, help = "old error rate")
	parser.add_argument('--nerate', type=float, default = 0.2, help = "new error rate")
	parser.add_argument('--trans', type=float, default = -1.0, help = "transition point for drift")
	parser.add_argument('--dfpath', type=str, default = "", help = "data file file path")
	parser.add_argument('--threshold', type=float, default = 3.5, help = "threshold for drift")
	parser.add_argument('--conflev', type=float, default = 0.2, help = "confidence level")
	parser.add_argument('--wsize', type=int, default = 100, help = "window size")
	parser.add_argument('--wpsize', type=int, default = 20, help = "window processing step size")
	parser.add_argument('--warmup', type=int, default = 100, help = "warmup size")
	parser.add_argument('--expf', type=float, default = 0.7, help = "exponential factor")
	parser.add_argument('--fprate', type=int, default = 100, help = "false positivev rate for drift")
	parser.add_argument('--savefp', type=str, default = "none", help = "model save file")
	parser.add_argument('--restorefp', type=str, default = "none", help = "model restore file")
	args = parser.parse_args()
	op = args.op

	if op == "agen":
		""" abrupt drift data """
		nsamp = args.nsamp
		oerate = args.oerate
		osampler = BernoulliTrialSampler(oerate)
		trans = None
		if args.trans > 0:
			trans = int(args.trans * nsamp)
			nsampler = BernoulliTrialSampler(args.nerate)
		curTime, pastTime = pastTime(10, "d")
		stime = pastTime
		for i in range(nsamp):
			# 1 = error or wrong prediction 0 = no error or correct prediction
			if trans is not None and i > trans:
				# higher error
				er = 1 if nsampler.sample() else 0
			else:
				# normal
				er = 1 if osampler.sample() else 0
			rid = genID(10)
			stime += random.randint(30, 300) 
			print("{},{},{}".format(rid, stime, er))
	
	elif op == "ddm":
		""" DDM detector """
		fpath = args.dfpath
		evals = getFileColumnAsInt(fpath, 2)
		
		if args.restorefp == "none":
			detector = DDM(args.threshold, args.warmup, args.wsize, args.wpsize)
		else:
			detector = DDM.create(args.restorefp)
		xp = list()
		ys = list()
		yd = list()
		for i in range(len(evals)):
			res = detector.add(evals[i])
			if res is not None:
				print("{:.3f},{:.3f},{:.3f},{}".format(res[0],res[1],res[2],res[3]))
				xp.append(i)
				ys.append(res[2])
				yd.append(res[3])
		
		if args.savefp != "none":
			detector.save(args.savefp)
		drawPairPlot(xp, ys, yd, "predictions", "value", "score", "drift")

	elif op == "eddm":
		""" EDDM detector """
		fpath = args.dfpath
		evals = getFileColumnAsInt(fpath, 2)
		if args.restorefp == "none":
			detector = EDDM(args.threshold, args.warmup, args.wsize, args.wpsize)
		else:
			detector = EDDM.create(args.restorefp)
			
		xp = list()
		ys = list()
		yd = list()
		for i in range(len(evals)):
			res = detector.add(evals[i])
			if res is not None:
				print("{:.3f},{:.3f},{:.3f},{}".format(res[0],res[1],res[2],res[3]))
				xp.append(i)
				ys.append(res[2])
				yd.append(res[3])

		if args.savefp != "none":
			detector.save(args.savefp)
		drawPairPlot(xp, ys, yd, "predictions", "value", "score", "drift")
		
	elif op == "fhddm":
		""" FHDDM detector """
		fpath = args.dfpath
		evals = getFileColumnAsInt(fpath, 2)
		if args.restorefp == "none":
			detector = FHDDM(args.conflev, args.warmup, args.wsize, args.wpsize)
		else:
			detector = FHDDM.create(args.restorefp)
		xp = list()
		ys = list()
		yd = list()
		for i in range(len(evals)):
			res = detector.add(evals[i])
			if res is not None:
				print("{:.3f},{}".format(res[0],res[1]))
				xp.append(i)
				ys.append(res[0])
				yd.append(res[1])
		
		if args.savefp != "none":
			detector.save(args.savefp)
		drawPairPlot(xp, ys, yd, "predictions", "value", "score", "drift")
		
	elif op == "ecdd":
		"""ECDD detector """
		fpath = args.dfpath
		evals = getFileColumnAsInt(fpath, 2)
		detector = ECDD(args.expf, args.fprate, args.warmup)
		xp = list()
		yd = list()
		for i in range(len(evals)):
			res = detector.add(evals[i])
			if res is not None:
				print("{:.3f},{}".format(res[0],res[1]))
				xp.append(i)
				yd.append(res[1])
		drawPlot(xp, yd, "predictions", "drift")
	else:
		exitWithMsg("invalid command")
		