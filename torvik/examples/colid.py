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

import os
import sys
from random import randint
import time
from array import *
import argparse
import statistics
import numpy as np
from matumizi.mlutil import *
from matumizi.util import *
from matumizi.sampler import *
from matumizi.daexp import *
from torvik.tnn import *
from torvik.fftn import *

"""
column type idenification
"""
def getStats(expl, ds):
	"""
	col stats
	"""
	stats = expl.getCatAlphaCharCountStats(ds)
	acmean = stats["mean"]
	acsd = stats["std dev"]
	
	stats = expl.getCatNumCharCountStats(ds)
	ncmean = stats["mean"]
	ncsd = stats["std dev"]
	
	stats = expl.getCatFldLenStats(ds)
	lmean = stats["mean"]
	lsd = stats["std dev"]
	
	stats = expl.getCatCharCountStats(ds, ' ')
	spmean = stats["mean"]
	spsd = 	 stats["std dev"]
	
	return [lmean, lsd, acmean, acsd, ncmean, ncsd, spmean, spsd]	

def sampRecOfClass(tdata, cl):
	"""
	sample rec of given class
	"""
	r = selectRandomFromList(tdata)
	while r[-1] != cl:
		r = selectRandomFromList(tdata)
	return r
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--fpath', type=str, default = "none", help = "source file path")
	parser.add_argument('--ssize', type=int, default = 50, help = "sample size")
	parser.add_argument('--nsamp', type=int, default = 100, help = "number of samples")
	parser.add_argument('--sfpref', type=str, default = "none", help = "sample filename prefix")
	parser.add_argument('--fncnt', type=int, default = 1000, help = "file name counter")
	parser.add_argument('--sfdir', type=str, default = "none", help = "sample file directory")
	parser.add_argument('--cfpath', type=str, default = "none", help = "column features file path")
	parser.add_argument('--nrepl', type=int, default = 2, help = "no of replicationns")
	parser.add_argument('--mlfpath', type=str, default = "none", help = "ml config file path")
	parser.add_argument('--clabel', type=str, default = "str", help = "class label numeric or string")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":
		""" extract required fields and process data from original data """
		srcFilePath = args.fpath
		i = 0
		for rec in fileRecGen(srcFilePath, ","):
			if i > 0:
				nrec = list()
				fname = rec[0][1:-1]
				lname = rec[1][1:-1]
				nrec.append(fname + " " + lname)
				nrec.append(rec[-9][1:-1])
				nrec.append(rec[-8][1:-1])
				nrec.append(rec[-6][1:-1])
				z = rec[-5]
				if len(z) == 7:
					z = z[1:-1]
				nrec.append(z)
				nrec.append(rec[-2][1:-1])
				print(",".join(nrec))
			i += 1


	elif op == "sample":
		""" generate sample files """
		fpath = args.fpath
		tdata = getFileLines(fpath)
		for i in range(args.nsamp):
			sfpath = args.sfdir + "/" +  args.sfpref + "_" + str(args.fncnt + i) + ".txt"
			with open(sfpath, "w") as fh:
				for j in range(args.ssize):
					rec = selectRandomFromList(tdata)[:3]
					mfield = randint(0,2)
					for k in range(3):
						if k != mfield:
							fv = selectRandomFromList(tdata)[k]
							rec[k] = fv
					fh.write(",".join(rec) + "\n")
	
	elif op == "cfeatures":
		""" create column features """
		sfpaths = getAllFiles(args.sfdir)
		expl = DataExplorer(False)
		clabels = ["0", "1", "2"] if args.clabel == "num" else ["N", "A", "C"]
		
		for sfp in 	sfpaths:
			#print(sfp)
			names = list()
			addressses = list()
			cities = list()
			for rec in fileRecGen(sfp):
				names.append(rec[0])
				addressses.append(rec[1])
				cities.append(rec[2])
			
			expl.addListCatData(names, "name")		
			expl.addListCatData(addressses, "address")		
			expl.addListCatData(cities, "city")		
			
			stats = getStats(expl, "name")
			print(toStrFromList(stats, 3) + "," + clabels[0])
			stats = getStats(expl, "address")
			print(toStrFromList(stats, 3) + "," + clabels[1])
			stats = getStats(expl, "city")
			print(toStrFromList(stats, 3) + ",", clabels[2])
	
	elif op == "cpairs":
		""" create col features pair """
		tdata = getFileLines(args.cfpath)
		le = len(tdata)
		for i in range(le):
			r1 = tdata[i]
			cp, cn = args.nrepl, args.nrepl
			while cp > 0 or cn > 0:
				j = randint(0, le - 1)
				while j == i:
					j = randint(0, le - 1)
				r2 = tdata[j]
				if r1[-1] == r2[-1]:
					#same type
					if cp > 0:
						#print(i,j,cp,cn)
						#print("P and P")
						r = r1[:-1].copy()
						r.extend(r2[:-1].copy())
						r.append("1")
						cp -= 1
						print(",".join(r))
				else:
					#opposite type
					if cn > 0:
						#print(i,j,cp,cn)
						#print("P and N")
						r = r1[:-1].copy()
						r.extend(r2[:-1].copy())
						r.append("0")
						cn -= 1
						print(",".join(r))
						
	elif op == "ctriplet":
		""" create col triplet """
		tdata = getFileLines(args.cfpath)
		le = len(tdata)
		for i in range(le):
			r1 = tdata[i]
			for j in range(args.nrepl):
				pr = None
				nr = None
				while pr is None or nr is None:
					k = randint(0, le - 1)
					while k == i:
						k = randint(0, le - 1)
					r2 = tdata[k]
					if r1[-1] == r2[-1]:
						if pr is None: 
							pr = r2[:-1].copy()
					else:
						if nr is None: 
							nr = r2[:-1].copy()
				
				r = r1[:-1].copy()
				r.extend(pr)
				r.extend(nr)
				r.append("0")
				print(",".join(r))
					
	elif op == "gtest":
		""" create test data """
		fpaths = args.cfpath.split(",")
		vtdata = getFileLines(fpaths[0])
		ttdata = getFileLines(fpaths[1])
		
		#one for each class
		classes = ["N", "A", "C"]
		for vr  in vtdata:
			cl = vr[-1]
			
			rn = list()
			ra = list()
			rc = list()
			allr = {"N" : rn, "A" : ra, "C" : rc}
			while len(rn) < 2 or len(ra) < 2 or len(rc) < 2:
				tr = selectRandomFromList(ttdata)
				tcl = tr[-1]
				allr[tcl].append(tr[:-1].copy())
			
			#pos prototype	
			r = vr[:-1].copy()
			r.extend(allr[cl][0])
			r.extend(allr[cl][1])
			r.append("0")
			print(",".join(r))
			
			#negative protyples
			for c in classes:
				if c != cl:
					r = vr[:-1].copy()
					r.extend(allr[c][0])
					r.extend(allr[c][1])
					r.append("0")
					print(",".join(r))
				
	elif op == "gpred":
		""" create test data """
		fpaths = args.cfpath.split(",")
		ptdata = getFileLines(fpaths[0])
		ttdata = getFileLines(fpaths[1])
		
		#one for each class
		classes = ["N", "A", "C"]
		for pr  in ptdata:
			rn = list()
			ra = list()
			rc = list()
			allr = {"N" : rn, "A" : ra, "C" : rc}
			while len(rn) < 2 or len(ra) < 2 or len(rc) < 2:
				tr = selectRandomFromList(ttdata)
				tcl = tr[-1]
				allr[tcl].append(tr[:-1].copy())
			
			#negative protyples
			for c in classes:
				r = pr[:-1].copy()
				r.extend(allr[c][0])
				r.extend(allr[c][1])
				r.append("0")
				print(",".join(r))

	elif op == "rmcl":
		""" remopve class label """
		tdata = getFileLines(args.fpath)
		for r in tdata:
			print(",".join(r[:-1]))
				
	elif op == "train":
		mod = FeedForwardMultiNetwork(args.mlfpath)
		mod.buildModel()
		FeedForwardMultiNetwork.batchTrain(mod)
		
	elif op == "test":
		mod = FeedForwardMultiNetwork(args.mlfpath)
		mod.buildModel()
		FeedForwardMultiNetwork.testModel(mod)

	elif op == "pred":
		mod = FeedForwardMultiNetwork(args.mlfpath)
		mod.buildModel()
		FeedForwardMultiNetwork.predictModel(mod)

	else:
		exitWithMsg("invalid command")