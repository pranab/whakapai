#!/usr/local/bin/python3

# matumizi: Machine Learning
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
import math
from datetime import datetime
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from matumizi.daexp import *

"""
Time series feature extraction
"""


class IntervalFeatureExtractor(object):

	def __init__(self):
		"""
		Initializer
		"""
		pass
		
	def featGen(self, dfpath, nintervals=None, intvmin=None, intvmax=None, intervals=None, ifpath=None, overlap=False, withLabel=True, prec=3):
		"""
		extracts mean, std dev and slope for multiple intervals
		
		Parameters
			dfpath : data file path
			nintervals : num of intervals
			intvmin : interval min size
			intvmax : interval max size
			intervals : list of intervals
			ifpath : intervals file path
			overlap: if inetral overlap allowed then True
			withLabel : True if each TS sequence is labeled
			prec : float output precision]
		"""
		for rec in fileRecGen(dfpath):
			frec = rec[:-1] if withLabel else rec
			
			if intervals is None:
				rlen = len(frec)
				intervals = list()
				if overlap:
					#interval overlap allowed
					for i in range(nintervals):
						intvlen = randomInt(intvmin, intvmax)
						stmax = rlen - intvlen - 1
						st = randomInt(0, stmax)
						en = st + intvlen
						intv = (st,en)
						intervals.append(intv)
				else:	
					#interval overlap not allowed
					stb = 0
					remain = rlen
					for i in range(nintervals):
						remintv = nintervals - i
						ste = stb + int((remain - remintv * intvmax) / remintv)
						st = randomInt(stb, ste)
						intvlen = randomInt(intvmin, intvmax)
						#print("renmain {}  stb {}  ste {}  st {} intvlen {}".format(remain,stb,ste,st,intvlen))
						en = st + intvlen
						intv = (st,en)
						intervals.append(intv)
						stb = en + 1 
						remain = rlen - stb
						
				
			#stats based features
			features = list()
			for intv in intervals:
				intvdata = frec[intv[0]:intv[1]]
				intvdata = toFloatList(intvdata)
				
				#mean and std dev
				mean = statistics.mean(intvdata)
				sd = statistics.stdev(intvdata, xbar=mean)
				
				#slope
				expl = DataExplorer()
				expl.setVerbose(False)
				expl.addListNumericData(intvdata, "mydata")
				slope = expl.fitLinearReg("mydata")["slope"]
				features.append(mean)
				features.append(sd)
				features.append(slope)
			
			if withLabel:
				features.append(rec[-1])
			feat = toStrFromList(features, prec)
			yield feat
			
		if ifpath is not None:
			with open(ifpath, "w") as fintv:
				for intv in intervals:
					fintv.write(str(intv[0]) + "," + str(intv[1]) + "\n")

				
					
		