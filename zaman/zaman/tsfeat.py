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
import numpy as np
from scipy import fft
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from matumizi.daexp import *
from matumizi.stats import *

"""
Time series feature extraction
"""

"""
interval statistics based feature extraction
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
			prec : float output precision
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

				
"""
quantization and histogram based feature extraction
"""
class QuantizedFeatureExtractor(object):

	def __init__(self):
		"""
		Initializer
		"""
		pass

	def binWidth(self, dfpath,  dformat="tabular", vcol=1, nbins=10, padding=None, withLabel=True):
		"""
		finds bin width and min value
		
		Parameters
			dfpath : data file path
			dformat : data format tabular or single column of data
			vcol : value column for columnar data
			nbins : number of bins
			padding : padding at ends to account for extreme values
			withLabel : True if each TS sequence is labeled			prec : float output precision
		"""
		#min and max
		dmaxv = None
		dminv = None
		if dformat == "tabular":
			for rec in fileRecGen(dfpath):
				frec = rec[:-1] if withLabel else rec
				frec = toFloatList(frec)
				maxv = max(frec)
				minv = min(frec)
				if dmaxv is None:
					dmaxv = maxv
				else:
					dmaxv = maxv if maxv > dmaxv else dmaxv
				if dminv is None:
					dminv = minv
				else:
					dminv = minv if minv < dminv else dminv
		else:
			dvalues = getFileColumnAsFloat(dfpath, vcol)	
			dmaxv = max(dvalues)	
			dminv = min(dvalues)	
		
		#padding and bin width
		if padding is not None:
			padsize = (dmaxv - dminv) * padding
			dminv -= padsize
			dmaxv += padsize
		bwidth = (dmaxv - dminv) / nbins
		
		re = (dminv, bwidth)
		return re
		
	def featGen(self, dfpath, vmin, bwidth, dformat="tabular", vcol=1, nbins=10,  histType="uniform", rowWise=True, withLabel=True, prec=3, wsize=50, retArr=True):
		"""
		calculates histogram for each record
		
		Parameters
			dfpath : data file path or a list
			vmin : min value
			bwidth : bin width list if equal samples histogram
			dformat : data format tabular or single column of data
			vcol : value column for columnar data
			nbins : number of bins
			histType : histogram type eqwidth for equal width bin and eqsample for equal number of samples in each bin
			rowWise : if True row wise feature generation
			withLabel : True if each TS sequence is labeled
			prec : float output precision
			wsize : window size for data is single column
			retArr ; If True returns array otherwise delem separated string
		"""
		if histType == "uniform":
			hgram = Histogram.createUninitializedWithNumBins(vmin, bwidth, nbins)
		else:
			existWithMsg("equal sample histogram not supported yet")
		
		if dformat == "tabular":
			# tabluar with multiple values per row
			for rec in fileRecGen(dfpath):
				frec = rec[:-1] if withLabel else rec
				frec = toFloatList(frec)
				for d in frec:
					hgram.add(d)
			
				if rowWise:
					#features per row
					features = hgram.distr()
					if withLabel:
						features.append(rec[-1])
					feat = features if retArr else toStrFromList(features, prec)
					hgram.initialize()
					yield feat
			
			if not rowWise:
				#all data
				features = hgram.distr()
				feat = features if retArr else toStrFromList(features, prec)
				yield feat
		else:
			dvalues = getFileColumnAsFloat(dfpath, vcol)
			
			#one value per window location
			if rowWise:
				#windowed
				slwin = SlidingWindow(dvalues, wsize)
				for wdata in slwin.windowGen():
					for d in wdata:
						hgram.add(d)
					features = hgram.distr()
					feat = features if retArr else toStrFromList(features, prec)
					hgram.initialize()
					yield feat
				
			else:
				#all data
				for d in dvalues:
					hgram.add(d)
				features = hgram.distr()
				feat = features if retArr else toStrFromList(features, prec)
				yield feat

		
		
"""
FFT based feature extraction
"""
class FourierTransformFeatureExtractor(object):

	def __init__(self):
		"""
		Initializer
		"""
		pass

	def featGen(self, dfpath, cutoff, dformat="tabular", vcol=1, rowWise=True, withLabel=True, prec=3, wsize=50, retArr=True):
		"""
		calculates FFT for each record
		
		Parameters
			dfpath : data file path or a list
			cutoff : cutoff index
			dformat : data format tabular or single column of data
			vcol : value column for columnar data
			rowWise : if True row wise feature generation
			withLabel : True if each TS sequence is labeled
			prec : float output precision
			wsize : window size for data is single column
			retArr ; If True returns array otherwise delem separated string
		"""
		if dformat == "tabular":
			# tabluar with multiple values per row
			allrecs = list()
			for rec in fileRecGen(dfpath):
				frec = rec[:-1] if withLabel else rec
				frec = toFloatList(frec)

				if rowWise:
					#features per row
					features = self.__fft(frec, cutoff)
					feat = features if retArr else toStrFromList(features, prec)
					yield feat
				else:
					allrecs.extend(frec)
					
			if not rowWise:
				#all data
				features = self.__fft(allrecs, cutoff)
				feat = features if retArr else toStrFromList(features, prec)
				yield feat

		else:
			#columnar data
			dvalues = getFileColumnAsFloat(dfpath, vcol)
			
			#one value per window location
			if rowWise:
				#windowed
				slwin = SlidingWindow(dvalues, wsize)
				for wdata in slwin.windowGen():
					features = self.__fft(wdata, cutoff)
					feat = features if retArr else toStrFromList(features, prec)
					yield feat
			else:
				#all data
				features = self.__fft(dvalues, cutoff)
				feat = features if retArr else toStrFromList(features, prec)
				yield feat
				
	def __fft(self, data, cutoff):
		"""
		calculates FFT for each record
		
		Parameters
			data : input data list like
			cutoff : cutoff index
		"""
		ft = fft.rfft(np.array(data))
		ft =  np.abs(ft)
		return ft[:cutoff]
		

