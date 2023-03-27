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
from calendar import timegm
import math
from datetime import datetime
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *

"""
Time series generation for different kinds of time series 
"""


class TimeSeriesGenerator(object):
	def __init__(self, configFile):
		"""
		Initializer
		
		Parameters
			configFile : config file path
		"""
		defValues = {}
		defValues["window.size"] = (None, "missing time window size")
		defValues["window.samp.interval.type"] = ("fixed", None)
		defValues["window.samp.interval.params"] = (None, "missing time interval parameters")
		defValues["window.samp.align.unit"] = (None, None)
		defValues["window.time.unit"] = ("s", None)
		defValues["output.value.type"] = ("float", None)
		defValues["output.value.precision"] = (3, None)
		defValues["output.value.format"] = ("long", None)
		defValues["output.time.format"] = ("epoch", None)
		defValues["output.value.nsamples"] = (1, None)
		defValues["ts.base"] = ("mean", None)
		defValues["ts.base.params"] = (None,None)
		defValues["ts.trend"] = ("nothing", None)
		defValues["ts.trend.params"] = (None, None)
		defValues["ts.cycles"] = ("nothing", None)
		defValues["ts.cycle.year.params"] = (None, None)
		defValues["ts.cycle.week.params"] = (None, None)
		defValues["ts.cycle.day.params"] = (None, None)
		defValues["ts.random.params"] = (None, None)
		defValues["rw.init.value"] = (5.0, None)
		defValues["rw.range"] = (1.0, None)
		defValues["ar.params"] = (None, None)
		defValues["ar.seed"] = (None, None)
		defValues["ar.exp.param"] = (None, None)
		defValues["corr.file.path"] = (None, None)
		defValues["corr.file.col"] = (None, None)
		defValues["corr.scale"] = (1.0, None)
		defValues["corr.noise.stddev"] = (None, None)
		defValues["corr.lag"] = (0, None)
		defValues["ccorr.file.path"] = (None, None)
		defValues["ccorr.file.col"] = (None, None)
		defValues["ccorr.co.params"] = (None, None)
		defValues["ccorr.unco.params"] = (None, None)
		defValues["si.params"] = (None, None)
		defValues["motif.params"] = (None, None)
		defValues["anomaly.params"] = (None, None)

		self.config = Configuration(configFile, defValues)
		self.delim = ","
		
		#start time
		winSz = self.config.getStringConfig("window.size")[0]
		items = winSz.split("_")
		self.curTm, self.pastTm = pastTime(int(items[0]), items[1])
	
		#sample interval
		sampIntvType = self.config.getStringConfig("window.samp.interval.type")[0]
		sampIntv = self.config.getStringConfig("window.samp.interval.params")[0].split(self.delim)
		self.intvDistr = None
		self.sampIntv = None
		if sampIntvType == "fixed":
			items = sampIntv[0].split("_")
			ts = int(items[0])
			unit = items[1]
			self.sampIntv = timeToSec(ts, unit)
		elif sampIntvType == "random":
			assertEqual(len(sampIntv), 2, "invalid number of params for sample interval")
			siMean = float(sampIntv[0])
			siSd = float(sampIntv[1])
			self.intvDistr = NormalSampler(siMean,siSd)
		else:
			raise ValueError("invalid sampling interval type")
		

		#time alignment
		sampAlignUnit = self.config.getStringConfig("window.samp.align.unit")[0]
		if sampAlignUnit is not None:
			self.pastTm = timeAlign(self.pastTm, sampAlignUnit)	
	
		#output format
		self.tsValType = self.config.getStringConfig("output.value.type")[0]
		self.valPrecision = self.config.getIntConfig("output.value.precision")[0]
		self.tsTimeFormat = self.config.getStringConfig("output.time.format")[0]
		if self.tsValType == "int":
			self.ouForm = "{},{}"
		else:
			self.ouForm = "{},{:."  + str(self.valPrecision) + "f}"
			
		#long or short  i.e whole time series in one line format 
		self.oformat = self.config.getStringConfig("output.value.format")[0]
		
		# time unit
		timeUnit = self.config.getStringConfig("window.time.unit")[0]
		if timeUnit == "ms":
			self.curTm *= 1000
			self.pastTm *= 1000

		# anomaly params
		anParams = self.config.getStringListConfig("anomaly.params")[0]
		self.anGenerator = self.__createAnomalyGen(anParams) if anParams is not None else None

	def randGaussianGen(self):
		"""
		generates random gaussian time series
		
		Parameters
		"""
		distr = self.config.getFloatListConfig("gr.distr")[0]
		mean = distr[0]
		sd = distr[1]
		sampler = NormalSampler(mean, sd)
		sampTm = self.pastTm
		
		while (sampTm < self.curTm):
			curVal = sampler.sample()
			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = self.__getDateTime(sampTm, self.tsTimeFormat)
				
			rec = self.ouForm.format(dt, curVal)
			sampTm += self.sampIntv
			yield rec


	def nonParamDistrGen(self):
		"""
		generates non parametric distribution based time series

		Parameters
		"""
		distr = self.config.getFloatListConfig("npr.distr")[0]
		minVal = distr[0]
		bw = distr[1]
		dis = distr[2:]
		sampler = NonParamRejectSampler(minVal, bw, dis)
		sampler.sampleAsFloat()
		sampTm = self.pastTm
		
		while (sampTm < self.curTm):
			curVal = sampler.sample()
			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = self.__getDateTime(sampTm, self.tsTimeFormat)
				
			rec = ouForm.format(dt, curVal)
			sampTm += self.sampIntv
			yield rec


	def trendCycleNoiseGen(self):
		"""
		generates time series based on trend, seasonality and gaussian noise

		Parameters
		"""
		tsBaseType = self.config.getStringConfig("ts.base")[0]
		items = self.config.getStringConfig("ts.base.params")[0].split(self.delim)
		if tsBaseType == "mean":
			tsMean = float(items[0])
		elif tsBaseType == "ar":
			arParams = self.config.getFloatListConfig("ar.params")[0]
			hist = list()
		else:
			raise ValueError("invalid base type")
		
		tsTrendType = self.config.getStringConfig("ts.trend")[0]
		items = self.config.getStringConfig("ts.trend.params")[0].split(self.delim)
		if tsTrendType == "linear":
			tsTrendSlope = float(items[0])
		elif tsTrendType == "quadratic":
			tsTrendQuadParams = toFloatList(items)
		elif tsTrendType == "logistic":
			tsTrendLogParams = toFloatList(items)
		else:
			raise ValueError("invalid trend type")
		
		cycles = self.config.getStringListConfig("ts.cycles")[0]
		yearCycle = weekCycle = dayCycle = None
		for c in cycles:
			key = "ts.cycle." + c + ".params"
			cycleValues = self.config.getFloatListConfig(key)[0]
			if c == "year":
				yearCycle = cycleValues
			elif c == "week":
				weekCycle = cycleValues
			elif c == "day":
				dayCycle = cycleValues
			
		
		tsRandDistr = self. __genRandSampler()
	
		sampIntv = int(self.intvDistr.sample()) if self.intvDistr is not None else self.sampIntv
		sampTm = self.pastTm
		counter = 0

		#print(self.pastTm,  self.curTm, (self.curTm - self.pastTm), sampIntv)
		i = 0
		while (sampTm < self.curTm):
			curVal = 0
		
			#base
			if tsBaseType == "mean":
				#mean
				curVal = tsMean
			else:
				#auto regressive
				curVal = self.__arValue(arParams, hist) 	
					
			#random remainder
			curVal += tsRandDistr.sample() if tsRandDistr is not None else 0

			#update history
			if tsBaseType == "ar":
				hist.insert(0, curVal)
				hist.pop(len(hist) - 1)

			#trend
			if tsTrendType == "linear":
				curVal += counter * tsTrendSlope
			elif tsTrendType == "quadratic":
				curVal += tsTrendQuadParams[0] * counter + tsTrendQuadParams[1] * counter * counter
			elif tsTrendType == "logistic":
				ex = math.exp(-tsTrendLogParams[0] * counter)
				curVal += tsTrendLogParams[0] * (1.0 - ex) / (1.0 + ex)
			counter += 1
		
			#cycle
			if yearCycle:
				month = monthOfYear(sampTm)
				curVal += yearCycle[month]
			if weekCycle:
				day = dayOfWeek(sampTm)
				curVal += weekCycle[day]
			if dayCycle:
				hour = hourOfDay(sampTm)
				curVal += dayCycle[hour]

	
			#date time
			if self.tsTimeFormat == "epoch":
				dt = sampTm
			else:
				dt = datetime.fromtimestamp(sampTm)
				dt = dt.strftime("%Y-%m-%d %H:%M:%S")
			
			#value
			if self.tsValType == "int":
				curVal = int(curVal)

			rec = self.ouForm.format(dt, curVal)
	
			#next
			if self.intvDistr is not None:
				sampIntv = int(self.intvDistr.sample())
			sampTm += sampIntv
		
			yield rec

	def randomWalkGen(self):
		"""
		generates random wallk based time series

		Parameters
		"""
		initVal = config.getFloatConfig("rw.init.value")[0]
		ranRange = config.getFloatConfig("rw.range")[0]
		sampTm = self.pastTm
		curVal = initVal
		sampIntv = self.sampIntv
		while (sampTm < self.curTm):
			#next
			curVal += randomFloat(-ranRange, ranRange)
			if self.tsValType == "int":
				curVal = int(curVal)
			
			#date time
			dt = self.__getDateTime(sampTm, self.tsTimeFormat)
			rec = ouForm.format(dt, curVal)
			
			if self.intvDistr is not None:
				sampIntv = int(self.intvDistr.sample())
			sampTm += sampIntv

			yield rec

	def expAutRegGen(self):
		"""
		generates exponential smoothing auto regression based time series

		Parameters
		"""
		ap  = config.getFloatConfig("ar.exp.param")[0]
		iap = 1.0 - ap
		hist = config.getFloatListConfig("ar.seed")[0]
		
		# exponential ar parameters
		arParams = list()
		term = ap
		arParams.append(term)
		for _ in range(1, len(hist), 1):
			term *= iap
			arParams.append(term)
			
		for rec in self.__autRegGen(arParams):
			yield rec
						

	def genAutRegGen(self):
		"""
		generates generic auto regression based time series

		Parameters
		"""
		#user specified ar parameters
		arParams = config.getFloatListConfig("ar.params")[0]
		for rec in self.__autRegGen(arparams):
			yield rec

	def multSineGen(self, exscomp):
		"""
		generates mutiple sine function based time series

		Parameters
			exscomp : extra sine components
		"""
		siParams = self.config.getFloatListConfig("si.params")[0]	
		ocomps = None
		if exscomp != "none":
			addSine = toFloatList(exscomp.split(","))
			osiParams = siParams.copy()
			osiParams.extend(addSine)
			ocomps = self.__sinComponents(osiParams)
		
		comps = self.__sinComponents(siParams)
					
		#random component
		rsampler = self.__genRandSampler()
		oformat = self.config.getStringConfig("output.value.format")[0]
		nsamples = self.config.getIntConfig("output.value.nsamples")[0]
		
		# for each time series sample
		for i in range(nsamples):
			sampTm = self.pastTm
			values = list()
			
			#last sample different
			scomps = ocomps if i == nsamples - 1 and ocomps is not None else comps
			
			#generate one time series
			while (sampTm < self.curTm):
				val = self.__addSines(scomps, sampTm)
				val += rsampler.sample() if rsampler is not None else 0
				if self.tsValType == "int":
					val = int(val)
			
				if oformat == "long":	
					#multiple rec per time series
					dt = self.__getDateTime(sampTm, self.tsTimeFormat)
					rec = ouForm.format(dt, val)
					yield rec
				else:
					values.append(val)
				
				sampTm += self.sampIntv

			if oformat == "short":
				# one rec per time series
				svalues = toStrList(values, 3)
				li = ",".join(svalues)
				yield li
	
	def crossCorrGen(self):
		"""
		generates cros correlated time series

		Parameters
		"""
		refFile = self.config.getStringConfig("ccorr.file.path")[0]
		refCol = self.config.getIntConfig("ccorr.file.col")[0]
		cors = self.config.getFloatListConfig("ccorr.co.params")[0]

		uncors = self.config.getFloatListConfig("ccorr.unco.params")[0]		
		comps = self.__sinComponents(uncors) if uncors is not None else None
		
		rsampler = self.__genRandSampler()
		ouForm = "{:."  + str(self.valPrecision) + "f}"
		
		# iterate source file
		for rec in fileRecGen(refFile, ","):
			sampTm = int(rec[0])
			rval = float(rec[refCol])
			nrec = rec.copy()
			
			#multiple correlated columns
			for c in cors:
				cval = c * rval
				cval += rsampler.sample() if rsampler is not None else 0
				cval = str(int(cval)) if self.tsValType == "int" else self.ouForm.format(cval)
				nrec.append(cval)
			
			# add multi sine  components column	
			if comps is not None:
				cval = self.__addSines(comps, sampTm)
				cval += rsampler.sample() if rsampler is not None else 0
				cval = str(int(cval)) if self.tsValType == "int" else self.ouForm.format(cval)
				nrec.append(cval)
					
			nRec = ",".join(nrec)
			yield nRec

	def corrGen(self):
		"""
		generates correlated time series

		Parameters
		"""
		refFile = self.config.getStringConfig("corr.file.path")[0]
		refCol = self.config.getIntConfig("corr.file.col")[0]
		scale = self.config.getFloatConfig("corr.scale")[0]
		noiseSd = self.config.getFloatConfig("corr.noise.stddev")[0]
		lag = self.config.getIntConfig("corr.lag")[0]
		noiseDistr = NormalSampler(0,noiseSd)
		lCount = 0
		for rec in fileRecGen(refFile, ","):
			if lCount >= lag:
				val = float(rec[refCol]) * scale + noiseDistr.sample()
				val = "{:.3f}".format(val)
				rec[refCol] = val
				nRec = ",".join(rec)
				yield nRec
			lCount += 1
	
	def motifGen(self):
		"""
		generates given motif based time series

		Parameters
		"""
		params = self.config.getStringListConfig("motif.params")[0]
		fpath = params[0]
		cindex = int(params[0])
		mdata = getFileColumnAsFloat(fpath, cindex)
		mlen = len(mdata)
		mcnt = 0
		rsampler = self.__genRandSampler()

		sampTm = self.pastTm
		while (sampTm < self.curTm):
			curVal = mdata[mcnt]
			curVal += rsampler.sample() if rsampler is not None else 0
			mcnt = (mcnt + 1) % mlen
			
			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = self.__getDateTime(sampTm, self.tsTimeFormat)
				
			rec = self.ouForm.format(dt, curVal)
			sampTm += self.sampIntv
			yield rec

	def spikeGen(self):
		"""
		generates spike based time series

		Parameters
		"""
		params = self.config.getStringListConfig("spike.params")[0]
		
		#gap pameters
		gsampler =  self.__spikeSampler(params[0])	
		
		#width sampler
		wsampler =  self.__spikeSampler(params[1])
		
		#incr value  sampler
		ivsampler =  self.__spikeSampler(params[2], False)

		# random noise sampler
		rsampler = self.__genRandSampler()
		
		gap = gsampler.sample()
		sampTm = self.pastTm
		inSpike = False
		preVal = rsampler.sample()
		iga = 0
		isp = 0
		
		while (sampTm < self.curTm):
			if inSpike:	
				if isp <= hwidth:
					curVal = preVal + vinc
				else:
					curVal = preVal - vinc
				curVal += rsampler.sample() if rsampler is not None else 0
				preVal = curVal
				isp += 1
				if isp == width:
					inSpike = False
					gap = gsampler.sample()
			else:
				curVal = rsampler.sample() if rsampler is not None else 0
				preVal = curVal
				iga += 1
				if iga == gap:
					vinc = ivsampler.sample()
					width = wsampler.sample()
					hwidth = int((width + 1) / 2)
					isp = 0
					inPike = True
					
			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = self.__getDateTime(sampTm, self.tsTimeFormat)
				
			rec = self.ouForm.format(dt, curVal)
			sampTm += self.sampIntv
			yield rec
				
		
	def insertAnomalySeqGen(self, dfpath, prec):
		"""
		inserts anomaly sequence to an existing  time series

		Parameters
			dfpath : data file path
			prec : float precision
		"""
		i = 0
		atype = self.anGenerator.getType()
		abeg, aend = self.anGenerator.getRange()
		anVal = None
		anValLast = None
		for rec in fileRecGen(dfpath, self.delim):
			if i >= abeg:
				if i < aend:
					if atype == "multsine":
						#time stamp needed for multi sine 
						utcTm = time.strptime(rec[0], self.tsTimeFormat)
						epochTm = timegm(utcTm)
						anVal = self.anGenerator.sample(i, epochTm)
					else :
						anVal = self.anGenerator.sample(i) 
					anValLast = anVal
				else:
					if atype == "meanshift":
						anVal = anValLast
						
				val = float(rec[1]) + anVal
				rec[1] = formatFloat(prec, val)
			
			i += 1
			yield self.delim.join(rec)

	def __spikeSampler(self, params, asInt=True):
		"""
		generates sampler for spiky time series

		Parameters
			params : parameters
			asInt : True if too be sampled as int
		"""
		params = params[0].split(":")
		if params[0] == "uniforma":
			sampler = UniformNumericSampler(int(params[1]), int(params[2])) if asInt else UniformNumericSampler(float(params[1]), float(params[2]))
		else:
			sampler =  NormalSampler(float(params[1]), float(params[2]))
			if asInt:
				sampler.sampleAsIntValue()
		
		return sampler
		
	def __genRandSampler(self):
		"""
		generates random sampler
		
		Parameters
		"""
		items = self.config.getFloatListConfig("ts.random.params")[0]
		rsampler = None
		if items is not None:
			tsRandMean = items[0]
			tsRandStdDev = items[1]
			rsampler = NormalSampler(tsRandMean, tsRandStdDev)
		return rsampler	
				
	def __getDateTime(self, tm, tmFormat):
		"""
		returns either epoch time for formatted date time
		
		Parameters
			tm : epoch time
			tmFormat : time format
		"""
		if tmFormat == "epoch":
			dt = tm
		else:
			dt = datetime.fromtimestamp(tm)
			dt = dt.strftime("%Y-%m-%d %H:%M:%S")
		return dt

	def __autRegGen(self, arParams):
		"""
		generates auto regression based time series

		Parameters
			arParams : auto regression parameters
		"""
		#ar parameters
		hist = config.getFloatListConfig("ar.seed")[0]
		
		#random component
		rsampler = self.__genRandSampler()

		sampTm = self.pastTm
		while (sampTm < self.curTm):
			curVal = self.__arValue(arParams, hist) 	
			curVal += rsampler.sample() if rsampler is not None else 0
			hist.insert(0, curVal)
			hist.pop(len(hist))

			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = self._getDateTime(sampTm, self.tsTimeFormat)
				
			sampTm += self.sampIntv
			rec = self.ouForm.format(dt, curVal)
			yield rec

	def __arValue(self, arParams, hist):
		"""
		auto regressed value
			
		Parameters
			arParams : auto regression parameters
			hist : history
		"""
		return np.dot(arParams, hist[:len(arParams)])

	def __sinComponents(self, params):
		"""
		returns list sine components

		Parameters
			params : parameters for sine function
		"""
		comps = list()
		for i in range(0, len(params), 2):
			amp = params[i]
			per = params[i + 1]
			phase = randomFloat(0, 2.0 * math.pi)
			co = (amp, per, phase)
			comps.append(co)
		return comps

	def __addSines(self, comps, sampTm):
		"""
		adds multiple sine comopnents
		
		Parameters
			comps : sine component parameters
			sampTm : epoch time
		"""
		val = 0
		for c in comps:
			t = 2.0 * math.pi * (sampTm % c[1]) / c[1]
			val += c[0] * math.sin(c[2] + t)
		return val

	def __createAnomalyGen(self, anParams):
		"""
		adds multiple sine comopnents
		
		Parameters
			anParams : anomaly params list
		"""
		atype = anParams[0]
		if atype == "random":
			agen = RandomAnomalyGenerator(anParams)
		elif atype == "multsine":
			agen = MultSineAnomalyGenerator(anParams)
		elif atype == "motif":
			agen = MotifAnomalyGenerator(anParams)
		elif atype == "meanshift":
			agen = MeanShiftAnomalyGenerator(anParams)
		else:
			exitWithMsg("invalid anomaly type")	
		
		return agen

class AnomalyGenerator(object):
	def __init__(self, params):
		"""
		Initializer for anomaly base class
		
		Parameters
			params : parameter list
		"""
		self.atype = params[0]
		self.beg = int(params[1])
		self.end = int(params[2])

	def getType(self):
		"""
		get type of anomaly
		
		Parameters
		"""
		return self.atype
		
	def getRange(self):
		"""
		get range of anomaly
		
		Parameters
		"""
		r = (self.beg, self.end)
		return r
			
			
class RandomAnomalyGenerator(AnomalyGenerator):
	def __init__(self, params):
		"""
		Initializer for random noise anomaly
		
		Parameters
			params : parameter list
		"""
		assertEqual(len(params), 4, "invalid number of parameters")
		sd = float(params[3])
		self.rsampler = NormalSampler(0, sd)
		super(RandomAnomalyGenerator, self).__init__(params)
		
	def sample(self, pos):
		"""
		samples anomalous value
		
		Parameters
			pos : pos in time series
		"""
		sval = self.rsampler.sample() if pos >= self.beg and pos < self.end else 0
		return sval

class MotifAnomalyGenerator(AnomalyGenerator):
	def __init__(self, params):
		"""
		Initializer for motif based anomaly generator
		
		Parameters
			parameters : parameter list
		"""
		assertEqual(len(params), 6, "invalid number of parameters")
		sd = float(params[3])
		fpath = params[4]
		cindex = int(params[5])
		self.mdata = getFileColumnAsFloat(fpath, cindex)
		self.mlen = len(mdata)
		self.mcnt = 0
		
		self.rsampler = NormalSampler(0, sd) if sd > 0 else None
		super(MotifAnomalyGenerator, self).__init__(params)

	def sample(self, pos):
		"""
		samples anomalous value
		
		Parameters
			pos : pos in time series
		"""
		sval = 0
		if pos >= self.beg and pos < self.end:
			sval = self.mdata[self.mcount]
			sval += self.rsampler.sample() if self.rsampler is not None else 0
			self.mcnt = (self.mcnt + 1) % self.mlen
		return sval

class MultSineAnomalyGenerator(AnomalyGenerator):
	def __init__(self, params):
		"""
		Initializer for multiple sine function anomaly
		
		Parameters
			parameters : parameter list
		"""
		sd = float(params[3])
		self.rsampler = NormalSampler(0, sd) if scd > 0 else None
		self.scomps = self.__sinComponents(params[4:])
		super(MultSineAnomalyGenerator, self).__init__(params)
		
	def sample(self, pos, sampTm):
		"""
		samples anomalous value
		
		Parameters
			pos : pos in time series
			sampTm : sample time
		"""
		sval = 0
		if pos >= self.beg and pos < self.end:
			for c in self.scomps:
				t = sampTm + c[2]
				t = 2.0 * math.pi * (t % c[1]) / c[1]
				sval += c[0] * math.sin(t)
			sval += self.rsampler.sample() if self.rsampler is not None else 0
		return sval

	def __sinComponents(self, params):
		"""
		returns list sine components

		Parameters
			params : parameters for sine function
		"""
		comps = list()
		for i in range(0, len(params), 2):
			amp = params[i]
			per = params[i + 1]
			phase = randomFloat(0, 2.0 * math.pi)
			co = (amp, per, phase)
			comps.append(co)
		return comps

class MeanShiftAnomalyGenerator(AnomalyGenerator):
	def __init__(self, params):
		"""
		Initializer for mean shift based anomaly generator
		
		Parameters
			parameters : parameter list
		"""
		assertEqual(len(params), 5, "invalid number of parameters")
		self.shift = float(params[3])
		self.cshift = 0
		sd = float(params[4])
		self.rsampler = NormalSampler(0, sd) if sd > 0 else None
		super(MeanShiftAnomalyGenerator, self).__init__(params)

	def sample(self, pos):
		"""
		samples anomalous value
		
		Parameters
			pos : pos in time series
		"""
		sval = 0
		if pos >= self.beg and pos < self.end:
			self.cshift += self.shift
			sval = self.cshift
			sval += self.rsampler.sample() if self.rsampler is not None else 0
		
		#persist shift after anomaly range
		elif pos >= self.end:
			sval = self.cshift
		return sval