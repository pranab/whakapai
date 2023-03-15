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
		defValues["rnp.distr"] = (None, None)
		defValues["ts.base"] = ("mean", None)
		defValues["ts.base.params"] = (None, "missing time series base parameters")
		defValues["ts.trend"] = ("nothing", None)
		defValues["ts.trend.params"] = (None, None)
		defValues["ts.cycles"] = ("nothing", None)
		defValues["ts.cycle.year.params"] = (None, None)
		defValues["ts.cycle.week.params"] = (None, None)
		defValues["ts.cycle.day.params"] = (None, None)
		defValues["ts.random"] = (True, None)
		defValues["ts.random.params"] = (None, None)
		defValues["rw.init.value"] = (5.0, None)
		defValues["rw.range"] = (1.0, None)
		defValues["ar.params"] = (None, None)
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
		defValues["ol.percent"] = (5, None)
		defValues["ol.distr"] = (None, "missing outlier distribution")

		self.config = Configuration(configFile, defValues)

		#start time
		winSz = config.getStringConfig("window.size")[0]
		items = winSz.split("_")
		self.curTm, self.pastTm = pastTime(int(items[0]), items[1])
	
		#sample interval
		sampIntvType = config.getStringConfig("window.samp.interval.type")[0]
		sampIntv = config.getStringConfig("window.samp.interval.params")[0].split(delim)
		self.intvDistr = None
		self.sampIntv = None
		if sampIntvType == "fixed":
			self.sampIntv = int(sampIntv[0])
		elif sampIntvType == "random":
			siMean = float(sampIntv[0])
			siSd = float(sampIntv[1])
			self.intvDistr = NormalSampler(siMean,siSd)
		else:
			raise ValueError("invalid sampling interval type")
		

		#time alignment
		sampAlignUnit = config.getStringConfig("window.samp.align.unit")[0]
		if sampAlignUnit is not None:
			self.pastTm = timeAlign(self.pastTm, sampAlignUnit)	
	
		#output format
		self.tsValType = config.getStringConfig("output.value.type")[0]
		self.valPrecision = config.getIntConfig("output.value.precision")[0]
		self.tsTimeFormat = config.getStringConfig("output.time.format")[0]
		if self.tsValType == "int":
			self.ouForm = "{},{}"
		else:
			self.ouForm = "{},{:."  + str(self.valPrecision) + "f}"

		# time unit
		timeUnit = config.getStringConfig("window.time.unit")[0]
		if timeUnit == "ms":
			self.curTm *= 1000
			self.pastTm *= 1000


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
			dt = getDateTime(sampTm, self.tsTimeFormat)
				
			rec = self.ouForm.format(dt, curVal)
			sampTm += sampIntv
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
			dt = getDateTime(sampTm, self.tsTimeFormat)
				
			rec = ouForm.format(dt, curVal)
			sampTm += sampIntv
			yield rec


	def trendCycleNoiseGen(self):
		"""
		generates time series based on trend, seasonality and gaussian noise

		Parameters
		"""
		tsBaseType = self.config.getStringConfig("ts.base")[0]
		items = self.config.getStringConfig("ts.base.params")[0].split(delim)
		if tsBaseType == "mean":
			tsMean = float(items[0])
		elif tsBaseType == "ar":
			arParams = self.config.getFloatListConfig("ar.params")[0]
			hist = list()
		else:
			raise ValueError("invalid base type")
		
		tsTrendType = self.config.getStringConfig("ts.trend")[0]
		items = self.config.getStringConfig("ts.trend.params")[0].split(delim)
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
			
		
		tsRandom = self.config.getBooleanConfig("ts.random")[0]
		tsRandDistr = None
		if tsRandom:
			tsRandDistr = self. __genRandSampler()
	
		if self.intvDistr is not None:
			sampIntv = int(self.intvDistr.sample())
	
		sampTm = self.pastTm
		counter = 0

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
			if tsRandDistr is not None:
				curVal += tsRandDistr.sample()

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
		
		while (sampTm < self.curTm):
			#value
			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, self.tsTimeFormat)
			
			rec = ouForm.format(dt, curVal)
			
			#next
			curVal += randomFloat(-ranRange, ranRange)
			
			if self.intvDistr is not None:
				sampIntv = int(self.intvDistr.sample())
			sampTm += sampIntv

			yield rec

	def autRegGen():
		"""
		generates auto regression based time series

		Parameters
		"""
		#ar parameters
		arParams = config.getFloatListConfig("ar.params")[0]
		hist = list()
		for i in range(len(arParams) - 1):
			hist.append(0.0)
		
		#random component
		rsampler = self.__genRandSampler()

		sampTm = self.pastTm
		i = 0		
		while (sampTm < self.curTm):
			curVal = self.__arValue(arParams, hist) 	
			curVal += rsampler.sample()
			hist.insert(0, curVal)
			hist.pop(len(hist) - 1)

			if self.tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, self.tsTimeFormat)
				
			sampTm += self.sampIntv
			i += 1
			if i > 5:	
				rec = self.ouForm.format(dt, curVal)
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
			ocomps = sinComponents(osiParams)
		
		comps = sinComponents(siParams)
					
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
				val = addSines(scomps, sampTm)
				val += rsampler.sample()
				if self.tsValType == "int":
					val = int(val)
			
				if oformat == "long":	
					#multiple rec per time series
					dt = getDateTime(sampTm, self.tsTimeFormat)
					rec = ouForm.format(dt, val)
					yield rec
				else:
					values.append(val)
				
				sampTm += sampIntv

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
		comps = sinComponents(uncors) if uncors is not None else None
		
		rsampler = self.__genRandSampler
		ouForm = "{:."  + str(self.valPrecision) + "f}"
		
		# iterate source file
		for rec in fileRecGen(refFile, ","):
			sampTm = int(rec[0])
			rval = float(rec[refCol])
			nrec = rec.copy()
			
			#multiple correlated columns
			for c in cors:
				cval = c * rval + rsampler.sample()
				cval = str(int(cval)) if self.tsValType == "int" else self.ouForm.format(cval)
				nrec.append(cval)
			
			# add multi sine  components column	
			if comps is not None:
				cval = addSines(comps, sampTm) + rsampler.sample()
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
		pass
		
	def insertAnomalySeq(self):
		"""
		inserts anomaly sequence to an existing  time series

		Parameters
		"""
		pass
	
		
	def __genRandSampler(self):
		"""
		generates random sampler
		
		Parameters
		"""
		items = self.config.getFloatListConfig("ts.random.params")[0]
		tsRandMean = items[0]
		tsRandStdDev = items[1]
		rsampler = NormalSampler(tsRandMean, tsRandStdDev)
		return rsampler	
				
	def __getDateTime(tm, tmFormat):
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

	def __arValue(arParams, hist):
		"""
		auto regressed value
			
		Parameters
			arParams : auto regression parameters
			hist : history
		"""
		val = 0.0
		for i in range(len(arParams)):
			if i == 0:
				val = arParams[i]
			else:
				val += arParams[i] * hist[i-1]
		return val

	def __sinComponents(params):
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

	def __addSines(comps, sampTm):
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



			
	

