#!/usr/local/bin/python3

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

import os
import sys
from random import randint
import time
import math
from datetime import datetime
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *

"""
Time series generation for general time series with optional trend, cycles and random remainder 
components and random walk time series 
"""

def loadConfig(configFile):
	"""
	load config file
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
	

	config = Configuration(configFile, defValues)
	return config

def getDateTime(tm, tmFormat):
	"""
	returns either epoch time for formatted date time
	"""
	if tmFormat == "epoch":
		dt = tm
	else:
		dt = datetime.fromtimestamp(tm)
		dt = dt.strftime("%Y-%m-%d %H:%M:%S")
	return dt

def arValue(arParams, hist):
	"""
	auto regressed value
	"""
	val = 0.0
	for i in range(len(arParams)):
		if i == 0:
			val = arParams[i]
		else:
			val += arParams[i] * hist[i-1]
	return val

def sinComponents(params):
	"""
	returns list sine components
	"""
	comps = list()
	for i in range(0, len(params), 2):
		amp = params[i]
		per = params[i + 1]
		phase = randomFloat(0, 2.0 * math.pi)
		co = (amp, per, phase)
		comps.append(co)
	return comps

def addSines(comps, sampTm):
	"""
	adds multiple sine comopnents
	"""
	val = 0
	for c in comps:
		t = 2.0 * math.pi * (sampTm % c[1]) / c[1]
		val += c[0] * math.sin(c[2] + t)
	return val
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "none", help = "config file path")
	parser.add_argument('--ocfpath', type=str, default = "none", help = "overloaded config file path")
	parser.add_argument('--exscomp', type=str, default = "none", help = "extra sine components")
	args = parser.parse_args()
	op = args.op
	
	#configuration
	confFile = args.cfpath
	config = loadConfig(confFile)
	delim = ","
	
	#override config
	ocfpath = args.ocfpath
	if ocfpath != "none":
		config.override(ocfpath)
	
	#start time
	winSz = config.getStringConfig("window.size")[0]
	items = winSz.split("_")
	curTm, pastTm = pastTime(int(items[0]), items[1])
	
	#sample interval
	sampIntvType = config.getStringConfig("window.samp.interval.type")[0]
	sampIntv = config.getStringConfig("window.samp.interval.params")[0].split(delim)
	intvDistr = None
	if sampIntvType == "fixed":
		sampIntv = int(sampIntv[0])
	elif sampIntvType == "random":
		siMean = float(sampIntv[0])
		siSd = float(sampIntv[1])
		intvDistr = GaussianRejectSampler(siMean,siSd)
	else:
		raise ValueError("invalid sampling interval type")
		
	#time alignment
	sampAlignUnit = config.getStringConfig("window.samp.align.unit")[0]
	if sampAlignUnit is not None:
		pastTm = timeAlign(pastTm, sampAlignUnit)	
	
	#output format
	tsValType = config.getStringConfig("output.value.type")[0]
	valPrecision = config.getIntConfig("output.value.precision")[0]
	tsTimeFormat = config.getStringConfig("output.time.format")[0]
	if tsValType == "int":
		ouForm = "{},{}"
	else:
		ouForm = "{},{:."  + str(valPrecision) + "f}"
		
	timeUnit = config.getStringConfig("window.time.unit")[0]
	if timeUnit == "ms":
		curTm *= 1000
		pastTm *= 1000

	#gaussian random
	if op == "rg":
		distr = config.getFloatListConfig("gr.distr")[0]
		mean = distr[0]
		sd = distr[1]
		sampler = NormalSampler(mean, sd)
		sampTm = pastTm
		
		while (sampTm < curTm):
			curVal = sampler.sample()
			if tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, tsTimeFormat)
				
			print(ouForm.format(dt, curVal))
			sampTm += sampIntv

	
	#non parametric random
	elif op == "rnp":
		distr = config.getFloatListConfig("npr.distr")[0]
		minVal = distr[0]
		bw = distr[1]
		dis = distr[2:]
		sampler = NonParamRejectSampler(minVal, bw, dis)
		sampler.sampleAsFloat()
		sampTm = pastTm
		
		while (sampTm < curTm):
			curVal = sampler.sample()
			if tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, tsTimeFormat)
				
			print(ouForm.format(dt, curVal))
			sampTm += sampIntv
	
	#generic time series with trend, cycle, gaussian remainder
	elif op == "gen":
		tsBaseType = config.getStringConfig("ts.base")[0]
		items = config.getStringConfig("ts.base.params")[0].split(delim)
		if tsBaseType == "mean":
			tsMean = float(items[0])
		elif tsBaseType == "ar":
			arParams = config.getFloatListConfig("ar.params")[0]
			hist = list()
		else:
			raise ValueError("invalid base type")
		
		tsTrendType = config.getStringConfig("ts.trend")[0]
		items = config.getStringConfig("ts.trend.params")[0].split(delim)
		if tsTrendType == "linear":
			tsTrendSlope = float(items[0])
		elif tsTrendType == "quadratic":
			tsTrendQuadParams = toFloatList(items)
		elif tsTrendType == "logistic":
			tsTrendLogParams = toFloatList(items)
		else:
			raise ValueError("invalid trend type")
		
		cycles = config.getStringListConfig("ts.cycles")[0]
		yearCycle = weekCycle = dayCycle = None
		for c in cycles:
			key = "ts.cycle." + c + ".params"
			cycleValues = config.getFloatListConfig(key)[0]
			if c == "year":
				yearCycle = cycleValues
			elif c == "week":
				weekCycle = cycleValues
			elif c == "day":
				dayCycle = cycleValues
			
		
		tsRandom = config.getBooleanConfig("ts.random")[0]
		if tsRandom:
			items = config.getStringConfig("ts.random.params")[0].split(delim)
			tsRandMean = float(items[0])
			tsRandStdDev = float(items[1])
			tsRandDistr = NormalSampler(tsRandMean,tsRandStdDev)
		
	
		if intvDistr:
			sampIntv = int(intvDistr.sample())
	
		sampTm = pastTm
		counter = 0

		while (sampTm < curTm):
			curVal = 0
		
			#base
			if tsBaseType == "mean":
				#mean
				curVal = tsMean
			else:
				#auto regressive
				curVal = arValue(arParams, hist) 	
					
			#random remainder
			if tsRandStdDev:
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
			if tsTimeFormat == "epoch":
				dt = sampTm
			else:
				dt = datetime.fromtimestamp(sampTm)
				dt = dt.strftime("%Y-%m-%d %H:%M:%S")
		
		
			#value
			if tsValType == "int":
				curVal = int(curVal)

			print(ouForm.format(dt, curVal))
	
			#next
			if intvDistr:
				sampIntv = int(intvDistr.sample())
			sampTm += sampIntv
	
	#random walk time series		
	elif op == "rw":
		initVal = config.getFloatConfig("rw.init.value")[0]
		ranRange = config.getFloatConfig("rw.range")[0]
		sampTm = pastTm
		curVal = initVal
		
		while (sampTm < curTm):
			#value
			if tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, tsTimeFormat)
			
			print(ouForm.format(dt, curVal))
			
			#next
			curVal += randomFloat(-ranRange, ranRange)
			
			if intvDistr:
				sampIntv = int(intvDistr.sample())
			sampTm += sampIntv
	
	#auto regressive		
	elif op == "ar":
		
		#ar parameters
		arParams = config.getFloatListConfig("ar.params")[0]
		hist = list()
		for i in range(len(arParams) - 1):
			hist.append(0.0)
		
		#random component
		items = config.getStringConfig("ts.random.params")[0].split(delim)
		tsRandMean = float(items[0])
		tsRandStdDev = float(items[1])
		rsampler = NormalSampler(tsRandMean, tsRandStdDev)

		sampTm = pastTm
		i = 0		
		while (sampTm < curTm):
			curVal = arValue(arParams, hist) 	
			curVal += rsampler.sample()
			hist.insert(0, curVal)
			hist.pop(len(hist) - 1)

			if tsValType == "int":
				curVal = int(curVal)
				
			#date time
			dt = getDateTime(sampTm, tsTimeFormat)
				
			if i > 5:	
				print(ouForm.format(dt, curVal))
			sampTm += sampIntv
			i += 1
	
	# generates combination of multiple sinusoidal
	elif op == "sine":
			
		siParams = config.getFloatListConfig("si.params")[0]	
		ocomps = None
		exscomp = args.exscomp
		if exscomp != "none":
			addSine = toFloatList(exscomp.split(","))
			osiParams = siParams.copy()
			osiParams.extend(addSine)
			ocomps = sinComponents(osiParams)
		
		comps = sinComponents(siParams)
					
		#random component
		items = config.getFloatListConfig("ts.random.params")[0]
		tsRandMean = items[0]
		tsRandStdDev = items[1]
		rsampler = NormalSampler(tsRandMean, tsRandStdDev)
		oformat = config.getStringConfig("output.value.format")[0]
		nsamples = config.getIntConfig("output.value.nsamples")[0]
		
		for i in range(nsamples):
			sampTm = pastTm
			values = list()
			scomps = ocomps if i == nsamples - 1 and ocomps is not None else comps
			while (sampTm < curTm):
				val = addSines(scomps, sampTm)
				val += rsampler.sample()
				if tsValType == "int":
					val = int(val)
			
				if oformat == "long":	
					#date time
					dt = getDateTime(sampTm, tsTimeFormat)
					print(ouForm.format(dt, val))
				else:
					values.append(val)
				
				sampTm += sampIntv

			if oformat == "short":
				svalues = toStrList(values, 3)
				li = ",".join(svalues)
				print(li)
			
			
	# generates cross correlated time series
	elif op == "ccorr":
		refFile = config.getStringConfig("ccorr.file.path")[0]
		refCol = config.getIntConfig("ccorr.file.col")[0]
		cors = config.getFloatListConfig("ccorr.co.params")[0]

		uncors = config.getFloatListConfig("ccorr.unco.params")[0]		
		comps = sinComponents(uncors) if uncors is not None else None
		
		items = config.getFloatListConfig("ts.random.params")[0]
		tsRandMean = items[0]
		tsRandStdDev = items[1]
		rsampler = NormalSampler(tsRandMean, tsRandStdDev)

		ouForm = "{:."  + str(valPrecision) + "f}"
		for rec in fileRecGen(refFile, ","):
			sampTm = int(rec[0])
			rval = float(rec[refCol])
			nrec = rec.copy()
			for c in cors:
				cval = c * rval + rsampler.sample()
				cval = str(int(cval)) if tsValType == "int" else ouForm.format(cval)
				nrec.append(cval)
				
			if comps is not None:
				cval = addSines(comps, sampTm) + rsampler.sample()
				cval = str(int(cval)) if tsValType == "int" else ouForm.format(cval)
				nrec.append(cval)
					
			nRec = ",".join(nrec)
			print(nRec)
	
	# generates correlated time series
	elif op == "corr":
		refFile = config.getStringConfig("corr.file.path")[0]
		refCol = config.getIntConfig("corr.file.col")[0]
		scale = config.getFloatConfig("corr.scale")[0]
		noiseSd = config.getFloatConfig("corr.noise.stddev")[0]
		lag = config.getIntConfig("corr.lag")[0]
		noiseDistr = GaussianRejectSampler(0,noiseSd)
		lCount = 0
		for rec in fileRecGen(refFile, ","):
			if lCount >= lag:
				val = float(rec[refCol]) * scale + noiseDistr.sample()
				val = "{:.3f}".format(val)
				rec[refCol] = val
				nRec = ",".join(rec)
				print(nRec)
			lCount += 1

	#add outlier
	elif op == "aol":
		pass
			
	else:
		raise ValueError("ivalid time series type")
			
	

