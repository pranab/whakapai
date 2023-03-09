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
from zaman.proph import *
from zaman.tsutil import *
"""
driver code for time series
"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cnfpath', type=str, default = "none", help = "prophet config file path")
	parser.add_argument('--cnfparams', type=str, default = "none", help = "overriding config paramas")
	parser.add_argument('--pdays', type=int, default = 10, help = "no of days in past for begin date")
	parser.add_argument('--trend', type=str, default = "true", help = "if there is trend in generated data")
	parser.add_argument('--cycle', type=str, default = "true", help = "if there is cycle in generated data")
	parser.add_argument('--dfpath', type=str, default = "none", help = "data file file path")
	args = parser.parse_args()
	op = args.op

	# classifier
	if op == "train" or op ==  "forecast" or op == "validate":
		forecaster = ProphetForcaster(args.cnfpath, None, None)

	# override config param
	if args.cnfparams != "none":
		#parameters over riiding config file
		params = args.cnfparams.split(",")
		for pa in params:
			items = pa.split("=")
			forecaster.setConfigParam(items[0], items[1])

	# execute	
	if op == "train":
		""" train """
		forecaster.train()
		
	elif op == "forecast":
		""" forecast """
		forecaster.forecast()
		
	elif op == "validate":
		""" validate """
		forecaster.validate()
		
	elif op == "appent":
		""" approximate entropy """
		ae = appEntropy([args.dfpath, 1], 3, 3)
		print(ae)

	elif op == "comp":
		""" components """
		components([args.dfpath, 1], "additive", 7, True, True)
		
	elif op == "acorr":
		""" auto correlation """
		doPlot([args.dfpath, 1])
		autoCorr([args.dfpath, 1], True, 20)
		
	elif op == "genp":
		""" generate power usage data """
		startNumDaysPast = args.pdays
		withTrend =  args.trend == "true"
		withSeasonalCycle = args.cycle == "true"
		#print("components {} {}".format(withTrend, withSeasonalCycle))
		
		curTime = int(time.time())
		startTime = curTime - (startNumDaysPast  + 1) * secInDay
		startTime = int((startTime / secInHour)) * secInHour
		intv = 60 * 60
		sampTime = startTime
		
		mean = 3.0
		yTrendDiff = 10.0
		trend = yTrendDiff / (365 * 24)
		yearCycle = [0.75, 0.48, 0.22, -0.6, -0.08, 0.19, 0.40, 0.68, 0.41, 0.12, 0.39, .72]
		dayCycle = [-0.10, -0.12, -0.16, -0.24, -0.28, -0.13, -0.08, 0.12, 0.25, 0.37, 0.45,\
			0.53, 0.42, 0.34, 0.26, 0.21, 0.16, 0.12, 0.10, 0.06, -0.01, -0.05, -0.08, -0.10]
		noiseDistr = GaussianRejectSampler(0, .05)


		cureTrend = 0
		while (sampTime < curTime):
			usage = mean + (cureTrend if withTrend else 0)

			if withSeasonalCycle:
				secIntoYear = sampTime % secInYear
				month = int(secIntoYear / secInMonth)
				#print("month {}".format(month))
				usage += yearCycle[month]

				hourIntoDay = int((sampTime % secInDay) / secInHour)
				#print("hour {}".format(hourIntoDay))
				usage += dayCycle[hourIntoDay]

			usage += noiseDistr.sample()

			dt = datetime.fromtimestamp(sampTime)
			dt = dt.strftime("%Y-%m-%d %H:%M:%S")
			print ("{},{:.3f}".format(dt, usage))

			cureTrend += trend
			sampTime += secInHour

	elif op == "gend":
		""" generate retail product demand data """
		startNumDaysPast = args.pdays
		withTrend =  args.trend == "true"
		withSeasonalCycle = args.cycle == "true"
		#print("components {} {}".format(withTrend, withSeasonalCycle))
		
		curTime = int(time.time())
		startTime = curTime - (startNumDaysPast  + 1) * secInDay
		startTime = int((startTime / secInDay)) * secInDay
		intv = 60 * 60 * 24
		sampTime = startTime
		
		mean = 200.0
		yTrendDiff = 20.0
		trend = yTrendDiff / 365
		weekCycle = [-5.0, -10.0, -20.0, -10.0, 10.0, 20.0, 25.0]
		noiseDistr = GaussianRejectSampler(0, 5.0)


		curTrend = 0
		while (sampTime < curTime):
			usage = mean + (curTrend if withTrend else 0)

			if withSeasonalCycle:
				secIntoWeek = sampTime % secInWeek
				day = int(secIntoWeek / secInDay)
				#print("month {}".format(month))
				usage += weekCycle[day]

			usage += noiseDistr.sample()

			dt = datetime.fromtimestamp(sampTime)
			dt = dt.strftime("%Y-%m-%d %H:%M:%S")
			print ("{},{:.3f}".format(dt, usage))

			curTrend += trend
			sampTime += secInDay

	else:
		raise ValueError("invalid command")
		



