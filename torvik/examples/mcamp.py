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

# Package imports
import os
import sys
import random
import statistics 
import numpy as np
import matplotlib.pyplot as plt 
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.daexp import *
from matumizi.sampler import *
from torvik.tnn import *

"""
AB test simulation with counterfactuals
"""

def loadData(model, dataFile):
	"""
	loads data

	Parameters
		model: regression model
		dataFile: data file path
	"""
	# parameters
	fieldIndices =  model.config.getIntListConfig("train.data.fields")[0]
	featFieldIndices = model.config.getIntListConfig("train.data.feature.fields")[0]
	#data
	(data, featData) = loadDataFile(dataFile, ",", fieldIndices, featFieldIndices)
	return featData.astype(np.float32)


def cntfactual(model, dataFile,  cfindxs, cfvals):
	"""
	get average target values with intervened column value
		
	Parameters
		model: regression model
		dataFile: data file path
		nsplit : num of splits
		cindex : intervened column index
		cvalue : intervened column value
	"""	
	#train or restore model
	useSavedModel = model.config.getBooleanConfig("predict.use.saved.model")[0]
	if useSavedModel:
		FeedForwardNetwork.restoreCheckpt(model)
	else:
		FeedForwardNetwork.batchTrain(model) 

	#feature data
	featData = loadData(model, dataFile)
	
	#scale all data
	scalingMethod = model.config.getStringConfig("common.scaling.method")[0]
	if (model.config.getStringConfig("common.preprocessing")[0] == "scale"):
		featData = scaleData(featData, scalingMethod)
	
	
	# intervene all columns
	for i, v in zip(cfindxs, cfvals):
		#intervened column values
		fc = featData[:,i]

		#scale intervened values
		if scalingMethod == "zscale":
			me = np.mean(fc)
			sd = np.std(fc)
			#print("me {:.3f}  sd {:.3f}".format(me, sd))
			scvalue = (v - me) / sd
		elif scalingMethod == "minmax":
			vmin = fc.min()
			vmax = fc.max()
			#print("vmin {:.3f}  vmax {:.3f}".format(vmin, vmax))
			vdiff = vmax - vmin
			scvalue = (v - vmin) / vdiff
		else:
			raise ValueError("invalid scaling method")
		
		
		#interven
		#print(featData[:5,:])
		featData[:,i] = scvalue
	
	
	#predict with intervened values
	model.eval()
	
	#print(featData[:5,:])
	tfeatData = torch.from_numpy(featData[:,:])
	yPred = model(tfeatData)
	yPred = yPred.data.cpu().numpy()
	#print(yPred)
	yPred = yPred[:,0]
	#print(yPred[:5])
	av = yPred.mean()
	
	if len(cfindxs) > 0:
		print("intervened values {}\tav xaction amount {:.2f}".format(str(cfvals), av))
	else:
		print("non intervened \tav xaction amount {:.2f}".format(av))
	

def setcamp(rec):
	"""
	sets campaign flag
		
	Parameters
		rec: record
	"""
	rec[5] = "1"
	return rec
	
def setexp(rec):
	"""
	sets expense
		
	Parameters
		rec: record
	"""
	#income group and campaign target
	if rec[1] == "1" and rec[5] == "1":
		exp = float(rec[-1])
		exp *= randomFloat(1.20, 1.25)
		rec[-1] = formatFloat(2, exp)
	return rec

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--genconf', type=str, default = "", help = "data gennerator config file")
	parser.add_argument('--nsamp', type=int, default = 1000, help = "no of samples to generate")
	parser.add_argument('--incid', type=str, default = "false", help = "include cust ID")
	parser.add_argument('--mlconf', type=str, default = "", help = "ML config file path")
	parser.add_argument('--cffile', type=str, default = "", help = "conterfactual test data file path")
	parser.add_argument('--cfindex', type=str, default = "", help = "coubterfactual column indexes")
	parser.add_argument('--cfval', type=str, default = "", help = "coubterfactual column values")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":
		"""  generate data """
		dgen = RegressionDataGenerator(args.genconf)
		for _ in range(args.nsamp):
			s = dgen.sample()
			pv = toStrFromList(s[0], 2)
			pv = pv + "," + toStr(s[1], 2)
			if args.incid == "true":
				pv = genID(10) + "," + pv
			print(pv)

	elif op == "setcamp":
		""" sets campaign flag """
		cfdatafp = args.cffile
		recs = mutateFileLines(cfdatafp, setcamp)
		for r in recs:
			print(",".join(r))
		
	elif op == "setexp":
		""" increase transaction amount based on income group and campaign target """
		cfdatafp = args.cffile
		recs = mutateFileLines(cfdatafp, setexp)
		for r in recs:
			print(",".join(r))

	elif op == "train":
		""" train  model """
		prFile = args.mlconf
		regressor = FeedForwardNetwork(prFile)
		regressor.buildModel()
		FeedForwardNetwork.batchTrain(regressor)
		
	elif op == "cntfac":
		""" intervened value and average """
		prFile = args.mlconf
		cfdatafp = args.cffile
		if  args.cfindex == "none":
			#non intervened
			cfindxs = list()
			cfvals = list()
		else:
			#intervened
			cfindxs = strToIntArray(args.cfindex)
			cfvals = strToFloatArray(args.cfval)
		regressor = FeedForwardNetwork(prFile)
		regressor.buildModel()
		cntfactual(regressor, cfdatafp, cfindxs, cfvals)

	else:
		exitWithMsg("invalid command")
	
		
		
