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

# Package imports
import os
import sys
import torch
from torch.utils.data import DataLoader
import random
import jprops
from random import randint
import optuna
from matumizi.util import *
from matumizi.mlutil import *
from torvik.tnn import *

"""
Neural network hyper paramter tuning with optuna. Supports different kinds of NN starting with Feed Forward
Network
"""
class NeuralNetworkTuner(object):
	def __init__(self, configFile):
		"""
		initializer
		
		Parameters
			configFile : config file path
		"""
		defValues = dict()
		defValues["common.verbose"] = (False, None)
		defValues["common.inv.score"] = (False, None)
		defValues["common.network.type"] = (None, "missing network type")
		defValues["common.config.params.direct"] = (None, "missing config parameter liist")
		defValues["common.config.params.processed"] = (None, None)
		defValues["common.config.params.control"] = (None, None)
		defValues["train.num.layers"] = ([2,4], None)
		defValues["train.num.units"] = (None, "missing range of number of units")
		defValues["train.activation"] = (["relu"], None)
		defValues["train.batch.normalize"] = (["true", "false"], None)
		defValues["train.dropout.prob"] = ([0.1, 0.6], None)
		defValues["train.out.num.units"] = (None, "missing number of output units")
		defValues["train.out.activation"] = (None, None)
		defValues["train.batch.size"] = ([20,100], None)
		defValues["train.opt.learning.rate"] = ([0.0001,0.01], None)
		defValues["train.lossFn"] = (None, None) 
		defValues["train.optimizer"] = (None, None) 
		defValues["control.common.verbose"] = (None, None) 
		defValues["control.train.track.error"] = (None, None) 
	
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]

	def showStudyResults(self, study):
		"""
		shows study results
		
		Parameters
			study : study object
		"""
		trial = study.best_trial
		ntrial = len(study.trials)
		if self.verbose:
			print("Number of finished trials: ", ntrial)
			print("Best trial:")
			print("Value: ", trial.value)
			print("Params: ")
			for key, value in trial.params.items():
				print("  {}: {}".format(key, value))
			
		res = dict()
		res["number of finished trials"] = ntrial
		res["best trial value"] = trial.value
		res["best trial params"] = trial.params
		return res
	
	
	@staticmethod
	def objective(trial, modelConfigFile, tuner):
		"""
		model training and error as cost

		Parameters
			trial : trial object
			modelConfigFile : model config file
			tuner : tuner object
		"""
		if tuner.verbose:
			print("next trial")
		tConfig = tuner.config

		#NN model	
		nntype = tConfig.getStringConfig("common.network.type")[0]
		nnModel = None
		
		#parameters with special processing
		if nntype == "ffn":
			nnModel = FeedForwardNetwork(modelConfigFile)
	
			#tuning parameters
			nlayers = tConfig.getIntListConfig("train.num.layers")[0]
			nunits = tConfig.getIntListConfig("train.num.units")[0]
			acts = tConfig.getStringListConfig("train.activation")[0]
			dropOutRange = tConfig.getFloatListConfig("train.dropout.prob")[0]
			outUnits = tConfig.getIntConfig("train.out.num.units")[0]
			outActs = tConfig.getStringListConfig("train.out.activation")[0]
			batchNormOptions = tConfig.getStringListConfig("train.batch.normalize")[0]
		
			#same choise for all layers
			numLayers = trial.suggest_int("numLayers", nlayers[0], nlayers[1]) if len(nlayers) > 1 else nlayers[0]
			batchNorm = trial.suggest_categorical("batchNorm", batchNormOptions) if len(batchNormOptions) > 1 else batchNormOptions[0]
	
			layerConfig = ""
			maxUnits = nunits[1]
			sep = ":"
			
			#hidden layers
			for i in range(numLayers-1):
				#layer data each with layer having smaller number of units than the layer before
				nunits[0] = nunits[0] if nunits[0] >= outUnits else outUnits
				nunit = trial.suggest_int("numUnits_l{}".format(i), nunits[0], maxUnits) if len(nunits) > 1 else nunits[0]
				dropOut = trial.suggest_float("dropOut_l{}".format(i), dropOutRange[0], dropOutRange[1]) if len(dropOutRange) > 1 else dropOutRange[0]
				act = trial.suggest_categorical("act", acts) if len(acts) > 1 else acts[0]
				lconfig = [str(nunit), act, batchNorm, "true", "{:.3f}".format(dropOut)]
				lconfig = sep.join(lconfig) + ","
				layerConfig = layerConfig + lconfig
				maxUnits = nunit
				
			#output layer
			if outActs is not None:
				outAct = trial.suggest_categorical("outAct", outActs) if len(outActs) > 1 else outActs[0]
			else:
				outact = "none"
				
			lconfig = [str(outUnits), outAct, "false", "false", "{:.3f}".format(-0.1)]
			lconfig = sep.join(lconfig)
			layerConfig = layerConfig + lconfig
			
			nnModel.setConfigParam("train.layer.data", layerConfig)
			if tuner.verbose:
				print("train.layer.data\t" + layerConfig)
		else:
			exitWithMsg("invalid network type")
		
		#direct parameters which get mapped one to one
		dparams =  tConfig.getStringListConfig("common.config.params.direct")[0]
		for dparam in dparams:
			items = dparam.split(":")
			pname = items[0]
			ptype = items[1]
				
			if ptype == "string" or ptype == "boolean":
				pvalues = tConfig.getStringListConfig(pname)[0]
				if pvalues is not None and len(pvalues) > 1:	
					selpval = trial.suggest_categorical(pname, pvalues)
					nnModel.setConfigParam(pname, selpval)
					if tuner.verbose:
						print(pname + "\t" + selpval)

			elif ptype == "int":
				pvalues = tConfig.getIntListConfig(pname)[0]
				if  pvalues is not None and len(pvalues) > 1:	
					selpval = trial.suggest_int(pname, pvalues[0], pvalues[1])
					nnModel.setConfigParam(pname, str(selpval))
					if tuner.verbose:
						print(pname + "\t" + str(selpval))

			elif ptype == "float":
				pvalues = tConfig.getFloatListConfig(pname)[0]
				if  pvalues is not None and len(pvalues) > 1:	
					spvalue = tConfig.getStringListConfig(pname)[0][0]
					fs = spvalue.split(".")
					prec = len(fs[1])
					selpval = trial.suggest_float(pname, pvalues[0], pvalues[1])
					selpvalStr = formatFloat(prec, selpval)
					nnModel.setConfigParam(pname, selpvalStr)
					if tuner.verbose:
						print(pname + "\t" + selpvalStr)
			
			
		#control parameters always boolean and not part of optimization e.g controlling output
		dparams =  tConfig.getStringListConfig("common.config.params.control")[0]
		if dparams is not None:
			for dparam in dparams:
				#remove prefix component which is control
				elems = dparam.split(".")
				pname = ".".join(elems[1:])
			
				pvalue = tConfig.getBooleanConfig(dparam)[0]
				if pvalue is not None:	
					spvalue = "True" if pvalue else "False"
					nnModel.setConfigParam(pname, spvalue)
					if tuner.verbose:
						print("control parameter " + pname + "\t" + spvalue)
					

		#train model
		nnModel.buildModel()
		score = nnModel.fit()
		cost = 1.0 / score if tConfig.getBooleanConfig("common.inv.score")[0] else score
		if tuner.verbose:
			print("cost\t{:.3f}".format(cost))
		return cost

	@staticmethod
	def tune(modelConfigFile, tunerConfigFile, numTrial):
		"""
		entry point to tune model

		Parameters
			modelConfigFile : NN model config file
			tunerConfigFile : tuner config file
			numTrial : num of trials
		"""
		tuner = NeuralNetworkTuner(tunerConfigFile)
		study = optuna.create_study()
		study.optimize(lambda trial: NeuralNetworkTuner.objective(trial, modelConfigFile, tuner), n_trials=numTrial)
		return tuner.showStudyResults(study)

