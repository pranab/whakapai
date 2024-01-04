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
		defValues["common.network.type"] = (False, "missing network type")
		defValues["common.config.params.direct"] = (None, "missing config parameter liist")
		defValues["common.config.params.processed"] = (None, None)
		defValues["train.num.layers"] = ([2,4], None)
		defValues["train.num.units"] = (None, "missing range of number of units")
		defValues["train.activation"] = ("relu", None)
		defValues["train.batch.normalize"] = (["true", "false"], None)
		defValues["train.dropout.prob"] = ([-0.1, 0.5], None)
		defValues["train.out.num.units"] = (None, "missing number of output units")
		defValues["train.out.activation"] = (None, "missing output activation")
		defValues["train.batch.size"] = ([16, 128], None)
		defValues["train.opt.learning.rate"] = ([.0001, .005], None)
	
		self.config = Configuration(configFile, defValues)
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]

	def showStudyResults(self, study):
		"""
		shows study results
		
		Parameters
			study : study object
		"""
		trial = study.best_trial
		if self.verbose:
			print("Number of finished trials: ", len(study.trials))
			print("Best trial:")
			print("Value: ", trial.value)
			print("Params: ")
			for key, value in trial.params.items():
				print("  {}: {}".format(key, value))
			
		res = dict()
		res["number of finished trials"] = len(study.trials)
		re["best trial value"] = trial.value
		re["best trial params"] = trial.params
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
			outNunits = tConfig.getIntConfig("train.out.num.units")[0]
			outAct = tConfig.getStringConfig("train.out.activation")[0]
			batchNormOptions = tConfig.getSrtringListConfig("train.batch.notm")[0]
		
			#same choise for all layers
			numLayers = trial.suggest_int("numLayers", nlayers[0], nlayers[1]) if len(nlayers) > 1 else nlayers[0]
			batchNorm = trial.suggest_categorical("batchNorm", batchNormOptions) if len(batchNormOptions) > 1 else batchNormOptions[0]
	
			layerConfig = ""
			maxUnits = nunits[1]
			sep = ":"
			for i in range(nlayers):
				if i < nlayers - 1:
					nunit = trial.suggest_int("numUnits_l{}".format(i), nunits[0], maxUnits) if len(nunits) > 1 else nunits[0]
					dropOut = trial.suggest_float("dropOut_l{}".format(i), dropOutRange[0], dropOutRange[1]) if len(dropOutRange) > 1 else dropOutRange[0]
					act = trial.suggest_categorical("dropOut_l{}".format(i), acts[0], acts[1]) if len(acts) > 1 else acts[0]
					lconfig = [str(nunit), act, batchNorm, "true", "{:.3f}".format(dropOut)]
					lconfig = sep.join(lconfig) + ","
					layerConfig = layerConfig + lconfig
					maxUnits = nunit
				else:
					lconfig = [str(outNunits), outAct, "false", "false", "{:.3f}".format(-0.1)]
					lconfig = sep.join(lconfig)
					layerConfig = layerConfig + lconfig
			nnModel.setConfigParam("train.layer.data", layerConfig)
		else:
			exitWithMsg("invalid network type")
		
		#direct parameters
		dparams =  tConfig.getStringListConfig("common.config.params.direct")[0]
		for dparam in dparams:
			items = dparam.split(":")
			pname = items[0]
			ptype = items[1]
				
			if ptype == "string":
				pvalues = tConfig.getStringListConfig(pname)[0]
				if len(pvalues) > 1:	
					selpval = trial.suggest_categorical(pname, pvalues)
					nnModel.setConfigParam(pname, selpval)
			elif ptype == "int":
				pvalues = tConfig.getIntListConfig(pname)[0]
				if len(pvalues) > 1:	
					selpval = trial.suggest_int(pname, pvalues)
					nnModel.setConfigParam(pname, str(selpval))
			elif ptype == "float":
				pvalues = tConfig.getFloatListConfig(pname)[0]
				if len(pvalues) > 1:	
					spvalue = tConfig.getStringListConfig(pname)[0][0]
					fs = spvalue.split(".")
					prec = len(fs[1])
					selpval = trial.suggest_float(pname, pvalues)
					nnModel.setConfigParam(pname, formatFloat(prec, selpval))

		#train model
		nnModel.buildModel()
		score = nnModel.fit()
		cost = 1.0 / score if tConfig.getBooleanConfig("common.inv.score")[0] else score
		return cost

	@staticmethod
	def tune(modelConfigFile, tunerConfigFile, numTrial):
		"""
		entry point to tune model

		Parameters
			trial : trial object
		"""
		tuner = NeuralNetworkTuner(tunerConfigFile)
		study = optuna.create_study()
		study.optimize(lambda trial: NeuralNetworkTuner.objective(trial, modelConfigFile, tuner), n_trials=numTrial)
		return tuner.showStudyResults(study)

