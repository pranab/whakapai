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
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import jprops
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from .opti import *
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *

class SimulatedAnnealingOptimizer(BaseOptimizer):
	"""
	optimize with simulated annealing
	"""
	def __init__(self, configFile, domain):
		"""
		intialize

		Parameters
			configFile : configuration file
			domain : application domain object
		"""
		defValues = {}
		defValues["opti.initial.temo"] = (10.0, None)
		defValues["opti.temp.update.interval"] = (5, None)
		defValues["opti.cooling.rate"] = (0.9, None)
		defValues["opti.cooling.rate.geometric"] = (False, None)
		
		super(SimulatedAnnealingOptimizer, self).__init__(configFile, defValues, domain)

	def run(self):
		"""
		run optimizer
		"""
		self.logger.info("*****  starting SimulatedAnnealingOptimizer  *****")
		self.curSoln = self.createCandidate()
		curCost = self.curSoln.cost
		cloneCand = Candidate()
		cloneCand.clone(self.curSoln)
		self.bestSoln = cloneCand
		bestCost = self.bestSoln.cost

		initialTemp = temp = self.config.getFloatConfig("opti.initial.temo")[0]
		tempUpdInterval = self.config.getIntConfig("opti.temp.update.interval")[0]
		coolingRate = self.config.getFloatConfig("opti.cooling.rate")[0]
		geometricCooling = self.config.getBooleanConfig("opti.cooling.rate.geometric")[0]

		#iterate
		for i in range(self.numIter):
			self.logger.info("iteration " + str(i))
			mutStat, mutatedCand = self.mutateAndValidate(self.curSoln, self.mutateMaxTry, True)
			nextCost = mutatedCand.cost
			if nextCost < curCost:
				#next cost better
				self.logger.debug("got lower cost soln")
				self.curSoln = mutatedCand
				curCost = self.curSoln.cost

				if mutatedCand.cost < bestCost:
					self.logger.info("best soln set")
					cloneCand = Candidate()
					cloneCand.clone(mutatedCand)
					self.bestSoln = cloneCand
					bestCost = self.bestSoln.cost

			else:
				#next cost worse
				self.logger.debug("got higher cost soln")
				t = temp if temp != 0 else .001
				e = math.exp((curCost - nextCost) / t)
				self.logger.debug("expo {:.6f}  temp {:.6f}".format(e, temp))
				if e > random.random():
					self.logger.debug("choosing higher cost soln")
					self.curSoln = mutatedCand
					curCost = self.curSoln.cost

			if i % tempUpdInterval == 0:
				self.logger.info("updating temp")
				if geometricCooling:
					temp *= coolingRate
				else:
					temp = (initialTemp - i * coolingRate)


class BayesianOptimizer(BaseOptimizer):
	"""
	optimize with bayesian optimizer. Finds max, For min cost function should return cost witj sigh inverted
	"""
	def __init__(self, configFile, domain):
		"""
		intialize

		Parameters
			configFile : configuration file
			domain : application domain object
		"""
		defValues = {}
		defValues["opti.initial.model.training.size"] = (1000, None)
		defValues["opti.acquisition.samp.size"] = (100, None)
		defValues["opti.prob.acquisition.strategy"] = ("pi", None)
		defValues["opti.acquisition.ucb.mult"] = (2.0, None)
		self.sample = None
		super(BayesianOptimizer, self).__init__(configFile, defValues, domain)
		self.model = GaussianProcessRegressor()

	def run(self):
		"""
		run optimizer
		"""
		assert Candidare.fixedSz, "BayesianOptimizer works only for fixed size solution"

		for sampler in self.compDataDistr:
			assert sampler.isNumeric(), "BayesianOptimizer works only for numerical data"

		#inir=tial population and moel fit
		trSize = self.config.getIntConfig("opti.initial.model.training.size")[0]
		features, targets = self.createSamples(trSize)
		self.model.fit(features, targets)

		#iterate
		acqSampSize = self.config.getIntConfig("opti.acquisition.samp.size")[0]
		prAcqStrategy = self.config.getIntConfig("opti.prob.acquisition.strategy")[0]
		acqUcbMult = self.config.getFloatConfig("opti.acquisition.ucb.mult")[0]
		for i in range(self.numIter):
			ofeature, otarget = self.optAcquire(features, targets, acqSampSize, prAcqStrategy, acqUcbMult)
			features = np.vstack((features, [ofeature]))
			targets = np.vstack((targets, [otarget]))
			self.model.fit(features, targets)

		ix = np.argmax(targets)
		self.bestSoln = features[ix]
		self.sample = (features, targets)

	def optAcquire(self, features, targets, acqSampSize, prAcqStrategy, acqUcbMult):
		"""
		run optimizer

		Parameters
			features : feature array
			targets : target rray
			acqSampSize : new sample acusition size
			prAcqStrategy : sample acusition strategy
			acqUcbMult : multiplier for upper confidence bound
		"""
		mu = self.model.predict(features)
		best = max(mu)

		sfeatures, stargets = self.createSamples(acqSampSize)
		smu, sstd = self.model.predict(sfeatures, return_std=True)
		if prAcqStrategy == "pi":
			#probability of improvement
			imp = best - smu
			z = imp / (sstd + 1E-9)
			scores = norm.cdf(z)
		elif prAcqStrategy == "ei":
			#expected improvement
			imp = best - smu
			z = imp / (sstd + 1E-9)
			scores = imp * norm.cdf(z) + sstd * norm.pdf(z)
		elif prAcqStrategy == "ucb":
			#upper confidence bound
			scores = smu + acqUcbMult * sstd
		else:
			raise ValueError("invalid acquisition strategy for next best candidate")
		ix = np.argmax(scores)
		sfeature = sfeatures[ix]
		starget = stargets[ix]

		return (sfeature, starget)

	def createSamples(self, size):
		"""
		sample features and targets

		Parameters
			size : no of samples
		"""
		features = list()
		targets = list()
		for i in range(size):
			cand = self.createCandidate()
			features.append(cand.getSolnAsFloat())
			targets.append(cand.cost)
		features = np.asarray(features)
		targets = np.asarray(targets)
		return (features, targets)







