#!/usr/local/bin/python3

# qinisa : Machine Learning
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
import random 
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *

"""
demo for model collapse by repeated sampling and building distribution from samples
"""
def barPlot(px, py, xlab, ylab):
	"""
	sets reward
		
	Parameters
		px : x values
		py : yvalues
		xlab : x label
		ylab : y label
	"""
	fig = plt.figure(figsize = (8, 4))
	plt.bar(px, py, color ='maroon', width = 0.4)
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.show()
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--niter', type=int, default = 100, help = "num of of iteration")
	parser.add_argument('--nsamp', type=int, default = 100, help = "num of of samples")
	args = parser.parse_args()

	niter = args.niter
	xmin = 10
	nbins = 18
	binWidth = 5.0
	entropy = list()
	px = list(range(15,101,5))
	
	sampler = NonParamRejectSampler(xmin, binWidth, 0.08, 0.12, 0.14, 0.18, 0.25, 0.34, 0.48, 0.33,
	0.23, 0.16, 0.19, 0.23, 0.21, 0.18, 0.15, 0.13, 0.12, 0.11)
	sampler.sampleAsFloat()
	samples = list(map(lambda i: sampler.sample(), range(args.nsamp)))
	hgram = Histogram.createUninitializedWithNumBins(xmin, binWidth, nbins)
	for v in samples:
		hgram.add(v)
	bvalues = hgram.distr()
	entropy.append(hgram.entropy())
	
	barPlot(px, bvalues, "values", "frequecy")
	
	
	for i in range(niter):
		sampler = NonParamRejectSampler(xmin, binWidth, bvalues)
		sampler.sampleAsFloat()
		samples = list(map(lambda i: sampler.sample(), range(args.nsamp)))
		hgram.initialize()
		for v in samples:
			hgram.add(v)
		bvalues = hgram.distr()
		entropy.append(hgram.entropy())
		if i > 0 and (i % 50 == 0  or i == niter - 1):
			barPlot(px, bvalues, "values", "frequecy")
	
		
	drawPlot(None, entropy, "iteration", "entropy")
		