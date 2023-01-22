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
import matplotlib.pyplot as plt 
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.daexp import *
from matumizi.sampler import *

"""
AB test simulation with counterfactuals
"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--genconf', type=str, default = "", help = "data gennerator config file")
	parser.add_argument('--nsamp', type=int, default = 1000, help = "no of samples to generate")
	args = parser.parse_args()
	op = args.op
	
	if op == "gen":
		"""  generate data """
		dgen = RegressionDataGenerator(args.genconf)
		for _ in range(args.nsamp):
			s = dgen.sample()
			pv = toStrFromList(s[0], 2)
			pv = pv + "," + toStr(s[1], 2)
			print(pv)
		
		
