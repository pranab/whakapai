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
import jprops
import numpy as np
import statistics 
import argparse
from matumizi.util import *
from matumizi.sampler import *
from arotau.opti import *
from arotau.optsolo import *
from arotau.swarmopt import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "", help = "config file path")
	args = parser.parse_args()
	op = args.op
	
	if op == "opt":
		optimizer = AntColonyOptimizer(args.cfpath)
		optimizer.run()
