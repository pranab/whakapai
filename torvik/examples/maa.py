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
import argparse
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *
from torvik.vae import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--cfpath', type=str, default = "none", help = "config file path")
	args = parser.parse_args()
	
	if args.op == "train":
		model = VarAutoEncoder(args.cfpath)
		model.buildModel()
		VarAutoEncoder.trainModel(model)
		
	elif args.op == "pred":
		model = VarAutoEncoder(args.cfpath)
		model.buildModel()
		VarAutoEncoder.predModel(model, True)
