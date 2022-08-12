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
from matumizi.sampler import *
from matumizi.mcsim import *

"""
Balances portfolio with Monte Carlo simulation and Sharpe ratio
"""

class PortFolio():
	"""
	portfolio
	"""
	def __init__(self):
		"""
		
		"""
		self.stocks = list()
		self.srets = list()
		self.rcovar = None
		self.nstock = None
		self.weights = None
		self.metric = -sys.float_info.max
		self.rfret = None
		self.spred = list()
	
	
	def loadStData(self, sdfPath, exfac):
		"""
		load and process stock data
		"""
		e1 = 1 - exfac
		e2 = e1 * e1
		files = getAllFiles(sdfPath)
		print(files)
		
		returns = list()
		for ss, qn, pp in self.stocks:
			print("next stock ", ss)
			for fp in files:
				fname = os.path.basename(fp)
				stname = fname.split("_")[0]
				#print("stock nane from file name ", stname)
				
				if stname == ss:
					#daily prices
					print("loading ", ss)
					prices = getFileColumnAsString(fp, 1)
					prices = prices [1:]
					prices = list(map(lambda p : float(p[1:]), prices))
			
					#predicted price and retuen
					sppred = exfac * prices[0] + exfac * e1 * prices[1] + exfac * e2 * prices[2]
					self.spred.append(sppred)
					up = pp / qn
					sret = (sppred - up) / up
					r = (ss, sret)
					self.srets.append(r)
			
					#daily returns
					bp = prices[-1]
					sdret = list(map(lambda p : (p - bp) / bp, prices))
					#print("daily return size ",  len(sdret))
					returns.append(sdret)
					break
			
		returns = np.array(returns)
		print("daily returns shape ",returns.shape)
		self.rcovar = np.cov(returns)
		print("covar shape ", self.rcovar.shape)


	def optimize(self):
		"""
		balance i.e make buy, sell recommendations
		
		"""
		tamount = 0
		amounts = list()
		for ss, qn , pp in self.stocks:
			amnt = pp
			amounts.append(amnt)
			tamount += amnt
		
		namounts = list(map(lambda w : w * tamount, self.weights))
		quantities = list()
		for am, nam, ppr in zip(amounts, namounts, self.spred):
			#no of stocks to buy or sell for each
			tamount = nam - am
			qnt = int(tamount / ppr)
			quantities.append(qnt)
		
		trans = list()	
		for s, q in zip(self.stocks, quantities):
			tr = (s[0], q)
			trans.append(tr)
			
		return trans

# portfolio object
pfolio = PortFolio()
		
def balance(args):
	"""
	callback for portfolio weights
	"""
	weights = args[:pfolio.nstock]
	#print("wieights ", weights)
	weights = scaleBySum(weights)
	#print("scaled wieights ", weights)
	
	#weighted return
	wr = 0
	for r, w in zip(pfolio.srets, weights):
		wr += (r[1] - pfolio.rfret) * w
	
	wrcv = 0	
	for i in range(pfolio.nstock):
		for j in range(pfolio.nstock):
			wrcv += pfolio.rcovar[i][j] * weights[i] * weights[j]
	
	metric = wr / wrcv
	print("score {:.3f}".format(metric))
	if metric > pfolio.metric:
		pfolio.metric = metric
		pfolio.weights = weights		
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--op', type=str, default = "none", help = "operation")
	parser.add_argument('--niter', type=int, default = "none", help = "num of iterations")
	parser.add_argument('--sdfpath', type=str, default = "none", help = "stock data file directory path")
	parser.add_argument('--spdpath', type=str, default = "none", help = "path of file containing purchase data")
	parser.add_argument('--exfac', type=float, default = 0.9, help = "exponential factor for prediction")
	parser.add_argument('--rfret', type=float, default = 0.2, help = "risk free return")
	args = parser.parse_args()
	op = args.op

	if op == "simu":
		tdata = getFileLines(args.spdpath)
		for rec in tdata:
			#stock symbol, quantity, purchase price
			sname = rec[0]
			quant = int(rec[1])
			pcost = float(rec[2])
			t = (sname, quant, pcost)
			pfolio.stocks.append(t)
		
		#create and run simulator	
		numIter = args.niter
		lfp = "./log/mcsim.log"
		simulator = MonteCarloSimulator(numIter, balance, lfp, "info")
		nstock = len(pfolio.stocks)
		for _ in range(nstock):
			simulator.registerUniformSampler(0.0, 1.0)
		pfolio.nstock = nstock
		pfolio.rfret = args.rfret
		pfolio.loadStData(args.sdfpath, args.exfac)	
		simulator.run()
		
		print("best score {:.3f}".format(pfolio.metric))
		print("weights ", pfolio.weights)
		print("buy and sell recommendations")
		trans = pfolio.optimize()
		for tr in trans:
			print(tr)
		
		
		
