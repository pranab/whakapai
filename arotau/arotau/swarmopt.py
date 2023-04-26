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
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
import jprops
from arotau.opti import *
from matumizi.util import *
from matumizi.mlutil import *
from matumizi.sampler import *


class AntColonyOptimizer(object):
	"""
	optimize with ant colony 
	"""
	def __init__(self, configFile):
		"""
		intialize

		Parameters
			configFile : configuration file
		"""
		defValues = {}
		defValues["common.verbose"] = (False, None)
		defValues["common.logging.file"] = (None, "missing log file path")
		defValues["common.logging.level"] = ("info", None)
		defValues["ac.graph.data"] = (None, None)
		defValues["ac.graph.base.node"] = (None, None)
		defValues["ac.ant.pool.size"] = (10, None)
		defValues["ac.num.iter"] = (10, None)
		defValues["ac.pheromone.exp"] = (1.0, None)
		defValues["ac.heuristic.exp"] = (1.0, None)
		defValues["ac.pheromone.evaporation.param"] = (0.5, None)
		defValues["ac.pheromone.update.policy"] = ("as", None)
		defValues["ac.pheromone.add.param"] = (None, None)
		defValues["ac.exploration.probab"] = (0.2, None)
		self.config = Configuration(configFile, defValues)
		
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		logFilePath = self.config.getStringConfig("common.logging.file")[0]
		logLevName = self.config.getStringConfig("common.logging.level")[0]
		self.logger = createLogger(__name__, logFilePath, logLevName)
		
		self.bestSoln = None
		
		self.hexp = self.config.getFloatConfig("ac.heuristic.exp")[0]
		self.pexp = self.config.getFloatConfig("ac.pheromone.exp")[0]
		graph = self.config.getStringListConfig("ac.graph.data")[0]
		
		self.antPoolSize = self.config.getIntConfig("ac.ant.pool.size")[0]
		self.base = self.config.getStringConfig("ac.graph.base.node")[0]
		self.ants = None
		self.weights = None
		self.plens = None
		self.__initAntPool()

		self.explProbab = self.config.getFloatConfig("ac.exploration.probab")[0]
		if self.explProbab is not None:
			self.explProbab = int(100 * self.explProbab)
		
		# if pheromone add param not set then set to num of nodes assuming max wt of edge is 1
		self.pheromAddParam = self.config.getFloatConfig("ac.pheromone.add.param")[0]
		self.pheromEvaporParam = self.config.getFloatConfig("ac.pheromone.evaporation.param")[0]
		self.pheromUpdPolicy = self.config.getStringConfig("ac.pheromone.update.policy")[0]
		
		self.iterBestSoln = None
		self.bestSoln = None
		
		self.edges = dict()
		self.nodes = None
		self.numNodes = None
		self.__processGraph(graph)
		
	def run(self):
		"""
		run optimizer
		"""
		self.logger.info("**** Starting AntColonyOptimizer ****")
		niter = self.config.getIntConfig("ac.num.iter")[0]
		
		#iterations
		for it in range(niter):
			self.logger.debug("iteration " +  str(it))
			#all nodes
			for j in range(self.numNodes - 1):
				#all ants
				for a in range(self.antPoolSize):
					visited = self.ants[a]
					self.logger.debug("ant {}  visited so far {}".format(a, str(visited)))
					nextVisits = self.__nextToVisitNodes(visited)
					assertGreater(len(nextVisits), 0, "failed to find next node to visit, visited so far  " + str(visited))
					trPr = dict()
					sumPr = 0
					cnode = visited[-1]
				
					# transition probabailities for the next nodes and select next node
					for nv in nextVisits:
						k = (cnode, nv)
						if k in self.edges:
							edge = self.edges[k]
						else:
							k = (nv, cnode)
							edge = self.edges[k]	
						pr = (edge[1] ** self.hexp) * (edge[2] ** self.pexp)
						sumPr += pr
						trPr[nv] = pr
				
					#normalize proability
					for k in trPr.keys():
						trPr[k] /= sumPr
				
					#select next node
					snode = self.__selectNextNode(trPr)	
					self.ants[a].append(snode)
					self.logger.debug("ant {}  new visit {}".format(a, snode))
						
					#updatre path cost and length
					sedge = self.__getEdge(cnode, snode)[1]
					self.plens[a] += sedge[0]
					self.weights[a] += sedge[1]
				
					#print(self.ants)
					
			#add base node
			for a in range(self.antPoolSize):
				assertEqual(len(self.ants[a]), len(self.nodes), "not all nodes visited")
				self.ants[a].append(self.base)
				sedge = self.__getEdge(self.ants[a][-2], self.ants[a][-1])[1]
				self.plens[a] += sedge[0]
				self.weights[a] += sedge[1]
			
			#current iteration best soln
			minCost = 10 * self.numNodes
			bi = None
			for i in range(len(self.plens)):
				if self.plens[i] < minCost:
					bi = i
					minCost = self.plens[i]
			self.iterBestSoln = (self.ants[bi], self.weights[bi], self.plens[bi])
			mformat = "current iteration best soln {} weight {:.6f}  cost {:.6f}"
			self.logger.debug(mformat.format(str(self.iterBestSoln[0]), self.iterBestSoln[1], self.iterBestSoln[2]))
				
			#global best soln
			if self.bestSoln is None or self.iterBestSoln[2] < self.bestSoln[2]:
				self.bestSoln = self.iterBestSoln
				mformat = "global best soln at iteration {} path {} weight {:.6f}  cost {:.6f}"
				self.logger.info(mformat.format(it, str(self.bestSoln[0]), self.bestSoln[1], self.bestSoln[2]))
				print(mformat.format(it, str(self.bestSoln[0]), self.bestSoln[1], self.bestSoln[2]))
				
			#update pheronope weights
			self.__updatePheronome()
			
			#reset
			self.__initAntPool()
				
		self.logger.info("final global best soln  path {} weight {:.6f}  cost {:.6f}".format(str(self.bestSoln[0]), self.bestSoln[1], self.bestSoln[2]))
		print("final global best soln  path {} weight {:.6f}  cost {:.6f}".format(str(self.bestSoln[0]), self.bestSoln[1], self.bestSoln[2]))
		
	def __initAntPool(self):
		"""
		initalize ant pool
		"""
		self.ants = list()
		for _ in range(self.antPoolSize):
			self.ants.append([self.base])
		self.logger.debug("ants " + str(self.ants))
		self.weights = [0] * self.antPoolSize
		self.plens = [0] * self.antPoolSize
	
										
	def __processGraph(self, edges):
		"""
		process graph data

		Parameters
			edges : graph edge data
		"""
		maxWt = 0
		nset = set()
		for e in edges:
			items = e.split(":")
			assertEqual(len(items), 3, "incorrect edge data num items " + str(len(items)))
			edge = (items[0], items[1])
			wt = float(items[2])
			self.edges[edge] = [wt]
			if wt > maxWt:
				maxWt = wt
			nset.add(items[0])
			nset.add(items[1])
			
		self.nodes = list(nset)
		self.numNodes = len(self.nodes)
		
		#scale by max value
		for e in self.edges.keys():
			wt = self.edges[e][0] / maxWt
			self.edges[e][0] = wt
			self.edges[e].append(1 / wt)

		
		# phereonome weight initilized based on random path weight 
		if self.pheromAddParam is None:
			self.pheromAddParam = self.numNodes
		for e in self.edges.keys():
			plen = randomFloat(0.2 * self.numNodes, 0.8 * self.numNodes)
			self.edges[e].append(self.pheromAddParam / plen)
			assertEqual(len(self.edges[e]), 3, "incorrect edge data num items " + str(len(self.edges[e])))
	
		self.logger.debug("graph data")
		for e in self.edges.keys():
			mformat = "edge  {} length {}  cost {:.3f} pheromone weight {:.3f}"
			self.logger.debug(mformat.format(str(e), self.edges[e][0], self.edges[e][1], self.edges[e][2]))

	def __nextToVisitNodes(self, visited):
		"""
		get nodes not visited that are immediate neighbors of the current node

		Parameters
			visited : visited nodes
		"""
		cnode = visited[-1]
		nextVisits = list()
		for e in self.edges.keys():
			if e[0] == cnode:
				if e[1] not in visited[:-1]:
					nextVisits.append(e[1])
			elif e[1] == cnode:	
				if e[0] not in visited[:-1]:
					nextVisits.append(e[0])
			else:
				pass
				
		return nextVisits
	
	def __getEdge(self, n1, n2):
		"""
		get edge properties

		Parameters
			n1 : node 1 of edge
			n2 : node 2 of edge
		"""
		if (n1,n2) in self.edges:
			ekey = (n1,n2)
			edge = self.edges[ekey]
		elif (n2,n1) in self.edges:
			ekey = (n2,n1)
			edge = self.edges[ekey]
		else:
			exitWithMsg("edge does not exist with nodes " + n1  + " " + n2)
		return (ekey, edge)
		
	def __selectNextNode(self, trPr):
		"""
		select next node to  visit

		Parameters
			trPr : transition probability of next nodes
		"""
		self.logger.debug("neighbor weight distribution " + str(trPr))
		greedy = True
		if self.explProbab is not None:
			if isEventSampled(self.explProbab):
				greedy = False
		
		snode = None
		if greedy:
			#select node with max probabaility
			maxp = 0
			for n in trPr.keys():
				if trPr[n] > maxp:
					snode = n
					maxp = trPr[n]
			self.logger.debug("selected {} greedily".format(snode))
		else:
			#sample node
			distr = list(zip(trPr.keys(), trPr.values()))
			sampler = CategoricalRejectSampler(distr)
			snode = sampler.sample()
			self.logger.debug("selected {} by sampling".format(snode))
			
		return snode
		
	def __updatePheronome(self):
		"""
		update all pheronome weights

		"""
		#evaporate
		for e in self.edges.keys():
			self.edges[e][2] *= (1 - self.pheromEvaporParam)
		
		#add weight	
		if self.pheromUpdPolicy == "as":
			#all ants in current iteration
			for a in range(len(self.ants)):
				ant = self.ants[a]
				plen = self.plens[a]
				self.__addPheromone(ant, plen)
		
		elif self.pheromUpdPolicy == "ib":
			#current iteration best ant
			ant = self.iterBestSoln[0]
			plen = self.iterBestSoln[2]
			self.__addPheromone(ant, plen)
	
		elif self.pheromUpdPolicy == "bs":
			#best solution so far
			ant = self.bestSoln[0]
			plen = self.bestSoln[2]
			self.__addPheromone(ant, plen)

		self.logger.debug("edge pheromone")
		for e in self.edges.keys():
			self.logger.debug("edge {} pheromone {:.3f}".format(str(e), self.edges[e][2]))
	
	def __addPheromone(self, ant, plen):
		"""
		add pheromon to all edges in the ant solution path

		Parameters
			ant : ant path
			plen : ant path length
		"""
		for i in range(len(ant) - 1):
			ekey =  self.__getEdge(ant[i], ant[i+1])[0]	
			self.edges[ekey][2] += self.pheromEvaporParam * self.pheromAddParam / plen
	