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
		defValues["ac.graph.data"] = (None, None)
		defValues["ac.graph.base.node"] = (None, None)
		defValues["ac.ant.pool.size"] = (10, None)
		defValues["ac.num.iter"] = (10, None)
		defValues["ac.pheromone.exp"] = (1.0, None)
		defValues["ac.heuristic.exp"] = (1.0, None)
		defValues["ac.pheromone.evaporation"] = (0.3, None)
		defValues["ac.pheromone.update.strategy"] = ("as", None)
		defValues["ac.exploration.probab"] = (None, None)
		self.config = Configuration(configFile, defValues)
		
		self.verbose = self.config.getBooleanConfig("common.verbose")[0]
		self.bestSoln = None
		
		self.hexp = self.config.getFloatConfig("ac.heuristic.exp")[0]
		self.pexp = self.config.getFloatConfig("ac.pheromone.exp")[0]
		graph = self.config.getStringListConfig("ac.graph.data")[0]
		self.edges = dict()
		self.nodes = None
		self.__processGraph(graph)
		
		self.antPoolSize = self.config.getIntConfig("ac.ant.pool.size")[0]
		self.base = self.config.getStringConfig("ac.graph.base.node")[0]
		self.ants = [[self.base]] * self.antPoolSize
		self.costs = [0] * self.antPoolSize
		self.explProbab = self.config.getFloatConfig("ac.exploration.probab")[0]
		if self.explProbab is not None:
			self.explProbab = int(100 * self.explProbab)
		
	def run(self):
		"""
		run optimizer
		"""
		niter = self.config.getIntConfig("ac.num.iter")[0]
		
		#iterations
		for _ in range(niter):
			#all nodes
			for j in range(len(self.nodes)):
				#all ants
				for a in range(self.antPoolSize):
					visited = self.ants[a]
					nextVisits = self.__nextToVisitNodes(visited)
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
						pr = (edge[0] ** self.hexp) * (edge[1] ** self.pexp)
						sumPr += pr
						trPr[nv] = pr
				
						#normalize proability
						for k in trPr.keys():
							trPr[k] /= sumPr
				
						#select next node
						snode = self.__selectNextNode(trPr)	
						visited.append(snode)
				
					#updatre path cost
					sedge = self.__getEdge(self, lnode, snode)
					self.costs[a] += sedge[0]
				
			#add base node
			for a in range(self.antPoolSize):
				assertEqual(len(self.ants[a]), len(self.nodes), "not all nodes visited")
				self.ants[a].append(self.base)
				sedge = self.__getEdge(self, self.ants[a][-2], self.ants[a][-1])
				self.costs[a] += sedge[0]
			
			#current iteration best soln
				
			#global best soln
				
			#update pheronope weights
			
			#reset
			self.ants = [[self.base]] * self.antPoolSize
			self.costs = [0] * self.antPoolSize
			
				
			
									
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
			edge = (e[0], e[1])
			wt = float(e[2])
			self.edges[edge] = [wt]
			if wt > maxWt:
				maxWt = wt
			nset.add(e[0])
			nset.add(e[1])
			
		self.nodes = list(nset)
		
		#scale
		sumw = 0
		for e in self.edges.keys():
			wt = maxWt / self.edges[e][0]
			self.edges[e][0] = wt
			sumw += wt
		
		# preonome weight is average heuristic weight	
		avWt = sumw / len(self.edges)
		for e in self.edges.keys():
			self.edges[e].append(avWt)
	
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
			edge = self.edges[(n1,n2)]
		elif (n2,n1) in self.edges:
			edge = self.edges[(n2,n1)]
		else:
			exitWithMsg("edge does not exist with nodes " + n1  + " " + n2)
		return edge
		
	def __selectNextNode(self, trPr):
		"""
		select next node to  visit

		Parameters
			trPr : transition probability of next nodes
		"""
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
		else:
			#sample node
			distr = list(zip(trPr.keys(), trPr.values()))
			sampler = CategoricalRejectSampler(distr)
			snode = sampler.sample()
			
		return snode
		
	def __updatePheronome(self):
		"""
		update all pheronome weights

		"""
		pass
		
		
	
		