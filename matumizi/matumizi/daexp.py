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
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn import metrics
import random
from math import *
from decimal import Decimal
import pprint
from statsmodels.graphics import tsaplots
from statsmodels.tsa import stattools as stt
from statsmodels.stats import stattools as sstt
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from scipy import stats as sta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hurst
from .util import *
from .mlutil import *
from .sampler import *
from .stats import *

"""
Load  data from a CSV file, data frame, numpy array or list
Each data set (array like) is given a name while loading
Perform various data exploration operation refering to the data sets by name
Save and restore workspace if needed
"""
class DataSetMetaData:
	"""
	data set meta data
	"""
	dtypeNum = 1
	dtypeCat = 2
	dtypeBin = 3
	def __init__(self, dtype):
		self.notes = list()
		self.dtype = dtype

	def addNote(self, note):
		"""
		add note
		"""
		self.notes.append(note)


class DataExplorer:
	"""
	various data exploration functions
	"""
	def __init__(self, verbose=True):
		"""
		initialize

		Parameters
			verbose : True for verbosity
		"""
		self.dataSets = dict()
		self.metaData = dict()
		self.pp = pprint.PrettyPrinter(indent=4)
		self.verbose = verbose

	def setVerbose(self, verbose):
		"""
		sets verbose

		Parameters
			verbose : True for verbosity
		"""
		self.verbose = verbose
		
	def save(self, filePath):
		"""
		save checkpoint
		
		Parameters
			filePath : path of file where saved
		"""
		self.__printBanner("saving workspace")
		ws = dict()
		ws["data"] = self.dataSets
		ws["metaData"] = self.metaData
		saveObject(ws, filePath)
		self.__printDone()

	def restore(self, filePath):
		"""
		restore checkpoint
		
		Parameters
			filePath : path of file from where to store
		"""
		self.__printBanner("restoring workspace")
		ws = restoreObject(filePath)
		self.dataSets = ws["data"]
		self.metaData = ws["metaData"]
		self.__printDone()


	def queryFileData(self, filePath,  *columns):
		"""
		query column data type  from a data file
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("querying column data type from a data frame")
		lcolumns = list(columns)
		noHeader = type(lcolumns[0]) ==  int
		if noHeader:			
			df = pd.read_csv(filePath,  header=None) 
		else:
			df = pd.read_csv(filePath,  header=0) 
		return self.queryDataFrameData(df,  *columns)

	def queryDataFrameData(self, df,  *columns):
		"""
		query column data type  from a data frame
		
		Parameters
			df : data frame with data
			columns : indexes followed by column name or column names
		"""
		self.__printBanner("querying column data type  from a data frame")
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		dtypes = list()
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			cnames = columns[nCols:]
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci]
				dtypes.append(self.getDataType(col))
		else:
			cnames = columns
			for c in columns:
				col = df[c]
				dtypes.append(self.getDataType(col))

		nt = list(zip(cnames, dtypes))
		result = self.__printResult("columns and data types", nt)
		return result

	def getDataType(self, col):
		"""
		get data type 
		
		Parameters
			col : contains data array like
		"""
		if isBinary(col):
			dtype = "binary"
		elif  isInteger(col):
			dtype = "integer"
		elif  isFloat(col):
			dtype = "float"
		elif  isCategorical(col):
			dtype = "categorical"
		else:
			dtype = "mixed"
		return dtype


	def addFileNumericData(self,filePath,  *columns):
		"""
		add numeric columns from a file
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding numeric columns from a file")
		self.addFileData(filePath, True, *columns)
		self.__printDone()


	def addFileBinaryData(self,filePath,  *columns):
		"""
		add binary columns from a file
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding binary columns from a file")
		self.addFileData(filePath, False, *columns)
		self.__printDone()

	def addFileData(self, filePath,  numeric, *columns):
		"""
		add columns from a file
		
		Parameters
			filePath : path of file with data
			numeric : True if numeric False in binary
			columns : indexes followed by column names or column names
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			df = pd.read_csv(filePath,  header=None) 
		else:
			df = pd.read_csv(filePath,  header=0) 
		self.addDataFrameData(df, numeric, *columns)

	def addDataFrameNumericData(self,filePath,  *columns):
		"""
		add numeric columns from a data frame
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding numeric columns from a data frame")
		self.addDataFrameData(filePath, True, *columns)


	def addDataFrameBinaryData(self,filePath,  *columns):
		"""
		add binary columns from a data frame
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding binary columns from a data frame")
		self.addDataFrameData(filePath, False, *columns)


	def addDataFrameData(self, df,  numeric, *columns):
		"""
		add columns from a data frame
		
		Parameters
			df : data frame with data
			numeric : True if numeric False in binary
			columns : indexes followed by column names or column names
		"""
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci]
				if numeric:
					assert isNumeric(col), "data is not numeric"
				else:
					assert isBinary(col), "data is not binary"
				col = col.to_numpy()
				cn = columns[i + nCols]
				dtype = DataSetMetaData.dtypeNum if numeric else DataSetMetaData.dtypeBin
				self.__addDataSet(cn, col, dtype)
		else:
			for c in columns:
				col = df[c]
				if numeric:
					assert isNumeric(col), "data is not numeric"
				else:
					assert isBinary(col), "data is not binary"
				col = col.to_numpy()
				dtype = DataSetMetaData.dtypeNum if numeric else DataSetMetaData.dtypeBin
				self.__addDataSet(c, col, dtype)

	def __addDataSet(self, dsn, data, dtype):
		"""
		add dada set
		
		Parameters
			dsn: data set name
			data : numpy array data 
		"""
		self.dataSets[dsn] = data
		self.metaData[dsn] = DataSetMetaData(dtype)


	def addListNumericData(self, ds,  name):
		"""
		add numeric data from a list
		
		Parameters
			ds : list with data
			name : name of data set
		"""
		self.__printBanner("add numeric data from a list")
		self.addListData(ds, True,  name)
		self.__printDone()


	def addListBinaryData(self, ds, name):
		"""
		add binary data from a list
		
		Parameters
			ds : list with data
			name : name of data set
		"""
		self.__printBanner("adding binary data from a list")
		self.addListData(ds, False,  name)
		self.__printDone()

	def addListData(self, ds, numeric,  name):
		"""
		adds list data
		
		Parameters
			ds : list with data
			numeric : True if numeric False in binary
			name : name of data set
		"""
		assert type(ds) == list, "data not a list"
		if numeric:
			assert isNumeric(ds), "data is not numeric"
		else:
			assert isBinary(ds), "data is not binary"
		dtype = DataSetMetaData.dtypeNum if numeric else DataSetMetaData.dtypeBin
		self.dataSets[name] = np.array(ds)
		self.metaData[name] = DataSetMetaData(dtype)


	def addFileCatData(self, filePath,  *columns):
		"""
		add categorical columns from a file
		
		Parameters
			filePath : path of file with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding categorical columns from a file")
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			df = pd.read_csv(filePath,  header=None) 
		else:
			df = pd.read_csv(filePath,  header=0) 

		self.addDataFrameCatData(df,  *columns)
		self.__printDone()

	def addDataFrameCatData(self, df,  *columns):
		"""
		add categorical columns from a data frame
		
		Parameters
			df : data frame with data
			columns : indexes followed by column names or column names
		"""
		self.__printBanner("adding categorical columns from a data frame")
		columns = list(columns)
		noHeader = type(columns[0]) ==  int
		if noHeader:			
			nCols = int(len(columns) / 2)
			colIndexes = columns[:nCols]
			nColsDf = len(df.columns)
			for i in range(nCols):
				ci = colIndexes[i]
				assert ci < nColsDf, "col index {} outside range".format(ci)
				col = df.loc[ : , ci]
				assert isCategorical(col), "data is not categorical"
				col = col.tolist()
				cn = columns[i + nCols]
				self.__addDataSet(cn, col, DataSetMetaData.dtypeCat)
		else:
			for c in columns:
				col = df[c].tolist()
				self.__addDataSet(c, col, DataSetMetaData.dtypeCat)

	def addListCatData(self, ds, name):
		"""
		add categorical list data
		
		Parameters
			ds : list with data
			name : name of data set
		"""
		self.__printBanner("adding categorical list data")
		assert type(ds) == list, "data not a list"
		assert isCategorical(ds), "data is not categorical"
		self.__addDataSet(name, ds, DataSetMetaData.dtypeCat)
		self.__printDone()

	def remData(self, ds):
		"""
		removes data set
		
		Parameters
			ds : data set name
		"""
		self.__printBanner("removing data set", ds)
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
		self.dataSets.pop(ds)
		self.metaData.pop(ds)
		names = self.showNames()
		self.__printDone()	
		return names
	
	def addNote(self, ds, note):
		"""
		get data
		
		Parameters
			ds : data set name or list or numpy array with data
			note: note text
		"""
		self.__printBanner("adding note")
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
		mdata = self.metaData[ds]
		mdata.addNote(note)
		self.__printDone()

	def getNotes(self, ds):
		"""
		get data
		
		Parameters
			ds : data set name or list or numpy array with data
		"""
		self.__printBanner("getting notes")
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)		
		mdata = self.metaData[ds]
		dnotes = mdata.notes
		if self.verbose:
			for dn in dnotes:
				print(dn)
		return dnotes

	def getNumericData(self, ds):
		"""
		get numeric data
		
		Parameters
			ds : data set name or list or numpy array with data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			assert self.metaData[ds].dtype == DataSetMetaData.dtypeNum, "data set {} is expected to be numerical type for this operation".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			assert isNumeric(ds), "data is not numeric"
			data = np.array(ds)
		elif type(ds) == np.ndarray:
			data = ds
		else:
			raise "invalid type, expecting data set name, list or ndarray"			
		return data


	def getCatData(self, ds):
		"""
		get categorical data
		
		Parameters
			ds : data set name or list  with data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			assert self.metaData[ds].dtype == DataSetMetaData.dtypeCat, "data set {} is expected to be categorical type for this operation".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			assert isCategorical(ds), "data is not categorical"
			data = ds
		else:
			raise "invalid type, expecting data set name or list"
		return data

	def getAnyData(self, ds):
		"""
		get any data
		
		Parameters
			ds : data set name or list  with data
		"""
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			data =   self.dataSets[ds]
		elif type(ds) == list:
			data = ds
		else:
			raise "invalid type, expecting data set name or list"
		return data

	def loadCatFloatDataFrame(self, ds1, ds2):
		"""
		loads float and cat data into data frame
		
		Parameters
			ds1: data set name or list
			ds2: data set name or list or numpy array
		"""
		data1 = self.getCatData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		df1 = pd.DataFrame(data=data1)
		df2 = pd.DataFrame(data=data2)
		df = pd.concat([df1,df2], axis=1)
		df.columns = range(df.shape[1])
		return df

	def showNames(self):
		"""
		lists data set names
		"""
		self.__printBanner("listing data set names")
		names = self.dataSets.keys()
		if self.verbose:
			print("data sets")
			for ds in names:
				print(ds)
		self.__printDone()
		return names

	def plot(self, ds, yscale=None):
		"""
		plots data
		
		Parameters
			ds: data set name or list or numpy array
			yscale: y scale
		"""
		self.__printBanner("plotting data", ds)
		data = self.getNumericData(ds)
		drawLine(data, yscale)

	def plotZoomed(self, ds, beg, end, yscale=None):
		"""
		plots zoomed data
		
		Parameters
			ds: data set name or list or numpy array
			beg: begin offset
			end: end offset
			yscale: y scale
		"""
		self.__printBanner("plotting data", ds)
		data = self.getNumericData(ds)
		drawLine(data[beg:end], yscale)

	def scatterPlot(self, ds1, ds2):
		"""
		scatter plots data
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
		"""
		self.__printBanner("scatter plotting data", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		x = np.arange(1, len(data1)+1, 1)
		plt.scatter(x, data1 ,color="red")
		plt.scatter(x, data2 ,color="blue")
		plt.show()

	def print(self, ds):
		"""
		prunt data
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("printing data", ds)
		assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
		data =   self.dataSets[ds]
		if self.verbore:
			print(formatAny(len(data), "size"))
			print("showing first 50 elements" )
			print(data[:50])

	def plotHist(self, ds, cumulative, density, nbins=20):
		"""
		plots histogram
		
		Parameters
			ds: data set name or list or numpy array
			cumulative : True if cumulative
			density : True to normalize for probability density
			nbins : no of bins
		"""
		self.__printBanner("plotting histogram", ds)
		data = self.getNumericData(ds)
		plt.hist(data, bins=nbins, cumulative=cumulative, density=density)
		plt.show()

	def isMonotonicallyChanging(self, ds):
		"""
		checks if monotonically increasing or decreasing
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("checking  monotonic change", ds)
		data = self.getNumericData(ds)
		monoIncreasing = all(list(map(lambda i : data[i] >= data[i-1], range(1, len(data), 1))))
		monoDecreasing = all(list(map(lambda i : data[i] <= data[i-1], range(1, len(data), 1))))
		result = self.__printResult("monoIncreasing", monoIncreasing, "monoDecreasing", monoDecreasing)
		return result

	def getFreqDistr(self, ds,  nbins=20):
		"""
		get histogram
		
		Parameters
			ds: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting histogram", ds)
		data = self.getNumericData(ds)
		frequency, lowLimit, binsize, extraPoints = sta.relfreq(data, numbins=nbins)
		result = self.__printResult("frequency", frequency, "lowLimit", lowLimit, "binsize", binsize, "extraPoints", extraPoints)
		return result


	def getCumFreqDistr(self, ds,  nbins=20):
		"""
		get cumulative freq distribution
		
		Parameters
			ds: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting cumulative freq distribution", ds)
		data = self.getNumericData(ds)
		cumFrequency, lowLimit, binsize, extraPoints = sta.cumfreq(data, numbins=nbins)
		result = self.__printResult("cumFrequency", cumFrequency, "lowLimit", lowLimit, "binsize", binsize, "extraPoints", extraPoints)
		return result

	def getExtremeValue(self, ds,  ensamp, nsamp, polarity, doPlotDistr, nbins=20):
		"""
		get extreme values
		
		Parameters
			ds: data set name or list or numpy array
			ensamp: num of samples for extreme values
			nsamp: num of samples
			polarity: max or min
			doPlotDistr: plot distr
			nbins: num of bins
		"""
		self.__printBanner("getting extreme values", ds)
		data = self.getNumericData(ds)
		evalues = list()
		for _ in range(ensamp):
			values = selectRandomSubListFromListWithRepl(data, nsamp)
			if polarity == "max":
				evalues.append(max(values))
			else:
				evalues.append(min(values))
		if doPlotDistr:
			plt.hist(evalues, bins=nbins, cumulative=False, density=True)
			plt.show()
		result = self.__printResult("extremeValues", evalues)
		return result


	def getEntropy(self, ds,  nbins=20):
		"""
		get entropy
		
		Parameters
			ds: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting entropy", ds)
		data = self.getNumericData(ds)
		result = self.getFreqDistr(data, nbins)
		entropy = sta.entropy(result["frequency"])
		result = self.__printResult("entropy", entropy)
		return result

	def getRelEntropy(self, ds1,  ds2, nbins=20):
		"""
		get relative entropy or KL divergence with both data sets numeric
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting relative entropy or KL divergence", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		result1 = self .getFeqDistr(data1, nbins)
		freq1  = result1["frequency"]
		result2 = self .getFeqDistr(data2, nbins)
		freq2  = result2["frequency"]
		entropy = sta.entropy(freq1, freq2)
		result = self.__printResult("relEntropy", entropy)
		return result

	def getAnyEntropy(self, ds,  dt, nbins=20):
		"""
		get entropy of any data typr numeric or categorical
		
		Parameters
			ds: data set name or list or numpy array
			dt : data type num or cat
			nbins: num of bins
		"""
		entropy = self.getEntropy(ds, nbins)["entropy"] if dt == "num" else self.getStatsCat(ds)["entropy"]
		result = self.__printResult("entropy", entropy)
		return result

	def getJointEntropy(self, ds1, ds2, nbins=20):
		"""
		get joint entropy with both data sets numeric
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting join entropy", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		hist, xedges, yedges = np.histogram2d(data1, data2, bins=nbins)
		hist = hist.flatten()
		ssize = len(data1)
		hist = hist / ssize
		entropy = sta.entropy(hist)
		result = self.__printResult("jointEntropy", entropy)
		return result
		

	def getAllNumMutualInfo(self, ds1,  ds2, nbins=20):
		"""
		get mutual information for both numeric data
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			nbins: num of bins
		"""
		self.__printBanner("getting mutual information", ds1, ds2)
		en1 = self.getEntropy(ds1,nbins)
		en2 = self.getEntropy(ds2,nbins)
		en = self.getJointEntropy(ds1, ds2, nbins)

		mutInfo = en1["entropy"] + en2["entropy"] - en["jointEntropy"]
		result = self.__printResult("mutInfo", mutInfo)
		return result


	def getNumCatMutualInfo(self, nds, cds ,nbins=20):
		"""
		get mutiual information between numeric and categorical data
		
		Parameters
			nds: numeric data set name or list or numpy array
			cds: categoric data set name or list 
			nbins: num of bins
		"""
		self.__printBanner("getting mutual information of numerical and categorical data", nds, cds)
		ndata = self.getNumericData(nds)
		cds = self.getCatData(cds)
		nentr = self.getEntropy(nds)["entropy"]
		
		#conditional entropy
		cdistr = self.getStatsCat(cds)["distr"]
		grdata = self.getGroupByData(nds, cds, True)["groupedData"]
		cnentr = 0
		for gr, data in grdata.items():
			self.addListNumericData(data, "grdata")	
			gnentr = self.getEntropy("grdata")["entropy"]
			cnentr += gnentr * cdistr[gr]
			
		mutInfo = nentr - cnentr
		result = self.__printResult("mutInfo", mutInfo, "entropy", nentr, "condEntropy", cnentr)
		return result
		 
	def getTwoCatMutualInfo(self, cds1, cds2):
		"""
		get mutiual information between 2 categorical data sets
		
		Parameters
			cds1 : categoric data set name or list 
			cds2 : categoric data set name or list 
		"""
		self.__printBanner("getting mutual information of two categorical data sets", cds1, cds2)
		cdata1 = self.getCatData(cds1)
		cdata2 = self.getCatData(cds1)
		centr = self.getStatsCat(cds1)["entropy"]
		
		#conditional entropy
		cdistr = self.getStatsCat(cds2)["distr"]
		grdata = self.getGroupByData(cds1, cds2, True)["groupedData"]
		ccentr = 0
		for gr, data in grdata.items():
			self.addListCatData(data, "grdata")	
			gcentr = self.getStatsCat("grdata")["entropy"]
			ccentr += gcentr * cdistr[gr]
			
		mutInfo = centr - ccentr
		result = self.__printResult("mutInfo", mutInfo, "entropy", centr, "condEntropy", ccentr)
		return result

	def getMutualInfo(self, dst, nbins=20):
		"""
		get mutiual information between 2 data sets,any combination numerical and categorical
		
		Parameters
			dst : data source , data type, data source , data type
			nbins : num of bins
		"""
		assertEqual(len(dst), 4, "invalid data source and data type list size")
		dtypes = ["num", "cat"]
		assertInList(dst[1], dtypes, "invalid data type")
		assertInList(dst[3], dtypes, "invalid data type")
		self.__printBanner("getting mutual information of any mix numerical and categorical data", dst[0], dst[2])
		
		if dst[1] == "num":
			mutInfo = self.getAllNumMutualInfo(dst[0], dst[2], nbins)["mutInfo"] if dst[3] == "num" \
			else self.getNumCatMutualInfo(dst[0], dst[2], nbins)["mutInfo"]
		else:
			mutInfo = self.getNumCatMutualInfo(dst[2], dst[0], nbins)["mutInfo"] if dst[3] == "num" \
			else self.getTwoCatMutualInfo(dst[2], dst[0])["mutInfo"]
		
		result = self.__printResult("mutInfo", mutInfo)
		return result


	def getCondMutualInfo(self, dst, nbins=20):
		"""
		get conditional  mutiual information between 2 data sets,any combination numerical and categorical
		
		Parameters
			dst : data source , data type, data source , data type, data source , data type
			nbins : num of bins
		"""
		assertEqual(len(dst), 6, "invalid data source and data type list size")
		dtypes = ["num", "cat"]
		assertInList(dst[1], dtypes, "invalid data type")
		assertInList(dst[3], dtypes, "invalid data type")
		assertInList(dst[5], dtypes, "invalid data type")
		self.__printBanner("getting conditional mutual information of any mix numerical and categorical data", dst[0], dst[2])
		
		if dst[5] == "cat":
			cdistr = self.getStatsCat(dst[4])["distr"]
			grdata1 = self.getGroupByData(dst[0], dst[4], True)["groupedData"]
			grdata2 = self.getGroupByData(dst[2], dst[4], True)["groupedData"]
			
		else:
			gdata = self.getNumericData(dst[4])
			hist = Histogram.createWithNumBins(gdata, nbins)
			cdistr = hist.distr()
			grdata1 = self.getGroupByData(dst[0], dst[4], False)["groupedData"]
			grdata2 = self.getGroupByData(dst[2], dst[4], False)["groupedData"]


		cminfo = 0
		for gr in grdata1.keys():
			data1 = grdata1[gr]
			data2 = grdata2[gr]
			if dst[1] == "num":
				self.addListNumericData(data1, "grdata1")
			else:
				self.addListCatData(data1, "grdata1")
					
			if dst[3] == "num":
				self.addListNumericData(data2, "grdata2")
			else:
				self.addListCatData(data2, "grdata2")
			gdst = ["grdata1", dst[1], "grdata2", dst[3]]
			minfo = self.getMutualInfo(gdst, nbins)["mutInfo"] 
			cminfo += minfo * cdistr[gr]
		
		result = self.__printResult("condMutInfo", cminfo)
		return result
		
	def getPercentile(self, ds, value):
		"""
		gets percentile
		
		Parameters
			ds: data set name or list or numpy array
			value: the value
		"""
		self.__printBanner("getting percentile", ds)
		data = self.getNumericData(ds)
		percent = sta.percentileofscore(data, value)
		result = self.__printResult("value", value, "percentile", percent)
		return result

	def getValueRangePercentile(self, ds, value1, value2):
		"""
		gets percentile
		
		Parameters
			ds: data set name or list or numpy array
			value1: first value
			value2: second value
		"""
		self.__printBanner("getting percentile difference for value range", ds)
		if value1 < value2:
			v1 = value1
			v2 = value2
		else:
			v1 = value2
			v2 = value1
		data = self.getNumericData(ds)
		per1 = sta.percentileofscore(data, v1)
		per2 = sta.percentileofscore(data, v2)
		result = self.__printResult("valueFirst", value1, "valueSecond", value2, "percentileDiff", per2 - per1)
		return result

	def getValueAtPercentile(self, ds, percent):
		"""
		gets value at percentile
		
		Parameters
			ds: data set name or list or numpy array
			percent: percentile
		"""
		self.__printBanner("getting value at percentile", ds)
		data = self.getNumericData(ds)
		assert isInRange(percent, 0, 100), "percent should be between 0 and 100"
		value = sta.scoreatpercentile(data, percent)
		result = self.__printResult("value", value, "percentile", percent)
		return result

	def getLessThanValues(self, ds, cvalue):
		"""
		gets values less than given value
		
		Parameters
			ds: data set name or list or numpy array
			cvalue: condition value
		"""
		self.__printBanner("getting values less than", ds)
		fdata = self.__getCondValues(ds, cvalue, "lt")
		result = self.__printResult("count", len(fdata),  "lessThanvalues", fdata )
		return result


	def getGreaterThanValues(self, ds, cvalue):
		"""
		gets values greater than given value
		
		Parameters
			ds: data set name or list or numpy array
			cvalue: condition value
		"""
		self.__printBanner("getting values greater than", ds)
		fdata = self.__getCondValues(ds, cvalue, "gt")
		result = self.__printResult("count", len(fdata), "greaterThanvalues", fdata )
		return result

	def __getCondValues(self, ds, cvalue, cond):
		"""
		gets cinditional values
		
		Parameters
			ds: data set name or list or numpy array
			cvalue: condition value
			cond: condition
		"""
		data = self.getNumericData(ds)
		if cond == "lt":
			ind = np.where(data < cvalue)
		else:
			ind = np.where(data > cvalue)
		fdata = data[ind]
		return fdata

	def getUniqueValueCounts(self, ds, maxCnt=10):
		"""
		gets unique values and counts
		
		Parameters
			ds: data set name or list or numpy array
			maxCnt; max value count pairs to return
		"""
		self.__printBanner("getting unique values and counts", ds)
		data = self.getNumericData(ds)
		values, counts = sta.find_repeats(data)
		cardinality = len(values)
		vc = list(zip(values, counts))
		vc.sort(key = lambda v : v[1], reverse = True)
		result = self.__printResult("cardinality", cardinality,  "vunique alues and repeat counts", vc[:maxCnt])
		return result

	def getCatUniqueValueCounts(self, ds, maxCnt=10):
		"""
		gets unique categorical values and counts
		
		Parameters
			ds: data set name or list or numpy array
			maxCnt: max value count pairs to return
		"""
		self.__printBanner("getting unique categorical values and counts", ds)
		data = self.getCatData(ds)
		series = pd.Series(data)
		uvalues = series.value_counts()
		values = uvalues.index.tolist()
		counts = uvalues.tolist()
		vc = list(zip(values, counts))
		vc.sort(key = lambda v : v[1], reverse = True)
		result = self.__printResult("cardinality", len(values),  "unique values and repeat counts", vc[:maxCnt])
		return result

	def getCatAlphaValueCounts(self, ds):
		"""
		gets alphabetic value count
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting alphabetic value counts", ds)
		data = self.getCatData(ds)
		series = pd.Series(data)
		flags = series.str.isalpha().tolist()
		count = sum(flags)
		result = self.__printResult("alphabeticValueCount", count)
		return result
		

	def getCatNumValueCounts(self, ds):
		"""
		gets numeric value count
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting numeric value counts", ds)
		data = self.getCatData(ds)
		series = pd.Series(data)
		flags = series.str.isnumeric().tolist()
		count = sum(flags)
		result = self.__printResult("numericValueCount", count)
		return result


	def getCatAlphaNumValueCounts(self, ds):
		"""
		gets alpha numeric value count
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting alpha numeric value counts", ds)
		data = self.getCatData(ds)
		series = pd.Series(data)
		flags = series.str.isalnum().tolist()
		count = sum(flags)
		result = self.__printResult("alphaNumericValueCount", count)
		return result

	def getCatAllCharCounts(self, ds):
		"""
		gets alphabetic, numeric and special char count list
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting alphabetic, numeric and special  char counts", ds)
		data = self.getCatData(ds)
		counts = list()
		for d in data:
			r = getAlphaNumCharCount(d)
			counts.append(r)
		result = self.__printResult("allTypeCharCounts", counts)
		return result

	def getCatAlphaCharCounts(self, ds):
		"""
		gets alphabetic char count list
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting alphabetic char counts", ds)
		data = self.getCatData(ds)
		counts = self.getCatAllCharCounts(ds)["allTypeCharCounts"]
		counts = list(map(lambda r : r[0], counts))
		result = self.__printResult("alphaCharCounts", counts)
		return result
	
	def getCatNumCharCounts(self, ds):
		"""
		gets numeric char count list
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting numeric char counts", ds)
		data = self.getCatData(ds)
		counts = self.getCatAllCharCounts(ds)["allTypeCharCounts"]
		counts = list(map(lambda r : r[1], counts))
		result = self.__printResult("numCharCounts", counts)
		return result

	def getCatSpecialCharCounts(self, ds):
		"""
		gets special char count list
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting special char counts", ds)
		counts = self.getCatAllCharCounts(ds)["allTypeCharCounts"]
		counts = list(map(lambda r : r[2], counts))
		result = self.__printResult("specialCharCounts", counts)
		return result

	def getCatAlphaCharCountStats(self, ds):
		"""
		gets alphabetic char count stats
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting alphabetic char count stats", ds)
		counts = self.getCatAlphaCharCounts(ds)["alphaCharCounts"]
		nz = counts.count(0)
		st = self.__getBasicStats(np.array(counts))
		result = self.__printResult("mean", st[0], "std dev", st[1], "max", st[2], "min", st[3], "zeroCount", nz)
		return result
		
	def getCatNumCharCountStats(self, ds):
		"""
		gets numeric char count stats
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting numeric char count stats", ds)
		counts = self.getCatNumCharCounts(ds)["numCharCounts"]
		nz = counts.count(0)
		st = self.__getBasicStats(np.array(counts))
		result = self.__printResult("mean", st[0], "std dev", st[1], "max", st[2], "min", st[3], "zeroCount", nz)
		return result

	def getCatSpecialCharCountStats(self, ds):
		"""
		gets special char count stats
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting special char count stats", ds)
		counts = self.getCatSpecialCharCounts(ds)["specialCharCounts"]
		nz = counts.count(0)
		st = self.__getBasicStats(np.array(counts))
		result = self.__printResult("mean", st[0], "std dev", st[1], "max", st[2], "min", st[3], "zeroCount", nz)
		return result

	def getCatFldLenStats(self, ds):
		"""
		gets field length stats
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting field length stats", ds)
		data = self.getCatData(ds)
		le = list(map(lambda d: len(d), data))
		st = self.__getBasicStats(np.array(le))
		result = self.__printResult("mean", st[0], "std dev", st[1], "max", st[2], "min", st[3])
		return result

	def getCatCharCountStats(self, ds, ch):
		"""
		gets specified char ocuurence count stats
		
		Parameters
			ds: data set name or list or numpy array
			ch : character
		"""
		self.__printBanner("getting field length stats", ds)
		data = self.getCatData(ds)
		counts = list(map(lambda d: d.count(ch), data))
		nz = counts.count(0)
		st = self.__getBasicStats(np.array(counts))
		result = self.__printResult("mean", st[0], "std dev", st[1], "max", st[2], "min", st[3], "zeroCount", nz)
		return result

	def getStats(self, ds, nextreme=5):
		"""
		gets summary statistics
		
		Parameters
			ds: data set name or list or numpy array
			nextreme: num of extreme values
		"""
		self.__printBanner("getting summary statistics", ds)
		data = self.getNumericData(ds)
		stat = dict()
		stat["length"] = len(data)
		stat["min"] = data.min()
		stat["max"] = data.max()
		series = pd.Series(data)
		stat["n smallest"] = series.nsmallest(n=nextreme).tolist()
		stat["n largest"] = series.nlargest(n=nextreme).tolist()
		stat["mean"] = data.mean()
		stat["median"] = np.median(data)
		mode, modeCnt = sta.mode(data)
		stat["mode"] = mode[0]
		stat["mode count"] = modeCnt[0]
		stat["std"] = np.std(data)
		stat["skew"] = sta.skew(data)
		stat["kurtosis"] = sta.kurtosis(data)
		stat["mad"] = sta.median_absolute_deviation(data)
		self.pp.pprint(stat)
		return stat

	def getStatsCat(self, ds):
		"""
		gets summary statistics for categorical data
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting summary statistics for categorical data", ds)
		data = self.getCatData(ds)
		ch = CatHistogram()
		for d in data:
			ch.add(d)
		mode = ch.getMode()
		entr = ch.getEntropy()
		uvalues = ch.getUniqueValues()
		distr = ch.getDistr()
		result = self.__printResult("entropy", entr, "mode", mode, "uniqueValues", uvalues, "distr", distr)
		return result
		

	def getGroupByData(self, ds, gds, gdtypeCat, numBins=20):
		"""
		group by 

		Parameters
			ds: data set name or list or numpy array
			gds: group by data set name or list or numpy array
			gdtpe : group by data type
		"""
		self.__printBanner("getting group by data", ds)
		data = self.getAnyData(ds)
		if gdtypeCat:
			gdata = self.getCatData(gds)
		else:
			gdata = self.getNumericData(gds)
			hist = Histogram.createWithNumBins(gdata, numBins)
			gdata = list(map(lambda d : hist.bin(d), gdata))
			
		self.ensureSameSize([data, gdata])
		groups = dict()
		for g,d in zip(gdata, data):
			appendKeyedList(groups, g, d)
				
		ve = self.verbose 
		self.verbose = False
		result = self.__printResult("groupedData", groups)
		self.verbose = ve
		return result
		
	def getDifference(self, ds, order, doPlot=False):
		"""
		gets difference of given order
		
		Parameters
			ds: data set name or list or numpy array
			order: order of difference
			doPlot : True for plot
		"""
		self.__printBanner("getting difference of given order", ds)
		data = self.getNumericData(ds)
		diff = difference(data, order)
		if doPlot:
			drawLine(diff)
		return diff

	def getTrend(self, ds, doPlot=False):
		"""
		get trend
		
		Parameters
			ds: data set name or list or numpy array
			doPlot: true if plotting needed
		"""
		self.__printBanner("getting trend")
		data = self.getNumericData(ds)
		sz = len(data)
		X = list(range(0, sz))
		X = np.reshape(X, (sz, 1))
		model = LinearRegression()
		model.fit(X, data)
		trend = model.predict(X)
		sc = model.score(X, data)
		coef = model.coef_
		intc = model.intercept_
		result = self.__printResult("coeff", coef, "intercept", intc,  "r square error", sc,  "trend", trend)
		
		if doPlot:
			plt.plot(data)
			plt.plot(trend)
			plt.show()
		return result

	def getDiffSdNoisiness(self, ds):
		"""
		get noisiness based on std dev of first order difference
		
		Parameters
			ds: data set name or list or numpy array
		"""
		diff = self.getDifference(ds, 1)
		noise = np.std(np.array(diff))
		result = self.__printResult("noisiness", noise)
		return result
		
	def getMaRmseNoisiness(self, ds, wsize=5):
		"""
		gets noisiness based on RMSE with moving average
		
		Parameters
			ds: data set name or list or numpy array
			wsize : window size
		"""
		assert wsize % 2 == 1, "window size must be odd"
		data = self.getNumericData(ds)
		wind = data[:wsize]
		wstat = SlidingWindowStat.initialize(wind.tolist())
		
		whsize = int(wsize / 2)
		beg = whsize
		end = len(data) - whsize - 1
		sumSq = 0.0
		mean = wstat.getStat()[0]
		diff = data[beg] - mean
		sumSq += diff * diff
		for i in range(beg + 1, end, 1):
			mean = wstat.addGetStat(data[i + whsize])[0]
			diff = data[i] - mean
			sumSq += (diff * diff)
			
		noise = math.sqrt(sumSq / (len(data) - 2 * whsize))	
		result = self.__printResult("noisiness", noise)
		return result
		
		
	def deTrend(self, ds, trend, doPlot=False):
		"""
		de trend
		
		Parameters
			ds: data set name or list or numpy array
			ternd : trend data
			doPlot: true if plotting needed
		"""
		self.__printBanner("doing de trend", ds)
		data = self.getNumericData(ds)
		sz = len(data)
		detrended =  list(map(lambda i : data[i]-trend[i], range(sz)))
		if doPlot:
			drawLine(detrended)
		return detrended

	def getTimeSeriesComponents(self, ds, model, freq, summaryOnly, doPlot=False):
		"""
		extracts trend, cycle and residue components of time series
		
		Parameters
			ds: data set name or list or numpy array
			model : model type
			freq : seasnality period
			summaryOnly : True if only summary needed in output
			doPlot: true if plotting needed
		"""
		self.__printBanner("extracting trend, cycle and residue components of time series", ds)
		assert model == "additive" or model == "multiplicative", "model must be additive or multiplicative"
		data = self.getNumericData(ds)
		res = seasonal_decompose(data, model=model, period=freq)
		if doPlot:
			res.plot()
			plt.show()

		#summar of componenets
		trend = np.array(removeNan(res.trend))
		trendMean = trend.mean()
		trendSlope = (trend[-1] - trend[0]) / (len(trend) - 1)
		seasonal = np.array(removeNan(res.seasonal))
		seasonalAmp = (seasonal.max() - seasonal.min()) / 2
		resid = np.array(removeNan(res.resid))
		residueMean = resid.mean()
		residueStdDev = np.std(resid)

		if summaryOnly:
			result = self.__printResult("trendMean", trendMean, "trendSlope", trendSlope, "seasonalAmp", seasonalAmp,
			"residueMean", residueMean, "residueStdDev", residueStdDev)
		else:
			result = self.__printResult("trendMean", trendMean, "trendSlope", trendSlope, "seasonalAmp", seasonalAmp,
			"residueMean", residueMean, "residueStdDev", residueStdDev, "trend", res.trend, "seasonal", res.seasonal,
			"residual", res.resid)
		return result

	def getGausianMixture(self, ncomp, cvType, ninit, *dsl):
		"""
		finds gaussian mixture parameters
		
		Parameters
			ncomp : num of gaussian componenets
			cvType : co variance type
			ninit: num of intializations
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting gaussian mixture parameters", *dsl)
		assertInList(cvType, ["full", "tied", "diag", "spherical"], "invalid covariance type")
		dmat = self.__stackData(*dsl)
		
		gm = GaussianMixture(n_components=ncomp,  covariance_type=cvType, n_init=ninit)
		gm.fit(dmat)
		weights = gm.weights_
		means = gm.means_
		covars = gm.covariances_
		converged = gm.converged_
		niter = gm.n_iter_
		aic = gm.aic(dmat)
		result = self.__printResult("weights", weights, "mean", means, "covariance", covars, "converged", converged, "num iterations", niter, "aic", aic)
		return result
		
	def getKmeansCluster(self, nclust, ninit, *dsl):
		"""
		gets cluster parameters
		
		Parameters
			nclust : num of clusters
			ninit: num of intializations
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting kmean cluster parameters", *dsl)
		dmat = self.__stackData(*dsl)
		nsamp = dmat.shape[0]
		
		km = KMeans(n_clusters=nclust, n_init=ninit)
		km.fit(dmat)
		centers = km.cluster_centers_
		avdist = sqrt(km.inertia_ / nsamp)
		niter = km.n_iter_
		score = km.score(dmat)
		result = self.__printResult("centers", centers, "average distance", avdist, "num iterations", niter, "score", score)
		return result

	def getPrincComp(self, ncomp, *dsl):
		"""
		finds pricipal componenet parameters
		
		Parameters
			ncomp : num of pricipal componenets
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting principal componenet parameters", *dsl)
		dmat = self.__stackData(*dsl)
		nfeat = dmat.shape[1]
		assertGreater(nfeat, 1, "requires multiple features")
		assertLesserEqual(ncomp, nfeat, "num of componenets greater than num of features")
		
		pca = PCA(n_components=ncomp)
		pca.fit(dmat)
		comps = pca.components_
		var = pca.explained_variance_
		varr = pca.explained_variance_ratio_
		svalues = pca.singular_values_
		result = self.__printResult("componenets", comps, "variance", var, "variance ratio", varr, "singular values", svalues)
		return result

	def getOutliersWithIsoForest(self, contamination,  *dsl):
		"""
		finds outliers using isolation forest
		
		Parameters
			contamination : proportion of outliers in the data set
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting outliers using isolation forest", *dsl)
		assert contamination >= 0 and contamination <= 0.5, "contamination outside valid range"
		dmat = self.__stackData(*dsl)

		isf = IsolationForest(contamination=contamination, behaviour="new")
		ypred = isf.fit_predict(dmat)
		mask = ypred == -1
		doul = dmat[mask, :]
		mask = ypred != -1
		dwoul = dmat[mask, :]
		result = self.__printResult("numOutliers", doul.shape[0], "outliers", doul, "dataWithoutOutliers", dwoul)	
		return result

	def getOutliersWithLocalFactor(self, contamination,  *dsl):
		"""
		gets outliers using local outlier factor
		
		Parameters
			contamination : proportion of outliers in the data set
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting outliers using local outlier factor", *dsl)
		assert contamination >= 0 and contamination <= 0.5, "contamination outside valid range"
		dmat = self.__stackData(*dsl)

		lof = LocalOutlierFactor(contamination=contamination)
		ypred = lof.fit_predict(dmat)
		mask = ypred == -1
		doul = dmat[mask, :]
		mask = ypred != -1
		dwoul = dmat[mask, :]
		result = self.__printResult("numOutliers", doul.shape[0], "outliers", doul, "dataWithoutOutliers", dwoul)	
		return result

	def getOutliersWithSupVecMach(self, nu,  *dsl):
		"""
		gets outliers using one class svm
		
		Parameters
			nu : upper bound on the fraction of training errors and a lower bound of the fraction of support vectors
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting outliers using one class svm", *dsl)
		assert nu >= 0 and nu <= 0.5, "error upper bound outside valid range"
		dmat = self.__stackData(*dsl)

		svm = OneClassSVM(nu=nu)
		ypred = svm.fit_predict(dmat)
		mask = ypred == -1
		doul = dmat[mask, :]
		mask = ypred != -1
		dwoul = dmat[mask, :]
		result = self.__printResult("numOutliers", doul.shape[0], "outliers", doul, "dataWithoutOutliers", dwoul)	
		return result

	def getOutliersWithCovarDeterminant(self, contamination,  *dsl):
		"""
		gets outliers using covariance determinan
		
		Parameters
			contamination : proportion of outliers in the data set
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting outliers using using covariance determinant", *dsl)
		assert contamination >= 0 and contamination <= 0.5, "contamination outside valid range"
		dmat = self.__stackData(*dsl)

		lof = EllipticEnvelope(contamination=contamination)
		ypred = lof.fit_predict(dmat)
		mask = ypred == -1
		doul = dmat[mask, :]
		mask = ypred != -1
		dwoul = dmat[mask, :]
		result = self.__printResult("numOutliers", doul.shape[0], "outliers", doul, "dataWithoutOutliers", dwoul)	
		return result

	def getOutliersWithZscore(self, ds, zthreshold, stats=None):
		"""
		gets outliers using zscore
		
		Parameters
			ds: data set name or list or numpy array
			zthreshold : z score threshold
			stats : tuple cintaining mean and std dev
		"""
		self.__printBanner("getting outliers using zscore", ds)
		data = self.getNumericData(ds)
		if stats is None:
			mean = data.mean()
			sd = np.std(data)
		else:
			mean = stats[0]
			sd = stats[1]
			
		zs = list(map(lambda d : abs((d - mean) / sd), data))
		outliers = list(filter(lambda r : r[1] > zthreshold, enumerate(zs)))
		result = self.__printResult("outliers", outliers)	
		return result

	def getOutliersWithRobustZscore(self, ds, zthreshold, stats=None):
		"""
		gets outliers using robust zscore
		
		Parameters
			ds: data set name or list or numpy array
			zthreshold : z score threshold
			stats : tuple containing median and median absolute deviation
		"""
		self.__printBanner("getting outliers using robust zscore", ds)
		data = self.getNumericData(ds)
		if stats is None:
			med = np.median(data)
			dev = np.array(list(map(lambda d : abs(d - med), data)))
			mad = 1.4296 *  np.median(dev)
		else:
			med = stats[0]
			mad = stats[1]
		
		rzs = list(map(lambda d : abs((d - med) / mad), data))
		outliers = list(filter(lambda r : r[1] > zthreshold, enumerate(rzs)))
		result = self.__printResult("outliers", outliers)	
		return result
		

	def getSubsequenceOutliersWithDissimilarity(self, subSeqSize, ds):
		"""
		gets subsequence outlier with subsequence pairwise disimilarity
		
		Parameters
			subSeqSize : sub sequence size
			ds: data set name or list or numpy array
		"""
		self.__printBanner("doing sub sequence anomaly detection with dissimilarity", ds)
		data = self.getNumericData(ds)
		sz = len(data)
		dist = dict()
		minDist = dict()
		for i in range(sz - subSeqSize):
			#first window
			w1 = data[i : i + subSeqSize]
			dmin = None
			for j in range(sz - subSeqSize):
				#second window not overlapping with the first
				if j + subSeqSize <=i or j >= i + subSeqSize:
					w2 = data[j : j + subSeqSize]
					k = (j,i)
					if k in dist:
						d = dist[k]
					else:
						d = euclideanDistance(w1,w2)
						k = (i,j)
						dist[k] = d
					if dmin is None:
						dmin = d
					else:
						dmin = d if d < dmin else dmin
			minDist[i] = dmin
		
		#find max of min
		dmax = None
		offset = None
		for k in minDist.keys():
			d = minDist[k]
			if dmax is None:
				dmax = d
				offset = k
			else:
				if d > dmax:
					dmax = d
					offset = k
		result = self.__printResult("subSeqOffset", offset, "outlierScore", dmax)	
		return result
	
	def getNullCount(self, ds):
		"""
		get count of null fields
		
		Parameters
			ds : data set name or list or numpy array with data
		"""
		self.__printBanner("getting null value count", ds)
		if type(ds) == str:
			assert ds in self.dataSets, "data set {} does not exist, please add it first".format(ds)
			data =  self.dataSets[ds]
			ser = pd.Series(data)
		elif type(ds) == list or type(ds) == np.ndarray:
			ser = pd.Series(ds)
			data = ds
		else:
			raise ValueError("invalid data type")
		nv = ser.isnull().tolist()
		nullCount = nv.count(True)
		nullFraction = nullCount / len(data)
		result = self.__printResult("nullFraction", nullFraction, "nullCount", nullCount)
		return result


	def fitLinearReg(self, dsx, ds, doPlot=False):
		"""
		fit  linear regression 
		
		Parameters
			dsx: x data set name or None
			ds: data set name or list or numpy array
			doPlot: true if plotting needed
		"""
		self.__printBanner("fitting linear regression", ds)
		data = self.getNumericData(ds)
		if dsx is None:
			x = np.arange(len(data))
		else:
			x = self.getNumericData(dsx)
		slope, intercept, rvalue, pvalue, stderr = sta.linregress(x, data)
		result = self.__printResult("slope", slope, "intercept", intercept, "rvalue", rvalue, "pvalue", pvalue, "stderr", stderr)
		if doPlot:
			self.regFitPlot(x, data, slope, intercept)
		return result

	def fitSiegelRobustLinearReg(self, ds, doPlot=False):
		"""
		siegel robust linear regression fit based on median
		
		Parameters
			ds: data set name or list or numpy array
			doPlot: true if plotting needed
		"""
		self.__printBanner("fitting siegel robust linear regression  based on median", ds)
		data = self.getNumericData(ds)
		slope , intercept = sta.siegelslopes(data)
		result = self.__printResult("slope", slope, "intercept", intercept)
		if doPlot:
			x = np.arange(len(data))
			self.regFitPlot(x, data, slope, intercept)
		return result

	def fitTheilSenRobustLinearReg(self, ds, doPlot=False):
		"""
		thiel sen  robust linear fit regression based on median
		
		Parameters
			ds: data set name or list or numpy array
			doPlot: true if plotting needed
		"""
		self.__printBanner("fitting thiel sen  robust linear regression based on median", ds)
		data = self.getNumericData(ds)
		slope, intercept, loSlope, upSlope = sta.theilslopes(data)
		result = self.__printResult("slope", slope, "intercept", intercept, "lower slope", loSlope, "upper slope", upSlope)
		if doPlot:
			x = np.arange(len(data))
			self.regFitPlot(x, data, slope, intercept)
		return result

	def plotRegFit(self, x, y, slope, intercept):
		"""
		plot linear rgeression fit line
		
		Parameters
			x : x values
			y : y values
			slope : slope
			intercept : intercept
		"""
		self.__printBanner("plotting linear rgeression fit line")
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(x, y, "b.")
		ax.plot(x, intercept + slope * x, "r-")
		plt.show()

	def getRegFit(self, xvalues, yvalues, slope, intercept):
		"""
		gets fitted line and residue
		
		Parameters
			x : x values
			y : y values
			slope : regression slope
			intercept : regressiob intercept
		"""
		yfit = list()
		residue = list()
		for x,y in zip(xvalues, yvalues):
			yf = x * slope + intercept
			yfit.append(yf)
			r = y - yf
			residue.append(r)
		result = self.__printResult("fitted line", yfit, "residue", residue)
		return result

	def getInfluentialPoints(self, dsx, dsy):
		"""
		gets influential points in regression model with Cook's distance
		
		Parameters
			dsx : data set name or list or numpy array for x
			dsy : data set name or list or numpy array for y
		"""
		self.__printBanner("finding influential points for linear regression", dsx, dsy)
		y = self.getNumericData(dsy)
		x = np.arange(len(data)) if dsx is None else self.getNumericData(dsx)
		model = sm.OLS(y, x).fit()
		np.set_printoptions(suppress=True)
		influence = model.get_influence()
		cooks = influence.cooks_distance
		result = self.__printResult("Cook distance", cooks)
		return result
	
	def getCovar(self, *dsl):
		"""
		gets covariance
		
		Parameters
			dsl: list of data set name or list or numpy array
		"""
		self.__printBanner("getting covariance", *dsl)
		data = list(map(lambda ds : self.getNumericData(ds), dsl))
		self.ensureSameSize(data)
		data = np.vstack(data)
		cv = np.cov(data)
		print(cv)
		return cv

	def getPearsonCorr(self, ds1, ds2, sigLev=.05):
		"""
		gets pearson correlation coefficient 
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
		"""
		self.__printBanner("getting pearson correlation coefficient ", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		stat, pvalue = sta.pearsonr(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def getSpearmanRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		gets spearman correlation coefficient
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("getting spearman correlation coefficient",ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		stat, pvalue = sta.spearmanr(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getKendalRankCorr(self, ds1, ds2, sigLev=.05):
		"""
		kendall’s tau, a correlation measure for ordinal data
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("getting kendall’s tau, a correlation measure for ordinal data", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		stat, pvalue = sta.kendalltau(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getPointBiserialCorr(self, ds1, ds2, sigLev=.05):
		"""
		point biserial  correlation  between binary and numeric
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("getting point biserial correlation  between binary and numeric", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		assert isBinary(data1), "first data set is not binary"
		self.ensureSameSize([data1, data2])
		stat, pvalue = sta.pointbiserialr(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getConTab(self, ds1, ds2):
		"""
		get contingency table for categorical data pair
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
		"""
		self.__printBanner("getting contingency table for categorical data", ds1, ds2)
		data1 = self.getCatData(ds1)
		data2 = self.getCatData(ds2)
		self.ensureSameSize([data1, data2])
		crosstab = pd.crosstab(pd.Series(data1), pd.Series(data2), margins = False)
		ctab = crosstab.values
		print("contingency table")
		print(ctab)
		return ctab

	def getChiSqCorr(self, ds1, ds2, sigLev=.05):
		"""
		chi square correlation for  categorical	data pair
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("getting chi square correlation for  two categorical", ds1, ds2)
		ctab = self.getConTab(ds1, ds2)
		stat, pvalue, dof, expctd = sta.chi2_contingency(ctab)
		result = self.__printResult("stat", stat, "pvalue", pvalue, "dof", dof, "expected", expctd)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result

	def getAnovaCorr(self, ds1, ds2, grByCol, sigLev=.05):
		"""
		anova correlation for  numerical categorical	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			grByCol : group by column
			sigLev: statistical significance level
		"""
		self.__printBanner("anova correlation for numerical categorical", ds1, ds2)
		df = self.loadCatFloatDataFrame(ds1, ds2) if grByCol == 0 else self.loadCatFloatDataFrame(ds2, ds1)
		grByCol = 0
		dCol = 1
		grouped = df.groupby([grByCol])
		dlist =  list(map(lambda v : v[1].loc[:, dCol].values, grouped))
		stat, pvalue = sta.f_oneway(*dlist)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably uncorrelated", "probably correlated", sigLev)
		return result


	def plotAutoCorr(self, ds, lags, alpha, diffOrder=0):
		"""
		plots auto correlation
		
		Parameters
			ds: data set name or list or numpy array
			lags: num of lags
			alpha: confidence level
		"""
		self.__printBanner("plotting auto correlation", ds)
		data = self.getNumericData(ds)
		ddata = difference(data, diffOrder) if diffOrder > 0 else data
		tsaplots.plot_acf(ddata, lags = lags, alpha = alpha)
		plt.show()

	def getAutoCorr(self, ds, lags, alpha=.05):
		"""
		gets auts correlation
		
		Parameters
			ds: data set name or list or numpy array
			lags: num of lags
			alpha: confidence level
		"""
		self.__printBanner("getting auto correlation", ds)
		data = self.getNumericData(ds)
		autoCorr, confIntv  = stt.acf(data, nlags=lags, fft=False, alpha=alpha)
		result = self.__printResult("autoCorr", autoCorr, "confIntv", confIntv)
		return result


	def plotParAcf(self, ds, lags, alpha):
		"""
		partial auto correlation
		
		Parameters
			ds: data set name or list or numpy array
			lags: num of lags
			alpha: confidence level
		"""
		self.__printBanner("plotting partial auto correlation", ds)
		data = self.getNumericData(ds)
		tsaplots.plot_pacf(data, lags = lags, alpha = alpha)
		plt.show()

	def getParAutoCorr(self, ds, lags, alpha=.05):
		"""
		gets partial auts correlation
		
		Parameters
			ds: data set name or list or numpy array
			lags: num of lags
			alpha: confidence level
		"""
		self.__printBanner("getting partial auto correlation", ds)
		data = self.getNumericData(ds)
		partAutoCorr, confIntv  = stt.pacf(data, nlags=lags, alpha=alpha)
		result = self.__printResult("partAutoCorr", partAutoCorr, "confIntv", confIntv)
		return result

	def getHurstExp(self, ds, kind, doPlot=True):
		"""
		gets Hurst exponent of time series
		
		Parameters
			ds: data set name or list or numpy array
			kind: kind of data change, random_walk, price
			doPlot: True for plot
		"""
		self.__printBanner("getting Hurst exponent", ds)
		data = self.getNumericData(ds)
		h, c, odata = hurst.compute_Hc(data, kind=kind, simplified=False)
		if doPlot:
			f, ax = plt.subplots()
			ax.plot(odata[0], c * odata[0] ** h, color="deepskyblue")
			ax.scatter(odata[0], odata[1], color="purple")
			ax.set_xscale("log")
			ax.set_yscale("log")
			ax.set_xlabel("time interval")
			ax.set_ylabel("cum dev range and std dev ratio")
			ax.grid(True)
			plt.show()
			
		result = self.__printResult("hurstExponent", h, "hurstConstant", c)
		return result
		
	def approxEntropy(self, ds, m, r):
		"""
		gets apprx entroty of time series (ref: wikipedia)
		
		Parameters
			ds: data set name or list or numpy array
			m:  length of compared run of data
			r: filtering level
		"""
		self.__printBanner("getting approximate entropy", ds)
		ldata = self.getNumericData(ds)
		aent = abs(self.__phi(ldata, m + 1, r) - self.__phi(ldata, m, r))
		result = self.__printResult("approxEntropy", aent)
		return result
		
	def __phi(self, ldata, m, r):
		"""
		phi function for approximate entropy
		
		Parameters
			ldata: data array
			m:  length of compared run of data
			r: filtering level
		"""
		le = len(ldata)
		x = [[ldata[j] for j in range(i, i + m - 1 + 1)] for i in range(le - m + 1)]
		lex = len(x)
		c = list()
		for i in range(lex):
			cnt = 0
			for j in range(lex):
				cnt += (1 if maxListDist(x[i], x[j]) <= r else 0)
			cnt /= (le - m + 1.0)
			c.append(cnt)
		return sum(np.log(c)) / (le - m + 1.0)
		

	def oneSpaceEntropy(self, ds, scaMethod="zscale"):
		"""
		gets one space  entroty  (ref:  Estimating mutual information by Kraskov)
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting one space entropy", ds)
		data = self.getNumericData(ds)
		sdata = sorted(data)
		sdata = scaleData(sdata, scaMethod)
		su = 0
		n = len(sdata)
		for i in range(1, n, 1):
			t = abs(sdata[i] - sdata[i-1])
			if t > 0:
				su += log(t)
		su /= (n -1)
		#print(su)
		ose = digammaFun(n) - digammaFun(1) + su
		result = self.__printResult("entropy", ose)
		return result
		
	
	def plotCrossCorr(self, ds1, ds2, normed, lags):
		"""
		plots cross correlation 
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			normed: If True, input vectors are normalised to unit 
			lags: num of lags
		"""  
		self.__printBanner("plotting cross correlation between two numeric", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		plt.xcorr(data1, data2, normed=normed, maxlags=lags)
		plt.show()

	def getCrossCorr(self, ds1, ds2):
		"""
		gets cross correlation
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
		"""
		self.__printBanner("getting cross correlation", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		self.ensureSameSize([data1, data2])
		crossCorr = stt.ccf(data1, data2)
		result = self.__printResult("crossCorr", crossCorr)
		return result

	def getFourierTransform(self, ds):
		"""
		gets fast fourier transform
		
		Parameters
			ds: data set name or list or numpy array
		"""
		self.__printBanner("getting fourier transform", ds)
		data = self.getNumericData(ds)
		ft = np.fft.rfft(data)
		result = self.__printResult("fourierTransform", ft)
		return result


	def testStationaryAdf(self, ds, regression, autolag, sigLev=.05):
		"""
		Adf stationary test null hyp not stationary
		
		Parameters
			ds: data set name or list or numpy array
			regression: constant and trend order to include in regression
			autolag: method to use when automatically determining the lag
			sigLev: statistical significance level
		"""
		self.__printBanner("doing ADF stationary test", ds)
		relist = ["c","ct","ctt","nc"]
		assert regression in relist, "invalid regression value"
		alList = ["AIC", "BIC", "t-stat", None]
		assert autolag in alList, "invalid autolag value"

		data = self.getNumericData(ds)
		re = stt.adfuller(data, regression=regression, autolag=autolag)
		result = self.__printResult("stat", re[0], "pvalue", re[1] , "num lags", re[2] , "num observation for regression", re[3],
		"critial values", re[4])
		self.__printStat(re[0], re[1], "probably not stationary", "probably stationary", sigLev)
		return result

	def testStationaryKpss(self, ds, regression, nlags, sigLev=.05):
		"""
		Kpss stationary test null hyp  stationary
		
		Parameters
			ds: data set name or list or numpy array
			regression: constant and trend order to include in regression
			nlags : no of lags
			sigLev: statistical significance level
		"""
		self.__printBanner("doing KPSS stationary test", ds)
		relist = ["c","ct"]
		assert regression in relist, "invalid regression value"
		nlList =[None, "auto", "legacy"]
		assert nlags in nlList or type(nlags) == int, "invalid nlags value"


		data = self.getNumericData(ds)
		stat, pvalue, nLags, criticalValues = stt.kpss(data, regression=regression, lags=nlags)
		result = self.__printResult("stat", stat, "pvalue", pvalue, "num lags", nLags, "critial values", criticalValues)
		self.__printStat(stat, pvalue, "probably stationary", "probably not stationary", sigLev)
		return result

	def testNormalJarqBera(self, ds, sigLev=.05):
		"""
		jarque bera normalcy test
		
		Parameters
			ds: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing ajrque bera normalcy test", ds)
		data = self.getNumericData(ds)
		jb, jbpv, skew, kurtosis =  sstt.jarque_bera(data)
		result = self.__printResult("stat", jb, "pvalue", jbpv, "skew", skew, "kurtosis", kurtosis)
		self.__printStat(jb, jbpv, "probably gaussian", "probably not gaussian", sigLev)
		return result


	def testNormalShapWilk(self, ds, sigLev=.05):
		"""
		shapiro wilks normalcy test
		
		Parameters
			ds: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing shapiro wilks normalcy test", ds)
		data = self.getNumericData(ds)
		stat, pvalue = sta.shapiro(data)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testNormalDagast(self, ds, sigLev=.05):
		"""
		D’Agostino’s K square  normalcy test

		Parameters
			ds: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing D’Agostino’s K square  normalcy test", ds)
		data = self.getNumericData(ds)
		stat, pvalue = sta.normaltest(data)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably gaussian", "probably not gaussian", sigLev)
		return result

	def testDistrAnderson(self, ds, dist, sigLev=.05):
		"""
		Anderson test for normal, expon, logistic, gumbel, gumbel_l, gumbel_r
		
		Parameters
			ds: data set name or list or numpy array
			dist: type of distribution
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Anderson test for for various distributions", ds)
		diList = ["norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r", "extreme1"]
		assert dist in diList, "invalid distribution"

		data = self.getNumericData(ds)
		re = sta.anderson(data)
		slAlpha = int(100 * sigLev)
		msg = "significnt value not found"
		for i in range(len(re.critical_values)):
			sl, cv = re.significance_level[i], re.critical_values[i]
			if int(sl) == slAlpha:
				if re.statistic < cv:
					msg = "probably {} at the {:.3f} siginificance level".format(dist, sl)
				else:
					msg = "probably not {} at the {:.3f} siginificance level".format(dist, sl)
		result = self.__printResult("stat", re.statistic, "test", msg)
		print(msg)
		return result

	def testSkew(self, ds, sigLev=.05):
		"""
		test skew wrt  normal distr
		
		Parameters
			ds: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("testing skew wrt normal distr", ds)
		data = self.getNumericData(ds)
		stat, pvalue = sta.skewtest(data)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same skew as normal distribution", "probably not same skew as normal distribution", sigLev)
		return result

	def testTwoSampleStudent(self, ds1, ds2, sigLev=.05):
		"""
		student t 2 sample test
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing student t 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ttest_ind(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)
		return result

	def testTwoSampleKs(self, ds1, ds2, sigLev=.05):
		"""
		Kolmogorov Sminov 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Kolmogorov Sminov 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ks_2samp(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)


	def testTwoSampleMw(self, ds1, ds2, sigLev=.05):
		"""
		Mann-Whitney  2 sample statistic
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Mann-Whitney  2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.mannwhitneyu(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleWilcox(self, ds1, ds2, sigLev=.05):
		"""
		Wilcoxon Signed-Rank 2 sample statistic
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Wilcoxon Signed-Rank 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.wilcoxon(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)


	def testTwoSampleKw(self, ds1, ds2, sigLev=.05):
		"""
		Kruskal-Wallis 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Kruskal-Wallis 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.kruskal(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably snot ame distribution", sigLev)

	def testTwoSampleFriedman(self, ds1, ds2, ds3, sigLev=.05):
		"""
		Friedman 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Friedman 2 sample  test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		data3 = self.getNumericData(ds3)
		stat, pvalue = sta.friedmanchisquare(data1, data2, data3)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleEs(self, ds1, ds2, sigLev=.05):
		"""
		Epps Singleton 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Epps Singleton 2 sample  test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.epps_singleton_2samp(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same distribution", "probably not same distribution", sigLev)

	def testTwoSampleAnderson(self, ds1, ds2, sigLev=.05):
		"""
		Anderson 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Anderson 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		dseq = (data1, data2)
		stat, critValues, sLev = sta.anderson_ksamp(dseq)
		slAlpha = 100 * sigLev

		if slAlpha == 10:
			cv = critValues[1]
		elif slAlpha == 5:
			cv = critValues[2]
		elif slAlpha == 2.5:
			cv = critValues[3]
		elif slAlpha == 1:
			cv = critValues[4]
		else:
			cv = None

		result = self.__printResult("stat", stat, "critValues", critValues, "critValue", cv, "significanceLevel", sLev)
		print("stat:   {:.3f}".format(stat))
		if cv is None:
			msg = "critical values value not found for provided siginificance level"
		else:
			if stat < cv:
				msg = "probably same distribution at the {:.3f} siginificance level".format(sigLev)
			else:
				msg = "probably not same distribution at the {:.3f} siginificance level".format(sigLev)
		print(msg)
		return result


	def testTwoSampleScaleAb(self, ds1, ds2, sigLev=.05):
		"""
		Ansari Bradley 2 sample scale statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Ansari Bradley 2 sample scale test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.ansari(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same scale", "probably not same scale", sigLev)
		return result

	def testTwoSampleScaleMood(self, ds1, ds2, sigLev=.05):
		"""
		Mood 2 sample scale statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Mood 2 sample scale test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.mood(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same scale", "probably not same scale", sigLev)
		return result

	def testTwoSampleVarBartlet(self, ds1, ds2, sigLev=.05):
		"""
		Ansari Bradley 2 sample scale statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Ansari Bradley 2 sample scale test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.bartlett(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleVarLevene(self, ds1, ds2, sigLev=.05):
		"""
		Levene 2 sample variance statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Levene 2 sample variance test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.levene(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleVarFk(self, ds1, ds2, sigLev=.05):
		"""
		Fligner-Killeen 2 sample variance statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Fligner-Killeen 2 sample variance test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue = sta.fligner(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue)
		self.__printStat(stat, pvalue, "probably same variance", "probably not same variance", sigLev)
		return result

	def testTwoSampleMedMood(self, ds1, ds2, sigLev=.05):
		"""
		Mood 2 sample median statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Mood 2 sample median test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat, pvalue, median, ctable = sta.median_test(data1, data2)
		result = self.__printResult("stat", stat, "pvalue", pvalue, "median", median, "contigencyTable", ctable)
		self.__printStat(stat, pvalue, "probably same median", "probably not same median", sigLev)
		return result

	def testTwoSampleZc(self, ds1, ds2, sigLev=.05):
		"""
		Zhang-C 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Zhang-C 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
			
		#find ranks
		pooled = np.concatenate([data1, data2])
		ranks = findRanks(data1, pooled)
		ranks.extend(findRanks(data2, pooled))
		
		s1 = 0.0
		for i in range(1, l1+1):
			s1 += math.log(l1 / (i - 0.5) - 1.0) * math.log(l / (ranks[i-1] - 0.5) - 1.0)
			
		s2 = 0.0
		for i in range(1, l2+1):
			s2 += math.log(l2 / (i - 0.5) - 1.0) * math.log(l / (ranks[l1 + i - 1] - 0.5) - 1.0)
		stat = (s1 + s2) / l
		print(formatFloat(3, stat, "stat:"))
		return stat

	def testTwoSampleZa(self, ds1, ds2, sigLev=.05):
		"""
		Zhang-A 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Zhang-A 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
		pooled = np.concatenate([data1, data2])
		cd1 = CumDistr(data1)
		cd2 = CumDistr(data2)
		sum = 0.0
		for i in range(1, l+1):
			v = pooled[i-1]
			f1 = cd1.getDistr(v)
			f2 = cd2.getDistr(v)
			
			t1 = f1 * math.log(f1)
			t2 = 0 if f1 == 1.0 else (1.0 - f1) * math.log(1.0 - f1)
			sum += l1 * (t1 + t2) / ((i - 0.5) * (l - i + 0.5))
			t1 = f2 * math.log(f2)
			t2 = 0 if f2 == 1.0 else (1.0 - f2) * math.log(1.0 - f2)
			sum += l2 * (t1 + t2) / ((i - 0.5) * (l - i + 0.5))
		stat = -sum
		print(formatFloat(3, stat, "stat:"))
		return stat

	def testTwoSampleZk(self, ds1, ds2, sigLev=.05):
		"""
		Zhang-K 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing Zhang-K 2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		l1 = len(data1)
		l2 = len(data2)
		l = l1 + l2
		pooled = np.concatenate([data1, data2])
		cd1 = CumDistr(data1)
		cd2 = CumDistr(data2)
		cd = CumDistr(pooled)
		
		maxStat = None
		for i in range(1, l+1):
			v = pooled[i-1]
			f1 = cd1.getDistr(v)
			f2 = cd2.getDistr(v)
			f = cd.getDistr(v)
			
			t1 = 0 if f1 == 0 else f1 * math.log(f1 / f)
			t2 = 0 if f1 == 1.0 else (1.0 - f1) * math.log((1.0 - f1) / (1.0 - f))
			stat = l1 * (t1 + t2)
			t1 = 0 if f2 == 0 else f2 * math.log(f2 / f)
			t2 = 0 if f2 == 1.0 else (1.0 - f2) * math.log((1.0 - f2) / (1.0 - f))
			stat += l2 * (t1 + t2)
			if maxStat is None or stat > maxStat:
				maxStat = stat
		print(formatFloat(3, maxStat, "stat:"))
		return maxStat


	def testTwoSampleCvm(self, ds1, ds2, sigLev=.05):
		"""
		2 sample cramer von mises
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
			sigLev: statistical significance level
		"""
		self.__printBanner("doing 2 sample CVM test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		data = np.concatenate((data1,data2))
		rdata = sta.rankdata(data)
		n = len(data1)
		m = len(data2)
		l = n + m

		s1 = 0
		for i in range(n):
			t = rdata[i] - (i+1)	
			s1 += (t * t)
		s1 *= n

		s2 = 0
		for i in range(m):
			t = rdata[i + n] - (i+1)	
			s2 += (t * t)
		s2 *= m

		u = s1 + s2
		stat = u / (n * m * l) - (4 * m * n - 1) / (6 * l)
		result = self.__printResult("stat", stat)
		return result

	def ensureSameSize(self, dlist):
		"""
		ensures all data sets are of same size
		
		Parameters
			dlist : data source list
		"""
		le = None
		for d in dlist:
			cle = len(d)
			if le is None:
				le = cle
			else:
				assert cle == le, "all data sets need to be of same size"


	def testTwoSampleWasserstein(self, ds1, ds2):
		"""
		Wasserstein 2 sample statistic	
		
		Parameters
			ds1: data set name or list or numpy array
			ds2: data set name or list or numpy array
		"""
		self.__printBanner("doing Wasserstein distance2 sample test", ds1, ds2)
		data1 = self.getNumericData(ds1)
		data2 = self.getNumericData(ds2)
		stat = sta.wasserstein_distance(data1, data2)
		sd = np.std(np.concatenate([data1, data2]))
		nstat = stat / sd
		result = self.__printResult("stat", stat, "normalizedStat", nstat)
		return result
	
	def getMaxRelMinRedFeatures(self, fdst, tdst, nfeatures, nbins=20):
		"""
		get top n features based on max relevance and min redudancy	algorithm
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nfeatures : desired no of features
			nbins : no of bins for numerical data
		"""	
		self.__printBanner("doing max relevance min redundancy feature selection")
		return self.getMutInfoFeatures(fdst, tdst, nfeatures, "mrmr", nbins)	

	def getJointMutInfoFeatures(self, fdst, tdst, nfeatures, nbins=20):
		"""
		get top n features based on joint mutual infoormation	algorithm
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nfeatures : desired no of features
			nbins : no of bins for numerical data
		"""	
		self.__printBanner("doingjoint mutual info feature selection")
		return self.getMutInfoFeatures(fdst, tdst, nfeatures, "jmi", nbins)
		
	def getCondMutInfoMaxFeatures(self, fdst, tdst, nfeatures, nbins=20):
		"""
		get top n features based on condition mutual information maximization algorithm
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nfeatures : desired no of features
			nbins : no of bins for numerical data
		"""	
		self.__printBanner("doing conditional mutual info max feature selection")
		return self.getMutInfoFeatures(fdst, tdst, nfeatures, "cmim", nbins)

	def getInteractCapFeatures(self, fdst, tdst, nfeatures, nbins=20):
		"""
		get top n features based on interaction capping algorithm
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nfeatures : desired no of features
			nbins : no of bins for numerical data
		"""	
		self.__printBanner("doing interaction capped feature selection")
		return self.getMutInfoFeatures(fdst, tdst, nfeatures, "icap", nbins)

	def getMutInfoFeatures(self, fdst, tdst, nfeatures, algo, nbins=20):
		"""
		get top n features based on various mutual information	based algorithm
		ref: Conditional likelihood maximisation : A unifying framework for information 
		theoretic feature selection, Gavin Brown
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nfeatures : desired no of features
			algo: mi based feature selection algorithm
			nbins : no of bins for numerical data
		"""	
		#verify data source types types
		le = len(fdst)
		nfeatGiven = int(le / 2)
		assertGreater(nfeatGiven, nfeatures, "no of features should be greater than no of features to be selected")
		fds = list()
		types = ["num", "cat"]
		for i in range (0, le, 2):
			ds = fdst[i]
			dt = fdst[i+1]
			assertInList(dt, types, "invalid type for data source " + dt)
			data = self.getNumericData(ds) if dt == "num" else self.getCatData(ds)
			p =(ds, dt)
			fds.append(p)
		algos = ["mrmr", "jmi", "cmim", "icap"]
		assertInList(algo, algos, "invalid feature selection algo " + algo)
		
		assertInList(tdst[1], types, "invalid type for data source " + tdst[1])
		data = self.getNumericData(tdst[0]) if tdst[1] == "num" else self.getCatData(tdst[0])
		#print(fds)
		
		sfds = list()
		selected = set()
		relevancies = dict()
		for i in range(nfeatures):
			#print(i)
			scorem = None
			dsm = None
			dsmt = None
			for ds, dt in fds:
				#print(ds, dt)
				if ds not in selected:
					#relevancy
					if ds in relevancies:
						mutInfo = relevancies[ds]
					else:
						mutInfo = self.getMutualInfo([ds, dt,  tdst[0], tdst[1]], nbins)["mutInfo"]
						relevancies[ds] = mutInfo
					relev = mutInfo
					#print("relev", relev)
					
					#redundancy
					smi = 0
					reds = list()
					for sds, sdt, _ in sfds:
						#print(sds, sdt)
						mutInfo = self.getMutualInfo([ds, dt,  sds, sdt], nbins)["mutInfo"]
						mutInfoCnd = self.getCondMutualInfo([ds, dt,  sds, sdt, tdst[0], tdst[1]], nbins)["condMutInfo"] \
						if algo != "mrmr" else 0
						
						red = mutInfo - mutInfoCnd
						reds.append(red)	
						
					if algo == "mrmr" or algo == "jmi":
						redun = sum(reds) / len(sfds) if len(sfds) > 0 else 0
					elif algo == "cmim" or algo == "icap":
						redun = max(reds) if len(sfds) > 0 else 0
						if algo == "icap":
							redun = max(0, redun)
					#print("redun", redun)
					score = relev - redun
					if scorem is None or score > scorem:
						scorem = score
						dsm = ds
						dsmt = dt
						
			pa = (dsm, dsmt, scorem)
			#print(pa)
			sfds.append(pa)
			selected.add(dsm)
			
		selFeatures = list(map(lambda r : (r[0], r[2]), sfds))
		result = self.__printResult("selFeatures", selFeatures)
		return result


	def getFastCorrFeatures(self, fdst, tdst, delta, nbins=20):
		"""
		get top features based on Fast Correlation Based Filter (FCBF)
		ref: Feature Selection for High-Dimensional Data: A Fast Correlation-Based Filter Solution
		Lei Yu
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			delta : feature, target correlation threshold
			nbins : no of bins for numerical data
		"""	
		le = len(fdst)
		nfeatGiven = int(le / 2)
		fds = list()
		types = ["num", "cat"]
		for i in range (0, le, 2):
			ds = fdst[i]
			dt = fdst[i+1]
			assertInList(dt, types, "invalid type for data source " + dt)
			data = self.getNumericData(ds) if dt == "num" else self.getCatData(ds)
			p =(ds, dt)
			fds.append(p)
		
		assertInList(tdst[1], types, "invalid type for data source " + tdst[1])
		data = self.getNumericData(tdst[0]) if tdst[1] == "num" else self.getCatData(tdst[0])
		
		# get features with symetric uncertainty above threshold
		tentr = self.getAnyEntropy(tdst[0], tdst[1], nbins)["entropy"]
		rfeatures = list()
		fentrs = dict()
		for ds, dt in fds:
			mutInfo = self.getMutualInfo([ds, dt,  tdst[0], tdst[1]], nbins)["mutInfo"]
			fentr = self.getAnyEntropy(ds, dt, nbins)["entropy"]
			sunc = 2 * mutInfo / (tentr + fentr)
			#print("ds {}  sunc {:.3f}".format(ds, sunc))
			if sunc >= delta:
				f = [ds, dt, sunc, False]
				rfeatures.append(f)
				fentrs[ds] = fentr
		
		# sort descending of sym uncertainty
		rfeatures.sort(key=lambda e : e[2], reverse=True)
		
		#disccard redundant features
		le = len(rfeatures)
		for i in range(le):
			if rfeatures[i][3]:
				continue
			for j in range(i+1, le, 1):
				if rfeatures[j][3]:
					continue
				mutInfo = self.getMutualInfo([rfeatures[i][0], rfeatures[i][1],  rfeatures[j][0], rfeatures[j][1]], nbins)["mutInfo"]
				sunc  = 2 * mutInfo / (fentrs[rfeatures[i][0]] + fentrs[rfeatures[j][0]])
				if sunc >= rfeatures[j][2]:
					rfeatures[j][3] = True
			
		frfeatures = list(filter(lambda f : not f[3], rfeatures))
		selFeatures = list(map(lambda f : [f[0], f[2]], frfeatures))		
		result = self.__printResult("selFeatures", selFeatures)
		return result
			
	def getInfoGainFeatures(self, fdst, tdst, nfeatures, nsplit, nbins=20):
		"""
		get top n features based on information gain or entropy loss
		
		Parameters
			fdst: list of pair of data set name or list or numpy array and data type
			tdst: target data set name or list or numpy array and data type (cat for classification num for regression)
			nsplit : num of splits
			nfeatures : desired no of features
			nbins : no of bins for numerical data
		"""	
		le = len(fdst)
		nfeatGiven = int(le / 2)
		assertGreater(nfeatGiven, nfeatures, "available features should be greater than desired")
		fds = list()
		types = ["num", "cat"]
		for i in range (0, le, 2):
			ds = fdst[i]
			dt = fdst[i+1]
			assertInList(dt, types, "invalid type for data source " + dt)
			data = self.getNumericData(ds) if dt == "num" else self.getCatData(ds)
			p =(ds, dt)
			fds.append(p)
		
		assertInList(tdst[1], types, "invalid type for data source " + tdst[1])
		assertGreater(nsplit, 3, "minimum 4 splits necessary")
		tdata = self.getNumericData(tdst[0]) if tdst[1] == "num" else self.getCatData(tdst[0])
		tentr = self.getAnyEntropy(tdst[0], tdst[1], nbins)["entropy"]
		sz =len(tdata)
		
		sfds = list()
		for ds, dt in fds:
			#print(ds, dt)
			if dt == "num":
				fd = self.getNumericData(ds)
				_ , _ , vmax, vmin = self.__getBasicStats(fd)
				intv = (vmax - vmin) / nsplit
				maxig = None
				spmin = vmin + intv
				spmax = vmax - 0.9 * intv
				
				#iterate all splits
				for sp in np.arange(spmin, spmax, intv):
					ltvals = list()
					gevals = list()
					for i in range(len(fd)):
						if fd[i] < sp:
							ltvals.append(tdata[i])
						else:
							gevals.append(tdata[i])
					
					self.addListNumericData(ltvals, "spds") if tdst[1] == "num" else self.addListCatData(ltvals, "spds")
					lten = self.getAnyEntropy("spds", tdst[1], nbins)["entropy"]
					self.addListNumericData(gevals, "spds") if tdst[1] == "num" else self.addListCatData(gevals, "spds")
					geen = self.getAnyEntropy("spds", tdst[1], nbins)["entropy"]
					
					#info gain
					ig = tentr - (len(ltvals) * lten / sz + len(gevals) * geen / sz)
					if maxig is None or ig > maxig:
						maxig = ig
				
				pa = (ds, maxig)
				sfds.append(pa)
			else:
				fd = self.getCatData(ds)
				fds = set(fd)
				fdps = genPowerSet(fds)
				maxig = None
				
				#iterate all subsets
				for s in fdps:
					if len(s) == len(fds):
						continue
					invals = list()
					exvals = list()
					for i in range(len(fd)):
						if fd[i] in s:
							invals.append(tdata[i])
						else:
							exvals.append(tdata[i])
							
					self.addListNumericData(invals, "spds") if tdst[1] == "num" else self.addListCatData(invals, "spds")
					inen = self.getAnyEntropy("spds", tdst[1], nbins)["entropy"]
					self.addListNumericData(exvals, "spds") if tdst[1] == "num" else self.addListCatData(exvals, "spds")
					exen = self.getAnyEntropy("spds", tdst[1], nbins)["entropy"]

					ig = tentr - (len(invals) * inen / sz + len(exvals) * exen / sz)
					if maxig is None or ig > maxig:
						maxig = ig
					
				pa = (ds, maxig)
				sfds.append(pa)
			
		#sort of info gain
		sfds.sort(key = lambda v : v[1], reverse = True)
		
		result = self.__printResult("selFeatures", sfds[:nfeatures])
		return result
				
	def __stackData(self, *dsl):
		"""
		stacks collumd to create matrix
		
		Parameters
			dsl: data source list
		"""
		dlist = tuple(map(lambda ds : self.getNumericData(ds), dsl))
		self.ensureSameSize(dlist)
		dmat = np.column_stack(dlist)
		return dmat	

	def __printBanner(self, msg, *dsl):
		"""
		print banner for any function
		
		Parameters
			msg: message
			dsl: list of data set name or list or numpy array
		"""
		tags = list(map(lambda ds : ds if type(ds) == str else "annoynymous", dsl))
		forData = " for data sets " if tags else ""
		msg = msg + forData + " ".join(tags) 
		if self.verbose:
			print("\n== " + msg + " ==")


	def __printDone(self):
		"""
		print banner for any function
		"""
		if self.verbose:
			print("done")

	def __printStat(self, stat, pvalue, nhMsg, ahMsg, sigLev=.05):
		"""
		generic stat and pvalue output
		
		Parameters
			stat : stat value
			pvalue : p value
			nhMsg : null hypothesis violation message
			ahMsg : null hypothesis  message
			sigLev : significance level
		"""
		if self.verbose:
			print("\ntest result:")
			print("stat:   {:.3f}".format(stat))
			print("pvalue: {:.3f}".format(pvalue))
			print("significance level: {:.3f}".format(sigLev))
			print(nhMsg if pvalue > sigLev else ahMsg)

	def __printResult(self,  *values):
		"""
		print results
		
		Parameters
			values : flattened kay and value pairs
		"""
		result = dict()
		assert len(values) % 2 == 0, "key value list should have even number of items"
		for i in range(0, len(values), 2):
			result[values[i]] = values[i+1]
		if self.verbose:
			print("result details:")
			self.pp.pprint(result)
		return result
		
	def __getBasicStats(self, data):
		"""
		get mean and std dev
		
		Parameters
			data : numpy array
		"""
		mean = np.average(data)
		sd = np.std(data)
		r = (mean, sd, np.max(data), np.min(data))
		return r


