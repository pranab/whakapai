# matumizi

Data Science utilities including following modules
* util : misc utility functions
* mlutil : machine learning related unitilies including a type aware confifiguration class
* stats : various stats classes and functions
* sampler : sampling from various statu=istical distributions
* daexp : many data exploration functions consoloidating numpy, scipy, statsmodel and scikit
* mcsim : monte carlo simulation
* sdrift : supervised machine learning model concept drift detection

## Instructions

1. Install:

Run
pip3 install -i https://test.pypi.org/simple/ matumizi==0.0.7

For installing latest, clone rep and run this at the project root directory
pip3 install .


2. Project page in testpypi

https://test.pypi.org/project/matumizi/0.0.7/


3. Blogs posts

* [Data exploration module overview including usage examples](https://pkghosh.wordpress.com/2020/07/13/learn-about-your-data-with-about-seventy-data-exploration-functions-all-in-one-python-class/) 
* [Monte Carlo simulation for project cost estimation](https://pkghosh.wordpress.com/2020/05/11/monte-carlo-simulation-library-in-python-with-project-cost-estimation-as-an-example/)
* [Information theory based feature selection](https://pkghosh.wordpress.com/2022/05/29/feature-selection-with-information-theory-based-techniques-in-python/)
* [Stock Portfolio Balancing with Monte Carlo Simulation](https://pkghosh.wordpress.com/2022/08/23/stock-portfolio-balancing-with-monte-carlo-simulation/)
* [Synthetic Regression Data Generation in Python](https://pkghosh.wordpress.com/2023/01/22/synthetic-regression-data-generation-in-python/)

4. Code usage example

Here is some example code that uses all 5 modules. You can find lots of examples in 
[another repo](https://github.com/pranab/avenir/tree/master/python/app) of mine. There the 
imports are direct and not through the package matmizi. The example directory also has example code


	import sys
	import math
	from matumizi import util as ut
	from matumizi import mlutil as ml
	from matumizi import sampler as sa
	from matumizi import stats as st
	from matumizi import daexp as de

	#generate some random strings
	ldata = ut.genIdList(10, 6)
	print("random strings")
	print(ldata)
	
	#select random sublist from a list
	sldata = ut.selectRandomSubListFromList(ldata, 4)
	print("nselected random strings)")
	print(sldata)
	
	random walk
	print("\nrandom walk")
	for pos in ml.randomWalk(20, 10, -2, 2):
		print(pos)
		
	#sample from non parametric sampler
	print("\nsampling from a non parametric sampler")
	sampler = sa.NonParamRejectSampler(10, 4, 1, 4, 8, 16, 14, 12, 8, 4, 2)
	for _ in range(8):
	d = sampler.sample()
		print(ut.formatFloat(3, d))
		
	#statistics from asliding window
	print("\nstats from sliding window")
	wsize = 30
	win = st.SlidingWindowStat.createEmpty(wsize)
	mean = 10
	sd = 2
	ns = sa.NormalSampler(mean, sd)
	for _ in range(40):
		#gaussian with some noise
		d = ns.sample() + sa.randomFloat(-1, 1)
		win.add(d)
	re = win.getStat()	
	print(re)
	
	#get time series components
	print("\ntime series components")
	expl = de.DataExplorer(False)
	mean = 100
	sd = 5
	period = 7
	trdelta = .1
	cycle = list(map(lambda v : 10 * math.sin(2 * math.pi * v / period), range(period)))
	sampler = sa.NormalSamplerWithTrendCycle(mean, sd, trdelta, cycle)
	ldata = list(map(lambda i : sampler.sample(), range(200)))
	expl.addListNumericData(ldata, "test")
	re = expl.getTimeSeriesComponents("test", "additive", period, True)
	print(re)
