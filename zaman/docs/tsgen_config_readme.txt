This document provides explanation of different configuration parameters for time series generator. A value of
"_" implies default values. Default values can be found in the constructor in tsgen.py

window.size=200_d
  	Size of time series. In this this 200_d means 200 days in past until current time. use "h", "m" and "s"
  	for hour, minute and second respectively
  
  
window.samp.interval.type=_
	Sampling interval which is enither "fixed"(default value)  or "random".
	
window.samp.interval.params=1_d
	Sampling interval. In this case 1_d means 1 day. Follows same convention as window.size parameter. if 
	window.samp.interval.type=random, therere will be 2 coma separated parameters mean and std dev of a normal
	distribution
	
window.samp.align.unit=d
	Alignment of window start. in this it's "d" which is day
	
window.time.unit=s
	Time unit for window

output.value.type=_
	Data type for time series values. it's "float"(default) or "int"
	
output.value.precision=2
	Precision for floating point output
	
output.value.format=_
	Output layout. if "long"(default) it's one sample per line of output. if "short" the whole time series is
	in one line of output. In this case output.value.nsamples needs to be set
	
output.value.nsamples=_
	Number of time serieses to generate. Only relevant when output.value.format=short. This will be preferred format 
	when many time series are generated as training data for a sequence machine learning model.
	
output.time.format=formatted
	Format for timestamp in the output. it's either "epoch"(default) or "formatted"
	
ts.random.params=0,100.0
	Random noise parameters which is mean and std dev for noise in trend seasonal time series
	
ts.base=_
	The base for trend seasonal time series. it's either "mean" or "ar" (auto regressive)
	
ts.base.params=10000.0
	The parameters for trend seasonal time series base. it's a float value for ts.base=average or coma separated list
	of auto regression coefficients when ts.base=ar
	
ts.trend=linear
	Trend with options "linear", "quadratic" and "logistic"

ts.trend.params=0.2
	Trend paraemeters. Rate of change for "linear", coma separated list for "quadratic" and expoonential constant 
	for "logistic"
	
ts.cycles=week
	Seasonality type with options "year", "week" and "day"

ts.cycle.year.params=_
	Yearly seasonality 12 values

ts.cycle.week.params=100,0,-50,100,700,800,900
	Weekly  seasonality 7 values
	
ts.cycle.day.params=_
	Daily   seasonality 24 values
	
rw.init.value=_
	TODO

rw.range=_
	TODO
	
ar.params=_
	TODO
	
ar.seed=_
	TODO

ar.exp.param=_
	TODO
	
corr.file.path=_
	TODO

corr.file.col=_
	TODO

corr.scale=_
	TODO

corr.noise.stddev=_
	TODO

corr.lag=_
	TODO

ccorr.file.path=_
	TODO

ccorr.file.col=_
	TODO

ccorr.co.params=_
	TODO

ccorr.unco.params=_
	TODO

si.params=_
	TODO

motif.params=_
	TODO

anomaly.params=meanshift,100,120,25.0,3.0
	Anaomaly sequence parameters. The first 3 are same for anomaly types. the 1st param is anomaly type. The 2nd 
	and tthe 3rd parameters are begin and end positions of anomaly sequence. The rate of shift. The last is std dev of normal
	distr noise. For zero mean random anomaly example parameters are  "random,100,120,3.0" The last parameter is the 
	normal distribution std deviaation. For motif based an example is "motif,100,120,3.0,motif.csv,2". The 4th parameter 
	is normal distribution std deviaation for noise. The 5th paramter is file path for motif definition. The 6th parameter
	is th index of column in motif file. For multiple sinusoidal anomaly an example is "multsine,100,120,3.0,20.0,100.0,30.0,80.0,..."
	The 4th parameter is normal distribution std deviaation for noise. The 5th parameter onwards is the amplitute and period  the 
	sine functions with as many pairs as the number of sine functions
	
	
