common.verbose=True
	Verbosity of output
	
common.feat.type=hist
	Type of feature engieering
	
train.data.file=ma_fa_norm.csv
	Normal data file path
	
train.data.field=1
	Column index of the value field
	
train.hist.padding=0.25
	Padding beyonf min and max of data because anomalous data range may be higher
	
train.hist.nbins=10
	Number of histogram bins
	
train.hist.type=_
	Histogram type e.g uniform
	
pred.data.file=ma_fa_anom.csv
	Anomalous data file path
	
pred.data.field=1
	Column index of the value field in anomalous data
	
pred.ts.field=0
	Column index of time stamp
	
pred.window.size=40
	Window size
	
pred.ano.threshold=0.98
	Anomaly threshold
	
pred.dist.metric=_
	Distance metric e.g l1 and l2
	
pred.output.file=ma_fa_anom_out.csv
	Anomlay detection output file path
	
pred.output.prec=6
	Output precision
