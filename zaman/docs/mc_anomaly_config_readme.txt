Explains config parameters for markov chain based anomaly detection

train.data.file=ecgt.csv
	Training data file path
	
train.data.field=1
	Index of column with time series values
	
train.discrete.size=.04
	Discretization step to use to transform data
	
train.val.margin=.10
	Extra margin in state transition probability matrix to leave room for extreme data valus 
	not seen in training data
	
train.save.model=True
	Should set to True to save model
	
train.model.file=./model/tsano_mc_esg.mod
	Model file path where model is saved
	
pred.data.file=ecgp.csv
	Prediction data file path
	
pred.data.field=1
	Index of column with time series values
pred.ts.field=0
	Index of column with time stamp

pred.window.size=4
	Window size for sub sequence
	
pred.ano.threshold=.000005
	Joint probability threshold. Anomalous sequence if below thos value
	
pred.output.file=ecga.csv
	Output file for anomaly score
	
pred.output.prec=_
	Folting pt precision for output