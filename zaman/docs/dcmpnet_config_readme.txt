common.verbose=True
	Set to True for verbose output
	
common.model.directory=./model/tnn
	Model directory path
	
common.model.file.trend=ret_tr_trend.mod
	Trend model file name
	
common.model.file.remain=ret_tr_remain.mod
	Remainder model file name
	
common.trend.config.file=tnn_ret_trend.properties
	Trend model FFN config file path
	
common.remain.config.file=tnn_ret_remain.properties
	Remainder model FFN config file path
	
train.data.file=ret_tr.csv
	Time series (TS) data file path
	
train.data.file.trend=ret_tr_trend.csv
	Trend training data file path
	
train.data.file.remain=ret_tr_remain.csv
	Remainder training data file path

train.data.lookback.size=_
	Look back window size
	
train.data.forecast.size=_
	Forecasdt window size
	
train.data.ts.col=_
	Time stamp column index in TS data
	
train.data.value.col=_
	Value stamp column index in TS data

train.data.split=_
	Training and validation data split ratio
	
train.data.regr.type=_
	Type of regresssion for trend extraction. Options: linear, square, cubic
	
valid.data.file.trend=ret_va_trend.csv
	Trend validation data file path
	
valid.data.file.remain=ret_va_remain.csv
	Remainder validation data file path

output.data.precision=2
	Output precision
