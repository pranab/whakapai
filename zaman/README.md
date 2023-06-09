# zaman

Everyting time series, EDA, forecasting, classificatio and anomaly detection
* proph : forecasting with fbprophet
* nuproph : forecasting with  neuralprophet  (coming)
* tsutil : varioous utilities for time series, EDA etc
* tsgen : time series generator



1. Install:

Package install
pip3 install -i https://test.pypi.org/simple zaman==0.0.2

Local build and install for the latest
Clone GirHub repo and run this at the repo root directory
./lbi.sh zaman

For dependencies run this form the project root directory
pip3 install -r requirements.txt

2. Project page in testpypi

https://test.pypi.org/project/zaman/0.0.2/


3. Blogs posts



4. Code usage example

Here is some example code for fbprophet based forecasting. All you need to do is to create the model 
object  and then call train() and forecast() for model training and forecasting. Please refer 
to the examples directory for full working example code

	from matumizi.util import *
	from matumizi.mlutil import *
	from matumizi.sampler import *
	from zaman.proph import *

	forecaster = ProphetForcaster("myconfig.properties", None, None)	
	forecaster.train()
	forecaster.forecast()	
	
