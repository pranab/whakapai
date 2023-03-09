# zaman

Everyting time series, EDA, forecasting, classificatio and anomaly detection
* proph : forecasting with fbprophet
* nuproph : forecasting with  neuralprophet  (coming)
* tsutil : varioous utilities for time series, EDA etc



1. Install:

Run
pip3 install -i https://test.pypi.org/simple zaman==0.0.1

For installing latest, clone repo and run this at .../whakapai directory 
./lbi.sh zaman


2. Project page in testpypi

https://test.pypi.org/project/zaman/0.0.1/


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
	
