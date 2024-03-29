# zaman

Everyting time series, EDA, forecasting, classificatio and anomaly detection
* proph : forecasting with fbprophet
* nuproph : forecasting with  neuralprophet  (coming)
* tseda : varioous utilities and data exploration for time series
* tsgen : time series generator
* tsano : time series anomaly detection
* tsfeat : feature extraction from time series



1. Install:

Package install
pip3 install -i https://test.pypi.org/simple zaman==0.0.9

Local build and install for the latest
Clone GirHub repo and run this at the repo root directory
./lbi.sh zaman

For dependencies run this form the project root directory
pip3 install -r requirements.txt

2. Project page in testpypi

https://test.pypi.org/project/zaman/0.0.7/


3. Blogs posts
* [Synthetic Time Series Data Generation](https://pkghosh.wordpress.com/2023/03/29/synthetic-time-series-data-generation/)
* [Time Series Sequence Anomaly Detection with Markov Chain](https://pkghosh.wordpress.com/2023/06/28/time-series-sequence-anomaly-detection-with-markov-chain/)
* [Time Series Data Exploration with Wavelet Transform](https://pkghosh.wordpress.com/2023/09/29/time-series-data-exploration-with-wavelet-transform/)
* [Time Series Forecasting with Decomposition and Two Linear Networks](https://pkghosh.wordpress.com/2023/12/28/time-series-forecasting-with-decomposition-and-two-linear-networks/)
* [Time Series Classification with Neural Network using Random Sub Sequence Statistics as Features](https://pkghosh.wordpress.com/2024/01/26/time-series-classification-with-neural-network-using-random-sub-sequence-statistics-as-features/)

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
	
