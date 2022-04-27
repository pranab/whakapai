# torvik

No code framework for various PyTorch Networks with these modules
* tnn :feed forward network aka multi player perceptron
* lstm : LSTM network
* ae : auto encoder network
* gcn : graph convolution network

## Instructions

1. Install:

Run
pip3 install -i https://test.pypi.org/simple/torvik

For installing latest, clone rep and run this at the project root directory
pip3 install .


2. Project page in testpypi

https://test.pypi.org/project/torvik/0.0.1/

3. Blogs posts

[Duplicate data detection with feed forward networ](https://pkghosh.wordpress.com/2021/07/21/duplicate-data-detection-with-neural-network-and-contrastive-learning/)
[Viral infection prediction witkh LSTM](https://pkghosh.wordpress.com/2020/08/18/predicting-individual-viral-infection-using-contact-data-with-lstm-neural-network/)
[Service ticket anomaly detection with auto encoder](https://pkghosh.wordpress.com/2021/01/20/customer-service-quality-monitoring-with-autoencoder-based-anomalous-case-detection/)

4. Code usage example

Here is some example code that uses various modules. 

	import sys
	import math
	from torvik.tnn import *
	from torvik.lstm import *
	from torvik.gcn import *
	
	#train feed forward network for classification
	loModel = FeedForwardNetwork("tnn_lo.properties_")
	loModel.buildModel()
	FeedForwardNetwork.batchTrain(loModel)	
	
	#train LSTM and then predict
	classfi = LstmNetwork("lstm_ct.properties_")
	classfi.buildModel()
	classfi.trainLstm()
	classfi.predictLstm()
	
	#train and validate GCN
	model = GraphConvoNetwork(args.cfile)
	model.buildModel()
	GraphConvoNetwork.trainModel(model)
	GraphConvoNetwork.validateModel(model)
	
	
