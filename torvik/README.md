# torvik

No code framework for various PyTorch Networks with these modules
* tnn : Feed Forward Network aka multi player perceptron
* lstm : LSTM network
* ae : Auto Encoder network
* gcn : Graph Convolution Network
* fftn : Feed Forward Network for cointrastive learning
* vae : Variational Auto Encoder
* nntuner : optuna based no code auto tuner

## Instructions

1. Install:

Package install
pip3 install -i https://test.pypi.org/simple torvik==0.0.6

Local build and install for the latest
Clone GirHub repo and run this at the repo root directory
./lbi.sh torvik

For dependencies run this form the project root directory
pip3 install -r requirements.txt

2. Project page in testpypi

https://test.pypi.org/project/torvik/0.0.1/

3. Blogs posts

* [Duplicate data detection with feed forward networ](https://pkghosh.wordpress.com/2021/07/21/duplicate-data-detection-with-neural-network-and-contrastive-learning/)
* [Viral infection prediction witkh LSTM](https://pkghosh.wordpress.com/2020/08/18/predicting-individual-viral-infection-using-contact-data-with-lstm-neural-network/)
* [Service ticket anomaly detection with auto encoder](https://pkghosh.wordpress.com/2021/01/20/customer-service-quality-monitoring-with-autoencoder-based-anomalous-case-detection/)
* [Subject matter expert dicovery with email text using GCN](https://pkghosh.wordpress.com/2022/04/28/discovering-subject-matter-experts-from-email-communication-data-using-graph-convolution-network/)
* [Tabular data column semantic type identification with contrastive deep learning](https://pkghosh.wordpress.com/2022/06/30/tabular-data-column-semantic-type-identification-with-contrastive-deep-learning/)
* [Simulating A/B Test with Counterfactual and Machine Learning Regression Model](https://pkghosh.wordpress.com/2023/02/28/simulating-a-b-test-with-counterfactual-and-machine-learning-regression-model/)

4. Easy steps to use

* Create the config file for your model
* Create the model object passing the config file path as an argument
* Call the methods of the model object to train, validate and predict

5. Configuration

Configuration covers all areas of model training, validation and prediction as follows. There are example 
configuration files in the config directory
* Meta data for input, file location, column specification etc
* Model checkpointing related, file path etc
* All parameters for the undertying PyTorch model
* Network architecture, icluding layers, no of units, activation type, batch norm and dropout
* Training error and validatino error tracking 

6. Code usage example

Here is some example code that uses various modules. All you need to do is to create a configiration
in properties file format for each model you want to train. Some example config files are in the
config directory. The examples directory has code for few use cases, including code to generate 
synthetic data

	import sys
	import math
	from torvik.tnn import *
	from torvik.lstm import *
	from torvik.gcn import *
	from torvik.ae import *
	
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
	model = GraphConvoNetwork("gcn_em.properties_")
	model.buildModel()
	GraphConvoNetwork.trainModel(model)
	GraphConvoNetwork.validateModel(model)
	
	#train and regenerate with auto encoder
	auenc = AutoEncoder("ae_ticket.properties")
	auenc.buildModel()
	auenc.trainModel()
	recs = auenc.regen()
	
	
