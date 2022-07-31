# qinisa

Reinforcement Learning(RL) and Multi Arm Bandit(MAB)
* bandit : base class for MAB
* mab : various MAB implementation classes
* rlba : RL base class
* reinfl : TD learning , Q learning



1. Install:

Run
pip3 install -i https://test.pypi.org/simple qinisa==0.0.2

For installing latest, clone repo and run this at the project root directory
pip3 install .


2. Project page in testpypi

https://test.pypi.org/project/qinisa/0.0.2/


3. Blogs posts

* [Workforce scheduling with ucb bandit algorithms](https://pkghosh.wordpress.com/2022/03/28/gig-economy-workforce-scheduling-with-reinforcement-learning/)
* [Pricing Policy Evaluation and Comparison with Temporal Difference Learning](https://pkghosh.wordpress.com/2022/07/31/pricing-policy-evaluation-and-comparison-with-temporal-difference-learning/)


4. Code usage example

Here is some example code that uses upper confidence bound MAB. All you need to do is to create the model object 
and then call act() to get the next action and then reward the action by calling setReward(act, reward). Please refer 
to the examples directory for full working example code

	from matumizi.util import *
	from matumizi.mlutil import *
	from matumizi.sampler import *
	from qinisa.mab import *

	emtempl = ["d1", "d2", "d3"]	
	model = UpperConfBound(emtempl, 20, True, "./log/rg.log", "info")
	act = model.getAction()	
	reward = 0.9
	model.setReward(act, reward)
	
