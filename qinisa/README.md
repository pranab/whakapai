# qinisa

Reinforcement Learning and Multi Arm Bandit
* bandit : base class for MAB
* mab : various MAB implementation classes



1. Install:

Run
pip3 install -i https://test.pypi.org/simple qinisa==0.0.1

For installing latest, clone repo and run this at the project root directory
pip3 install .


2. Project page in testpypi

https://test.pypi.org/project/qinisa/0.0.1/

3. Blogs posts

* [Workforce scheduling with ucb bandit algorithms](https://pkghosh.wordpress.com/2022/03/28/gig-economy-workforce-scheduling-with-reinforcement-learning/)


4. Code usage example

Here is some example code that uses upper confidence bound MAB. All you need to do is to create the model object 
and then call act() to get the next action and then reward the action by calling setReward(act, reward)

	from matumizi.util import *
	from matumizi.mlutil import *
	from matumizi.sampler import *
	from qinisa.mab import *

	emtempl = ["d1", "d2", "d3"]	
	model = UpperConfBound(emtempl, 20, True, "./log/rg.log", "info")
	act = model.act()	
	reward = 0.9
	model.setReward(act, reward)
	