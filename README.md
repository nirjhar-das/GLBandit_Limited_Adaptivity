Official Code base for the paper [_Optimal Regret with Limited Adaptivity for Generalized Linear Contextual Bandits_](https://arxiv.org/abs/) by [Ayush Sawarni](https://sawarniayush.github.io/), [Nirjhar Das](https://nirjhar-das.github.io/), [Siddharth Barman](https://www.csa.iisc.ac.in/~barman/) and [Gaurav Sinha](https://sinhagaurav.github.io/).

## Installation
The code base has dependency on basic packages listed in [requirements.txt](./requirements.txt). It can be installed via the following command:
```
$ pip install -r requirements.txt 
```

## Usage
This code base implements `RS-GLinUCB` (Algorithm 2 in the aforementioned paper). Other baseline algorithms include `ECOLog`, `GLM-UCB`, `GLOC` and `OFULogPlus`, whose codes are taken from [logistic_bandit](https://github.com/criteo-research/logistic_bandit/tree/master) code base of [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a/faury22a.pdf) and reimplemented with minor modifications. The references of the baseline algorithms are as follows:

- `ECOLog` from [Faury et al. 2022](https://proceedings.mlr.press/v151/faury22a/faury22a.pdf)
- `GLM-UCB` from [Filippi et al. 2010](https://papers.nips.cc/paper/2010/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf),
- `OL2M` from [Zhang et al. 2016](http://proceedings.mlr.press/v48/zhangb16.pdf),
- `GLOC` from [Jun et al. 2017](https://proceedings.neurips.cc/paper/2017/file/28dd2c7955ce926456240b2ff0100bde-Paper.pdf),
- `OFULogPlus` from [Lee et al. 2023](https://arxiv.org/abs/2310.18554)

The Jupyter notebook [simulations.ipynb](./simulations.ipynb) implements the regret simulation experiements form the paper for the logistic and the probit reward models. It also implements the simulation for actual time taken by `RS-GLinUCB` and `ECOLog`. The result of these experiments can be seen below.

![all_results](./Results/Combined_Result.png)