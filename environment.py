import numpy as np
import json
from tqdm import tqdm
import os
from copy import deepcopy
from glmbanditexp.utils.utils import dsigmoid, sigmoid, dprobit, probit

class GLMBandit:
    def __init__(self, name=None, config=None, load=None):
        if load is None:
            self.name = name
            self.seed = config['seed']
            self.rng = np.random.default_rng(config['seed'])
            self.model = config['model']
            if self.model == 'Logistic':
                self.R = 1
            elif self.model == 'Probit':
                self.R = 1
            else:
                self.R = 5
            self.L = config['num_arms']
            self.d = config['theta_dim']
            self.S = config['theta_norm']
            self.full_norm = config.get('full_norm', None) if config['full_norm'] is not None else False
            self.theta = self.create_parameter()
            self.M = 1.0
            self.T = config['horizon_length']
            self.num_context = self.T
            self.arms = [self.create_arms() for _ in range(self.T)]
            self.kappa = self.calculate_kappa()
            self.t = 0
            self.best_arm, self.max_reward = self.get_best_arm()
        else:
            with open(load, 'r') as f:
                data = json.load(f)
                for i in range(data['num_context']):
                    data['arms'][i] = [np.array(a) for a in data['arms'][i]]
                data['theta'] = np.array(data['theta'])
                self.__dict__ = deepcopy(data)
                self.rng = np.random.default_rng(self.seed)
    
    
    def create_parameter(self):
        theta_proxy = self.rng.uniform(-1, 1, size=self.d)
        if self.full_norm:
            theta = self.S * theta_proxy / np.linalg.norm(theta_proxy)
        else:
            theta = self.S * self.rng.uniform(0, 1) * theta_proxy / np.linalg.norm(theta_proxy)
        return theta
    
    def create_arms(self):
        arms = []
        # i_max = self.rng.integers(self.L)
        # x_max = self.theta / np.linalg.norm(self.theta) * self.M
        i = 0
        while(i < self.L):
            # x_proxy = self.rng.uniform(-1, 1, size=self.d)
            # x_i = self.rng.uniform(0, self.M) * x_proxy / np.linalg.norm(x_proxy)
            x_proxy = self.rng.normal(0.0, 1.0, size=(self.d,))
            norm = np.linalg.norm(x_proxy)
            r = self.rng.uniform(0.0, 1.0) ** (1.0 / self.d)
            x_i = x_proxy * r / norm
            arms.append(x_i)
            i += 1
            # if i == i_max:
            #     arms.append(x_max)
            #     i += 1
            # else:
            #     x_proxy = self.rng.uniform(-1, 1, size=self.d)
            #     x_i = self.rng.uniform(0, self.M) * x_proxy / np.linalg.norm(x_proxy)
            #     if self.model == 'Logistic':
            #         if np.dot(x_i, self.theta) < -0.05 * np.dot(x_max, self.theta):
            #             arms.append(x_i)
            #arms.append(x_i)
            #            i += 1
        return arms
    
    def calculate_kappa(self):
        min_mu_dot = np.inf
        print('Calculating kappa...')
        for arm_set in tqdm(self.arms):
            X_mat = np.array(arm_set)
            dotp = np.dot(X_mat, self.theta)
            if self.model == 'Logistic':
                mu_dot = dsigmoid(dotp)
            elif self.model == 'Probit':
                mu_dot = dprobit(dotp)
            min_mu_dot = min_mu_dot if np.min(mu_dot) > min_mu_dot else np.min(mu_dot)
            # for i, x in enumerate(arm_set):
            #     if self.model == 'Logistic':
            #         mu_dot = dsigmoid(np.dot(x, self.theta))
            #     elif self.model == 'Probit':
            #         mu_dot = dprobit(np.dot(x, self.theta))
            #     if mu_dot < min_mu_dot:
            #         min_mu_dot = mu_dot
        return 1.0/min_mu_dot
    
    def get_best_arm(self):
        max_reward = [- np.inf for _ in range(self.num_context)]
        best_arm = [-1 for _ in range(self.num_context)]
        for j in range(self.num_context):
            for i in range(self.L):
                if self.model == 'Logistic':
                    reward = sigmoid(np.dot(self.theta, self.arms[j][i]))
                elif self.model == 'Probit':
                    reward = probit(np.dot(self.theta, self.arms[j][i]))
                if reward > max_reward[j]:
                    best_arm[j] = i
                    max_reward[j] = reward
        return best_arm, max_reward
    
    def get_first_action_set(self):
        return self.arms[0]

    def get_max_reward(self):
        return self.max_reward[self.t]


    def step(self, action):
        dot_products = [np.dot(self.theta, self.arms[self.t][a]) \
                           for a in action]
        if self.model == 'Logistic':
            rewards = [sigmoid(dot_product) for dot_product in dot_products]
            noisy_rewards = [float(self.rng.binomial(1, reward)) for reward in rewards]
            regrets = [self.max_reward[self.t] - reward for reward in rewards]
        elif self.model == 'Probit':
            rewards = [probit(dot_product) for dot_product in dot_products]
            noisy_rewards = [float(self.rng.normal(dot_product) > 0.0) for dot_product in dot_products]
            regrets = [self.max_reward[self.t] - reward for reward in rewards]
        #print("Best arm:", self.best_arm[self.t])
        self.t += 1
        return noisy_rewards, regrets, self.arms[self.t % self.T]
    
    def reset(self):
        self.t = 0
    

    def save_metadata(self, folder):
        dict_copy = deepcopy(self.__dict__)
        for i in range(dict_copy['num_context']):
            dict_copy['arms'] = [a.tolist() for a in dict_copy['arms']]
        dict_copy['theta'] = dict_copy['theta'].tolist()
        del dict_copy['rng']
        with open(os.path.join(folder, 'Env_Metadata.json'), 'w+') as f:
            json.dump(dict_copy, f, indent=4)
