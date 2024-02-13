import numpy as np
from utils import mat_norm, solve_glm_mle, dsigmoid, dprobit

class RSGLBandit:
    def __init__(self, arm_set, lmbda, C, wc, gm, kappa, R, S, model):
        self.arms = arm_set
        self.d = self.arms[0].shape[0]
        self.K = len(self.arms)
        self.lmbda = lmbda
        self.kappa = kappa
        self.R = R
        self.S = S
        self.V_inv = (1.0 / self.lmbda) * np.eye(self.d)
        self.curr_H = self.lmbda * np.eye(self.d)
        self.prev_H = self.lmbda * np.eye(self.d)
        self.model = model
        self.t = 1
        self.wc = wc
        self.w_const = 1.0 / (self.wc * self.kappa * self.R**4 * self.S**2 * self.d)
        self.theta_hat_w = np.zeros((self.d,))
        self.theta_hat_tau = np.zeros((self.d,))
        self.warm_up_X, self.warm_up_Y = [], []
        self.non_warm_up_X, self.non_warm_up_Y = [], []
        self.flag = False
        self.C = C
        self.gm = gm
        self.reg = self.gm * self.S * np.sqrt(self.d * np.log(1.0 +self.t))
        self.e = np.exp(1.0)
        self.regret_arr = []
    

    def play_arm(self):
        # Check warm up

        self.flag = False
        max_x, max_norm, max_ind = None, 0, -1
        for i in range(self.K):
            x = self.arms[i]
            mnorm = mat_norm(x, self.V_inv)**2
            if mnorm > (self.w_const / np.log(1.0 + self.t)):
                if mnorm > max_norm:
                    max_x = x
                    max_norm = mnorm
                    max_ind = i
                self.flag = True
        if self.flag:
            print('Warm up round at step', self.t)
            max_x_V_inv = np.dot(self.V_inv, max_x)
            self.V_inv -= np.outer(max_x_V_inv, max_x_V_inv) / (1.0 + max_norm)
            self.a_t = max_ind
        
        # Non warm-up
        else:
            # Check determinant
            if np.linalg.det(self.curr_H) > ((1 + self.C) * np.linalg.det(self.prev_H)):
                self.prev_H = self.curr_H
                thth, succ_flag = solve_glm_mle(self.theta_hat_tau, np.array(self.non_warm_up_X), \
                                                np.array(self.non_warm_up_Y), self.reg, self.model)
                if succ_flag:
                    self.theta_hat_tau = thth
            
            # Eliminate arms based on warm-up theta
            max_lcb = -np.inf
            arm_idx = []
            ucb_arr = []
            # Compute LCB and UCB of each arm
            for i in range(self.K):
                lcb_idx = np.dot(self.theta_hat_w, self.arms[i]) \
                        - self.reg * np.sqrt(self.kappa) * mat_norm(self.arms[i], self.V_inv)
                ucb_idx = np.dot(self.theta_hat_w, self.arms[i]) \
                        + self.reg * np.sqrt(self.kappa) * mat_norm(self.arms[i], self.V_inv)
                ucb_arr.append(ucb_idx)
                if lcb_idx > max_lcb:
                    max_lcb = lcb_idx
            # Eliminate arms
            for i in range(self.K):
                if ucb_arr[i] >= max_lcb:
                    arm_idx.append(i)
            # Find max UCB arm
            max_ind = -np.inf
            self.a_t = -1
            for i in arm_idx:
                ucb_ind = np.dot(self.theta_hat_tau, self.arms[i]) \
                        + self.reg * mat_norm(self.arms[i], np.linalg.inv(self.prev_H))
                if ucb_ind > max_ind:
                    max_ind = ucb_ind
                    self.a_t = i
        #print("Arm played", self.a_t)
        return self.a_t
    
    def update(self, reward, regret, next_arms):
        self.regret_arr.append(regret)
        # If this was a warm up round, append (x, y) to warm-up set, re-compute theta_hat_w
        if self.flag:
            self.warm_up_X.append(self.arms[self.a_t])
            self.warm_up_Y.append(reward)
            thth, succ_flag = solve_glm_mle(self.theta_hat_w, np.array(self.warm_up_X), \
                                                np.array(self.warm_up_Y), self.reg, self.model)
            if succ_flag:
                self.theta_hat_w = thth # update theta_hat_w if mle solution was successful
        
        # Else add (x, y) to non warm-up set
        else:
            self.non_warm_up_X.append(self.arms[self.a_t])
            self.non_warm_up_Y.append(reward)
            if self.model == 'Logistic':
                mudp = dsigmoid(np.dot(self.arms[self.a_t], self.theta_hat_w))
            elif self.model == 'Probit':
                mudp = dprobit(np.dot(self.arms[self.a_t], self.theta_hat_w))
            # Update current H matrix
            self.curr_H += (mudp / self.e) * np.outer(self.arms[self.a_t], self.arms[self.a_t])
        
        self.t += 1
        self.reg = self.gm * self.S * np.sqrt(self.d * np.log(1.0 + self.t))
        self.arms = next_arms
            

            


            


