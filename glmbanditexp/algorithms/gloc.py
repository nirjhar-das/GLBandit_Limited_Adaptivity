
import numpy as np
from glmbanditexp.utils.utils import sigmoid, probit, mat_norm

"""
Class for the GLOC algorithm of [Jun et al. 2017].

Additional Attributes
---------------------
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
kappa: float
    minimal variance
v_matrix_inv: np.array(dim x dim)
    inverse design matrix for Ol2m
zt : np.array(dim)
    for computing theta_hat
tetha_hat : np.array(dim)
    center of confidence set
theta: np.array(dim)
    ONS parameter
oco_regret_bound: float
    data-dependent bound on ONS's OCO regret
conf_width : float
    radius of confidence set 
"""


class Gloc:
    def __init__(self, arm_set, kappa, R, S, model, delta):
        self.name = 'GLOC (Jun et al. 2017)'
        self.model = model
        self.param_norm_ub = S
        self.arm_norm_ub = 1.0
        self.dim = arm_set[0].shape[0]
        self.failure_level = delta
        self.l2reg = self.dim
        self.kappa = 1 / kappa # 1/kappa in the paper, but here to match the def. of [Jun et al. 2017]
        self.v_matrix = self.l2reg * np.eye(self.dim)
        self.v_matrix_inv = (1/self.l2reg)*np.eye(self.dim)
        self.zt = np.zeros((self.dim,))
        self.theta_hat = np.zeros((self.dim,))
        self.theta = np.zeros((self.dim,))
        self.oco_regret_bound = 2 * self.kappa * self.param_norm_ub**2 * self.l2reg
        self.conf_radius = 0
        if self.model == 'Logistic':
            self.L = 0.25
        elif self.model == 'Probit':
            self.L = 1.0 / np.sqrt(2.0 * np.pi)
        self.R = R

        self.arm_set = arm_set
        self.arms = []
        self.rewards = []
        self.regret_arr = []
        self.a_t = -1

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.v_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.zt = np.zeros((self.dim,))
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta = np.random.normal(0, 1, (self.dim,))
        self.oco_regret_bound = 2 * self.kappa * self.param_norm_ub ** 2 * self.l2reg
        self.conf_radius = 0

    def update(self, reward, regret, arm_set):
        """
        Updates estimators.
        """
        # update OCO regret bound (Thm. 3 of [Jun et al. 2017]
        arm = self.arm_set[self.a_t]
        # update design matrix and inverse
        self.v_matrix_inv += - np.dot(self.v_matrix_inv,
                                      np.dot(np.outer(arm, arm), self.v_matrix_inv)) / (
                                     1 + np.dot(arm, np.dot(self.v_matrix_inv, arm)))
        self.v_matrix += np.outer(arm, arm)
        
        if self.model == 'Logistic':
            current_grad = (sigmoid(np.dot(arm, self.theta)) - reward) * arm
        elif self.model == 'Probit':
            current_grad = (probit(np.dot(arm, self.theta)) - reward) * arm
        self.oco_regret_bound += (0.5/self.kappa) * mat_norm(current_grad, self.v_matrix_inv)**2

        # compute new confidence set center
        self.zt += np.dot(self.theta, arm) * arm
        self.theta_hat = np.dot(self.v_matrix_inv, self.zt)

        # compute new ONS parameter
        unprojected_estimate = self.theta - np.dot(self.v_matrix_inv, current_grad) / self.kappa
        self.theta = self.param_norm_ub * unprojected_estimate / np.linalg.norm(unprojected_estimate)

        self.regret_arr.append(regret)
        self.arm_set = arm_set

    def play_arm(self):
        self.update_ucb_bonus()
        # if not arm_set.type == 'ball':
        #     # find optimistic arm
        #     arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        # else:  # TS, only valid for unit ball arm-set
        #     param = gaussian_sample_ellipsoid(self.theta, self.v_matrix, self.conf_radius)
        #     arm = self.arm_norm_ub * param / np.linalg.norm(param)
        
        self.a_t = np.argmax([self.compute_optimistic_reward(arm) for arm in self.arm_set])
        
        return self.a_t

    def update_ucb_bonus(self):
        """
        Updates the ucb bonus function (cf. Thm 1 of [Jun et al. 2017])
        """
        res_square = 1 + (4/self.kappa)*self.oco_regret_bound
        # res_square += 8 * (self.param_norm_ub / self.kappa) ** 2 * np.log(2 * np.sqrt(
        #     1 + 2 * self.oco_regret_bound / self.kappa + 4 * (
        #                 self.param_norm_ub / self.kappa) ** 4 / self.failure_level ** 2) / self.failure_level)
        res_square += 8 * (self.L**2 / self.kappa) ** 2 * np.log(2 * np.sqrt(
            1 + 2 * self.oco_regret_bound / self.kappa + 4 * (
                        self.L**2 / self.kappa) ** 4 / self.failure_level ** 2) / self.failure_level)
        self.conf_radius = np.sqrt(res_square)

    def compute_optimistic_reward(self, arm):
        """
        Returns the UCB for arm.
        """
        norm = mat_norm(arm, self.v_matrix_inv)
        if self.model == 'Logistic':
            pred_reward = sigmoid(np.dot(self.theta, arm))
        elif self.model == 'Probit':
            pred_reward = probit(np.dot(self.theta, arm))
        bonus = self.conf_radius * norm
        return pred_reward+bonus
