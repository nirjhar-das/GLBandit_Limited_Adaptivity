import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from glmbanditexp.utils.utils import sigmoid, dsigmoid, mat_norm, solve_glm_mle, probit, dprobit

"""
Class for the GLM-UCB algorithm of [Filippi et al. 2010].

Additional Attributes
---------------------
do_proj : bool
    whether to perform the projection step required by theory
lazy_update_fr : int
    integer dictating the frequency at which to do the learning if we want the algo to be lazy
l2_reg: float
    ell-two regularization of maximum-likelihood problem (lambda)
design_matrix: np.array(dim x dim)
    sum of arms outer product (V_t)
design_matrix_inv: np.array(dim x dim)
    inverse of design_matrix
theta_hat : np.array(dim)
    maximum-likelihood estimator
theta_tilde : np.array(dim)
    projected version of theta_hat
ctr : int
    counter for lazy updates
ucb_bonus : float
    upper-confidence bound bonus
kappa : float
    inverse of minimum worst-case reward-sensitivity
"""


class GlmUCB:
    def __init__(self, arm_set, kappa, R, S, model, T, delta, do_proj=False, lazy_update_fr=5):
        """
        :param do_proj: whether to perform the projection step required by theory (default: False)
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        self.name = 'GLM-UCB (Filippi et al. 2010)'
        self.model = model
        self.param_norm_ub = S
        self.arm_norm_ub = 1.0
        self.dim = arm_set[0].shape[0]
        self.failure_level = delta
        if self.model == 'Logistic':
            self.L = 0.25
        elif self.model == 'Probit':
            self.L = 1.0 / np.sqrt(2.0 * np.pi)
        self.do_proj = do_proj
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.l2reg = self.dim
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
        # self.ctr = 0
        # self.ucb_bonus = 0
        self.kappa = kappa
        self.R = R
        self.const = np.sqrt(3 + 2 * (np.log(1 + 2/self.l2reg)))
        self.conf_multiplier = 2 * self.L * self.const * self.R * self.kappa * \
                                np.sqrt(2 * self.dim * np.log(2 * self.dim * T / self.failure_level))
        # containers
        self.arm_set = arm_set
        self.arms = []
        self.rewards = []
        self.regret_arr = []
        self.a_t = -1
        self.t = 1

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.design_matrix = self.l2reg * np.eye(self.dim)
        self.design_matrix_inv = (1 / self.l2reg) * np.eye(self.dim)
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.theta_tilde = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.arms = []
        self.rewards = []
        self.a_t = -1

    def update(self, reward, regret, arm_set):
        """
        Update the MLE, project if required/needed.
        """
        self.arms.append(self.arm_set[self.a_t])
        self.design_matrix += np.outer(self.arm_set[self.a_t], self.arm_set[self.a_t])
        self.design_matrix_inv += -np.dot(self.design_matrix_inv, np.dot(np.outer(self.arm_set[self.a_t], self.arm_set[self.a_t]), self.design_matrix_inv)) \
                                  / (1 + np.dot(self.arm_set[self.a_t], np.dot(self.design_matrix_inv, self.arm_set[self.a_t])))
        self.rewards.append(reward)
        self.regret_arr.append(regret)
        self.arm_set = arm_set
        self.theta_hat, _ = solve_glm_mle(np.copy(self.theta_hat), np.array(self.arms), np.array(self.rewards), self.l2reg, self.model)
        self.t += 1
        # learn the m.l.e by iterative approach (a few steps of Newton descent)
        # if self.ctr % self.lazy_update_fr == 0 or len(self.rewards) < 200:
        #     # if lazy we learn with a reduced frequency
            
        #     for _ in range(5):
        #         coeffs = sigmoid(np.dot(self.arms, theta_hat)[:, None])
        #         y = coeffs - np.array(self.rewards)[:, None]
        #         grad = self.l2reg * theta_hat + np.sum(y * self.arms, axis=0)
        #         hessian = np.dot(np.array(self.arms).T,
        #                          coeffs * (1 - coeffs) * np.array(self.arms)) + self.l2reg * np.eye(self.dim)
        #         theta_hat -= np.linalg.solve(hessian, grad)
        #     self.theta_hat = theta_hat

        # update counter
        # self.ctr += 1

        # perform projection (if required)
        if self.do_proj and len(self.rewards) > 2:
            if np.linalg.norm(self.theta_hat) > self.param_norm_ub:
                self.theta_tilde = self.theta_hat
            else:
                self.theta_tilde = self.projection(self.arms)
        else:
            self.theta_tilde = self.theta_hat

    def play_arm(self):
        # update bonus bonus
        # self.update_ucb_bonus()
        # if not arm_set.type == 'ball':
        #     # find optimistic arm
        #     arm = np.reshape(arm_set.argmax(self.compute_optimistic_reward), (-1,))
        # else:  # TS, only valid for unit ball arm-set
        #     param = gaussian_sample_ellipsoid(self.theta_tilde, self.design_matrix, self.ucb_bonus)
        #     arm = self.arm_norm_ub * param / np.linalg.norm(param)
        # update design matrix and inverse
        if self.model == 'Logistic':
            means = np.array([sigmoid(np.dot(arm, self.theta_tilde)) for arm in self.arm_set])
            bonuses = self.conf_multiplier * np.array([mat_norm(arm, self.design_matrix_inv) for arm in self.arm_set])
        elif self.model == 'Probit':
            means = np.array([probit(np.dot(arm, self.theta_tilde)) for arm in self.arm_set])
            bonuses = self.conf_multiplier * np.sqrt(np.log(self.t)) * np.array([mat_norm(arm, self.design_matrix_inv) for arm in self.arm_set])
        
        ucb = means + bonuses
        self.a_t = np.argmax(ucb)
        return self.a_t

    # def update_ucb_bonus(self):
    #     """
    #     Updates the UCB bonus.
    #     """
        # logdet = slogdet(self.design_matrix)[1]
        # res = np.sqrt(2 * np.log(1 / self.failure_level) + logdet - self.dim * np.log(self.l2reg))
        # res *= 0.25 * self.kappa
        # res += np.sqrt(self.l2reg)*self.param_norm_ub
        # self.ucb_bonus = res

    # def compute_optimistic_reward(self, arm):
    #     """
    #     Computes the UCB.
    #     """
    #     norm = mat_norm(arm, self.design_matrix_inv)
    #     pred_reward = sigmoid(np.sum(self.theta_tilde * arm))
    #     bonus = self.ucb_bonus * norm
    #     return pred_reward + bonus

    def proj_fun(self, theta, arms):
        """
        Filippi et al. projection function
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        fun = np.dot(diff_gt, np.dot(self.design_matrix_inv, diff_gt))
        return fun

    def proj_grad(self, theta, arms):
        """
        Filippi et al. projection function gradient
        """
        diff_gt = self.gt(theta, arms) - self.gt(self.theta_hat, arms)
        grads = 2 * np.dot(self.design_matrix_inv, np.dot(self.hessian(theta, arms), diff_gt))
        return grads

    def gt(self, theta, arms):
        if self.model == 'Logistic':
            coeffs = sigmoid(np.dot(arms, theta))[:, None]
        elif self.model == 'Probit':
            coeffs = probit(np.dot(arms, theta))[:, None]
        res = np.sum(arms * coeffs, axis=0) + self.l2reg / self.kappa * theta
        return res

    def hessian(self, theta, arms):
        if self.model == 'Logistic':
            coeffs = dsigmoid(np.dot(arms, theta))[:, None]
        elif self.model == 'Probit':
            coeffs = dprobit(np.dot(arms, theta))[:, None]
        res = np.dot(np.array(arms).T, coeffs * arms) + self.l2reg / self.kappa * np.eye(self.dim)
        return res

    def projection(self, arms):
        fun = lambda t: self.proj_fun(t, arms)
        grads = lambda t: self.proj_grad(t, arms)
        norm = lambda t: np.linalg.norm(t)
        constraint = NonlinearConstraint(norm, 0, self.param_norm_ub)
        opt = minimize(fun, x0=np.zeros(self.dim), method='SLSQP', jac=grads, constraints=constraint)
        return opt.x
