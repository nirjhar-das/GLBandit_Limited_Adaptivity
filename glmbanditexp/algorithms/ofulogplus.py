import numpy as np
from glmbanditexp.utils.utils import sigmoid
from scipy.optimize import minimize, NonlinearConstraint


class OFULogPlus:
    def __init__(self, arm_set, kappa, R, S, model, T, delta, lazy_update_fr=1, plot_confidence=False, N_confidence=500):
        """
        :param lazy_update_fr:  integer dictating the frequency at which to do the learning if we want the algo to be lazy (default: 1)
        """
        self.name = 'OFULog+ (Lee et al. 2023)'
        self.param_norm_ub = S
        self.arm_norm_ub = 1.0
        self.dim = arm_set[0].shape[0]
        self.failure_level = delta
        self.L = 0.25
        self.lazy_update_fr = lazy_update_fr
        # initialize some learning attributes
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 0
        self.ucb_bonus = 0
        self.log_loss_hat = 0
        # containers
        self.a_t = -1
        self.arm_set = arm_set
        self.arms = []
        self.rewards = []
        self.regret_arr = []
        self.plot = plot_confidence
        self.N = N_confidence
        self.T = T

    def reset(self):
        """
        Resets the underlying learning algorithm
        """
        self.theta_hat = np.random.normal(0, 1, (self.dim,))
        self.ctr = 1
        self.a_t = -1
        self.arms = []
        self.rewards = []
        self.regret_arr = []

    def update(self, reward, regret, next_arms):
        """
        Updates estimator.
        """
        self.arms.append(self.arm_set[self.a_t])
        self.rewards.append(reward)
        self.regret_arr.append(regret)

        # norm-constrained convex optim for MLE
        if self.ctr % self.lazy_update_fr == 0 or len(self.rewards) < 200:
            # if lazy we learn with a reduced frequency
            obj = lambda theta: self.logistic_loss(theta)
            cstrf_norm = lambda theta: np.linalg.norm(theta)
            constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', constraints=[constraint_norm])
            # , options={'maxiter': 20}
            self.theta_hat = opt.x
        # update counter
        self.ctr += 1
        self.arm_set = next_arms

    def play_arm(self):
        self.update_ucb_bonus()
        self.log_loss_hat = self.logistic_loss(self.theta_hat)
        self.a_t = np.argmax([self.compute_optimistic_reward(arm) for arm in self.arm_set])
        return self.a_t

    def update_ucb_bonus(self):
        self.ucb_bonus = 10 * self.dim * np.log(np.e + (self.param_norm_ub * len(self.rewards) / (2 * self.dim))) + 2 * (
                    np.e - 2 + self.param_norm_ub) * np.log(1 / self.failure_level)

    def compute_optimistic_reward(self, arm):
        if self.ctr == 1:
            res = np.random.normal(0, 1)
        else:
            obj = lambda theta: -np.sum(arm * theta)
            cstrf = lambda theta: self.logistic_loss(theta) - self.log_loss_hat
            cstrf_norm = lambda theta: np.linalg.norm(theta)
            constraint = NonlinearConstraint(cstrf, 0, self.ucb_bonus)
            constraint_norm = NonlinearConstraint(cstrf_norm, 0, self.param_norm_ub)
            opt = minimize(obj, x0=self.theta_hat, method='SLSQP', constraints=[constraint, constraint_norm])
            # , options={'maxiter': 20}
            res = np.sum(arm * opt.x)

            ## plot confidence set
            if self.plot and len(self.rewards) == self.T - 2:
                ## store data
                interact_rng = np.linspace(-self.param_norm_ub-0.5, self.param_norm_ub+0.5, self.N)
                x, y = np.meshgrid(interact_rng, interact_rng)
                f = lambda x, y: self.logistic_loss_seq(np.array([x, y])) - self.log_loss_hat
                z = (f(x, y) <= self.ucb_bonus) & (np.linalg.norm(np.array([x, y]), axis=0) <= self.param_norm_ub)
                z = z.astype(int)
                np.savez(f"S={self.param_norm_ub}/OFULogPlus.npz", x=x, y=y, z=z, theta_hat=self.theta_hat)
        return res

    def logistic_loss(self, theta):
        """
        Computes the full log-loss estimated at theta
        """
        res = 0
        if len(self.rewards) > 0:
            coeffs = np.clip(sigmoid(np.dot(self.arms, theta)[:, None]), 1e-12, 1 - 1e-12)
            res += -np.sum(np.array(self.rewards)[:, None] * np.log(coeffs / (1 - coeffs)) + np.log(1 - coeffs))
        return res

    def logistic_loss_seq(self, theta):
        res = 0
        for s, r in enumerate(self.rewards):
            mu_s = 1 / (1 + np.exp(-np.tensordot(self.arms[s].reshape((self.dim,1)), theta, axes=([0], [0]))))
            mu_s = np.clip(mu_s, 1e-12, 1 - 1e-12)
            if r == 0:
                res += -(1 - r) * np.log(1 - mu_s)
            else:
                res += -r * np.log(mu_s)
        return res.squeeze()