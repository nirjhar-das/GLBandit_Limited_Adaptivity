import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def mat_norm(vec, matrix):
    return np.sqrt(np.dot(vec, np.dot(matrix, vec)))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return - np.sum(Y * np.log(sigmoid(np.dot(X, theta))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))
    elif model == 'Probit':
        return - np.sum(Y * np.log(probit(np.dot(X, theta))) + (1 - Y) * np.log(1 - probit(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))

def grad_log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return - np.dot(Y, X) + np.dot(sigmoid(np.dot(X, theta)), X) + lmbda * theta
    elif model == 'Probit':
        return - np.dot(Y, X) + np.dot(probit(np.dot(X, theta)), X) + lmbda * theta

def hess_log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return np.sum([dsigmoid(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])
    elif model == 'Probit':
        return np.sum([dprobit(np.dot(theta, x)) * np.outer(x, x) for x in X], axis=0) + lmbda*np.eye(theta.shape[0])

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def solve_glm_mle(theta_prev, X, Y, lmbda, model):
    # res = minimize(log_loss_glm, theta_prev,\
    #                jac=grad_log_loss_glm, hess=hess_log_loss_glm, \
    #                 args=(X, Y, lmbda, model), method='Newton-CG')
    res = minimize(log_loss_glm, theta_prev, args=(X, Y, lmbda, model))
    # if not res.success:
    #     print(res.message)

    theta_hat, succ_flag = res.x, res.success
    return theta_hat, succ_flag