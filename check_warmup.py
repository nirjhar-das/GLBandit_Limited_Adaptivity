import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from glmbanditexp.utils.utils import mat_norm


def check_warmup_count(d=5, K=20, T=20000, kappa=50.0, R=1.0, S=5.0, lmbda=2.0, seed=186329):

    rng = np.random.default_rng(seed)
    count = np.zeros((T,))
    w_const = 1.0 / (0.01 * kappa * R**4 * S**2 * d)
    vectors = rng.normal(loc=0, scale=0.1, size=(T, K, d))
    V_inv = (1.0/lmbda)*np.eye(d)

    for t in tqdm(range(T)):
        flag = False
        max_x, max_norm = None, 0
        for i in range(K):
            x = vectors[t][i]
            mnorm = mat_norm(x, V_inv)**2
            if mnorm > w_const / np.log(2.0 + t):
                if mnorm > max_norm:
                    max_x = x
                    max_norm = mnorm
                flag = True
        if flag:
            count[t] = 1 if t == 0 else count[t-1] + 1
            max_x_V_inv = np.dot(V_inv, max_x)
            V_inv -= np.outer(max_x_V_inv, max_x_V_inv) / (1.0 + max_norm)
        else:
            count[t] = count[t-1] if t > 0 else 0
    
    return count

def avg_warmup_count(num_trials = 5, d=5, K=10, T=200000, kappa=50.0, R=1.0, S=5.0, lmbda=20.0, seed=186329):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(984320, size=num_trials)

    count = np.zeros((T,))
    for i in range(num_trials):
        print('Running trial', i+1)
        count += check_warmup_count(d, K, T, kappa, R, S, lmbda, seed=seeds[i])
    
    count /= num_trials
    return count

def plot_avg_warmup_count(count, show_flag=False, folder = '.'):
    T = count.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(np.arange(1, T+1), count)
    ax.set_title('# Warm-up Round vs Time')
    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel('# Warm-up Rounds')
    if not os.path.exists(os.path.join(folder, 'Results')):
        os.mkdir(os.path.join(folder, 'Results'))
    filename = os.path.join(folder, 'Results', 'Warm-up-Count.png')
    if not show_flag:
        plt.savefig(filename, dpi=200)
    else:
        plt.show()


if __name__ == '__main__':
    count = avg_warmup_count()
    plot_avg_warmup_count(count)
