from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2 * np.sum(np.square(X), axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    n, d = X.shape
    b = (1 / n) * np.sum(y - (weight.T).dot(X.T))
    for k in range(0, d):
        exclued_sum_added_bias = b + (weight.T).dot(X.T) - np.multiply(weight[k], X[:, k])
        c_k = 2 * X[:, k].dot(y - exclued_sum_added_bias)
        if c_k < -_lambda:
          weight[k] = (c_k + _lambda) / a[k]
        elif c_k > _lambda:
          weight[k] = (c_k - _lambda) / a[k]
        else: 
          weight[k] = 0
    return [weight, b]

@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    n, _ = X.shape
    predicted_y = (weight.T).dot(X.T) + bias
    print(y.shape)
    mse = np.sum((predicted_y - y) ** 2) + _lambda * np.sum(np.abs(weight))
    return mse

@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None
    w = start_weight.copy()
    while not convergence_criterion(w, old_w, convergence_delta):
        old_w = np.copy(w)
        (w, bias) = step(X, y, w, a, _lambda)
    return (w, bias)    


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if old_w is None:
        return False
    return np.max(np.abs(weight-old_w)) <= convergence_delta



@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # input:
    n = 500
    d = 1000
    k = 100
    sd = 1
    # First get sythetic data and standardize it:
    X = np.random.normal(size=(n,d))
    X = (X - np.mean(X, axis=0))
    X = X / np.std(X, axis=0)

    w = np.array(list(range(1, k + 1)))
    w = np.append(w, np.zeros(d - len(w)))
    w = w / k

    offset = np.random.normal(scale=sd, size=(n, ))
    y = np.matmul(w, X.T) + offset

    # Finding lambda:
    lambs = np.zeros(k)
    lambs[0] = 2 * np.max(np.abs(np.sum(X.T * (y - (1 / n) * np.sum(y)), axis=1)))
    for j in range(1, k):
        lambs[j] = lambs[j-1] / 2

    # training
    W_train = train(X, y, lambs[0])[0]
    count_non_zero = np.zeros(k)
    count_non_zero[0] = np.count_nonzero(W_train)

    fdr = []
    tpr = []
    fdr.append(0)
    tpr.append(0)
    i = 0
    while count_non_zero[i] < d - 5:
        trained_set = train(X, y, lambs[i+1], start_weight=W_train)[0]
        W_train = trained_set
        counts_non_zero = np.count_nonzero(trained_set)
        count_non_zero[i+1] = counts_non_zero
        if np.count_nonzero(trained_set) == 0 or k == 0:
            continue
        fdr.append(np.count_nonzero(trained_set[k:]) / count_non_zero[i+1])
        tpr.append(np.count_nonzero(trained_set[:k]) / k)
        i += 1
        
    # plot

    plt.figure(1)
    plt.plot(lambs[:i], count_non_zero[:i])
    plt.xlabel('lambda')
    plt.ylabel('non-zeros')
    plt.xscale('log')
    
    plt.figure(2)
    plt.plot(fdr[:i], tpr[:i])
    plt.xlabel('false discovery rate (FDR)')
    plt.ylabel('true positive rate (TPR)')
    
    plt.show()    



if __name__ == "__main__":
    main()
