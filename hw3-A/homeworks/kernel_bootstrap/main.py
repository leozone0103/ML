from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.exp(-gamma * (np.square(np.subtract.outer(x_i, x_j))))


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    m = len(x)
    diag = _lambda * np.eye(m)
    return np.linalg.solve(kernel_function(x, x, kernel_param) + diag, y)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    p = []
    for i in range(num_folds):
        x_train = np.concatenate((x[:i * fold_size], x[(i+1) * fold_size:]))
        y_train = np.concatenate((y[:i * fold_size], y[(i+1) * fold_size:]))
        x_test = x[i * fold_size : (i+1) * fold_size]
        y_test = y[i * fold_size : (i+1) * fold_size]
        predicted = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        k = kernel_function(x_train, x_test, kernel_param)
        y_pred = predicted.dot(k)
        p.append(np.square(y_pred - y_test))  
    return np.mean(p)

@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    i_s = np.linspace(-5, -1)
    _lambdas = 10 ** i_s
    gamma = 1 / np.median(np.square(np.subtract.outer(x, x)))
    errors = [cross_validation(x, y, rbf_kernel, gamma, lamb, num_folds) for lamb in _lambdas]
    smallests = errors.index(min(errors))
    smallest_lamb = _lambdas[smallests]
    return smallest_lamb, gamma



@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    i_s = np.linspace(-5, -1)
    _lambdas = 10 ** i_s
    gamma = 1 / np.median(np.square(np.subtract.outer(x, x)))
    d_s = range(7, 22)
    res = [None, None, None]
    for lam in _lambdas:
        for d in d_s:
            temp = cross_validation(x, y, poly_kernel, d, lam, num_folds)
            if res[0] is None or temp < res[0]:
                res = [temp, lam, d]
    return res[1], res[2]


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        -C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        -D. Repeat A, B, C with x_300, y_300
        -E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    rbf_lam, rbf_gamma = rbf_param_search(x_30, y_30, len(x_30))
    poly_lam, poly_d = poly_param_search(x_30, y_30, len(x_30))
    print("A: rbf kernel optimal values are: gamma = ", rbf_gamma, ", and lambda = ", rbf_lam)
    print("A: poly kernel optimal values are: d = ", poly_d, ", and lambda = ", poly_lam)


    fg = np.linspace(0, 1, num=100)
    plt.figure(1) # rbf
    rbf_y = train(x_30, y_30, rbf_kernel, rbf_gamma, rbf_lam).dot(rbf_kernel(x_30, fg, rbf_gamma))
    plt.plot(x_30, y_30, 'o', label="original data")
    plt.plot(fg, f_true(fg), label="true f(x)")
    plt.plot(fg, rbf_y, label="predicted f(x) in rbf kernel")
    plt.ylim(-6, 6)
    plt.legend(loc ="lower right")
    plt.title("Kernel Ridge Regression for rbf Kernel")
    plt.show()

    plt.figure(1) # poly
    poly_y = train(x_30, y_30, poly_kernel, poly_d, poly_lam).dot(poly_kernel(x_30, fg, poly_d))
    plt.plot(x_30, y_30, 'o', label="original data")
    plt.plot(fg, f_true(fg), label="true f(x)")
    plt.plot(fg, poly_y, label="predicted f(x) in poly kernel")
    plt.legend(loc ="lower right")
    plt.title("Kernel Ridge Regression for poly Kernel")
    plt.show()


    # C:
    rbf_lam_c, rbf_gamma_c = rbf_param_search(x_300, y_300, 10)
    poly_lam_c, poly_d_c = poly_param_search(x_300, y_300, 10)
    print("C: rbf kernel optimal values are: gamma = ", rbf_gamma_c, ", and lambda = ", rbf_lam_c)
    print("C: poly kernel optimal values are: d = ", poly_d_c, ", and lambda = ", poly_lam_c)

    plt.figure(3) # rbf
    rbf_y_c = train(x_300, y_300, rbf_kernel, rbf_gamma_c, rbf_lam_c).dot(rbf_kernel(x_300, fg, rbf_gamma_c))
    plt.plot(x_300, y_300, 'o', label="original data")
    plt.plot(fg, f_true(fg), label="true f(x)")
    plt.plot(fg, rbf_y_c, label="predicted f(x) in rbf kernel")
    plt.ylim(-6, 6)
    plt.legend(loc ="lower right")
    plt.title("Kernel Ridge Regression for rbf Kernel")
    plt.show()

    plt.figure(4) # poly
    poly_y_c = train(x_300, y_300, poly_kernel, poly_d_c, poly_lam_c).dot(poly_kernel(x_300, fg, poly_d_c))
    plt.plot(x_300, y_300, 'o', label="original data")
    plt.plot(fg, f_true(fg), label="true f(x)")
    plt.plot(fg, poly_y_c, label="predicted f(x) in poly kernel")
    plt.legend(loc ="lower right")
    plt.title("Kernel Ridge Regression for poly Kernel")
    plt.show()
    




if __name__ == "__main__":
    main()
