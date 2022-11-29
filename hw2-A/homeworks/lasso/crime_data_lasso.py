if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
import pandas as pd 

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    trails = 200
    start_col = "ViolentCrimesPerPop"
    X_train = df_train.drop(start_col, axis=1).values
    Y_train = df_train[start_col].values
    X_test = df_test.drop(start_col, axis=1).values
    Y_test = df_test[start_col].values

    entries = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    col_index = [df_train.columns.get_loc(col)-1 for col in entries]

    # initializing lambdas
    lambs = []
    n = len(Y_train)
    nt = len(Y_test)
    lambs.append(2 * np.max(np.abs(np.sum(X_train.T * (Y_train - ((1 / n) * np.sum(Y_train))), axis=1))))
    j = 0
    while (lambs[j] >= 0.01):
        lambs.append(lambs[j] / 2)
        j += 1
    lamb_size = len(lambs)

    # train:
    W_train = train(X_train, Y_train, lambs[0])[0]
    count_non_zeros = np.zeros(lamb_size + 1)
    a1 = np.zeros(lamb_size)
    a2 = np.zeros(lamb_size)
    a3 = np.zeros(lamb_size)
    a4 = np.zeros(lamb_size)
    a5 = np.zeros(lamb_size)

    training_error = []
    test_error = []
    for i in range(lamb_size - 1):
        train_set, bias = train(X_train, Y_train, lambs[i+1], start_weight=W_train)
        W_train = train_set
        count_non_zeros[i+1] = np.count_nonzero(train_set)
        a1[i] = train_set[col_index[0]]
        a2[i] = train_set[col_index[1]]
        a3[i] = train_set[col_index[2]]
        a4[i] = train_set[col_index[3]]
        a5[i] = train_set[col_index[4]]
        training_error.append((1/n) * np.sum((np.matmul(X_train, train_set) - Y_train + bias) ** 2))
        test_error.append((1/nt) * np.sum((np.matmul(X_test, train_set) - Y_test + bias) ** 2))
        print(lambs[i+1])
        print(train_set)

    plotting
    plt.figure(1)
    plt.plot(lambs, count_non_zeros[:-1])
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('non-zeros')
    plt.show()

    plt.figure(2)
    plt.plot(lambs, a1)
    plt.plot(lambs, a2)
    plt.plot(lambs, a3)
    plt.plot(lambs, a4)
    plt.plot(lambs, a5)
    plt.xlabel('lambda')
    plt.xscale('log')
    plt.legend(entries, loc ="lower right")
    plt.show()

    plt.figure(3)
    plt.plot(lambs[:-1], training_error)
    plt.plot(lambs[:-1], test_error)
    plt.xscale('log')
    plt.legend(["traning error", "test error"], loc ="lower right")
    plt.xlabel('lambda')
    plt.show()

if __name__ == "__main__":
    main()
