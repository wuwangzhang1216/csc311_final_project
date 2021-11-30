from utils import *
from scipy.linalg import sqrtm
from tabulate import tabulate

import numpy as np
import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # update u_n and z_n
    u[n] += lr * (c - u[n].T @ z[q]) * z[q]
    z[q] += lr * (c - u[n].T @ z[q]) * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    val_error_list = []
    train_error_list = []
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        update_u_z(train_data, lr, u, z)
        if i % 2500 == 0:
            val_error_list.append(squared_error_loss(val_data, u, z))
            train_error_list.append(squared_error_loss(train_data, u, z))
    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, val_error_list, train_error_list


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    # Try out different k and select the best k using validation set
    # k_list = [1, 5, 15, 25, 50, 100]
    # accuracy_list = []
    #
    # for k in k_list:
    #     reconstructed = svd_reconstruct(train_matrix, k)
    #     acc = sparse_matrix_evaluate(val_data, reconstructed)
    #     accuracy_list.append(acc)
    #
    # table = [["K value", "accuracy"]]
    # for i in range(len(accuracy_list)):
    #     table.append([k_list[i], accuracy_list[i]])
    # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    #
    # best_k = k_list[accuracy_list.index(max(accuracy_list))]
    # best_reconstructed = svd_reconstruct(train_matrix, best_k)
    # test_acc = sparse_matrix_evaluate(test_data, best_reconstructed)
    # print('the best k={} has a validation accuracy of {} and test accuracy of '
    #       '{}'.format(best_k, max(accuracy_list), test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    # Try out different k and select the best k using validation set
    # k_list = [1, 5, 15, 25, 50, 100]
    # learning_rate = 0.01
    # num_iteration = 500000
    # accuracy_list = []
    # for k in k_list:
    #     mat = als(train_data, k, learning_rate, num_iteration)
    #     acc = sparse_matrix_evaluate(val_data, mat)
    #     accuracy_list.append(acc)
    #
    # table = [["K value", "accuracy"]]
    # for i in range(len(accuracy_list)):
    #     table.append([k_list[i], accuracy_list[i]])
    # print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    #
    # # Report final validation and test accuracy
    # best_k = k_list[accuracy_list.index(max(accuracy_list))]
    # print('the best k={} has a validation accuracy of {}'
    #       .format(best_k, max(accuracy_list)))

    # part e
    # plot and report how the training and validation squared-errorlosses
    # change as a function of iteration
    # Note that the best performance k is 50 showed above and we choose the same
    # hyperparameters as above
    learning_rate = 0.01
    num_iteration = 500000
    chosen_k = 50
    # val_error_list = []
    # train_error_list = []
    iteration_list = [*range(1, num_iteration + 1, 2500)]

    # compute the final matrix
    matrix, val_error_list, train_error_list = als(train_data, val_data, chosen_k, learning_rate, num_iteration)
    # Initialize u and z
    # u = np.random.uniform(low=0, high=1 / np.sqrt(chosen_k),
    #                       size=(len(set(train_data["user_id"])), chosen_k))
    # z = np.random.uniform(low=0, high=1 / np.sqrt(chosen_k),
    #                       size=(len(set(train_data["question_id"])), chosen_k))
    # for i in range(num_iteration):
    #     update_u_z(train_data, learning_rate, u, z)
    #     val_error_list.append(squared_error_loss(val_data, u, z))
    #     train_error_list.append(squared_error_loss(train_data, u, z))

    plt.plot(iteration_list, train_error_list, label="Train")
    plt.plot(iteration_list, val_error_list, label="Validation")
    plt.xlabel("Iteration")
    plt.ylabel("Squared-Error Loss")
    plt.title("Iteration VS Squared-Error Loss for k=50, l_r=0.01")
    plt.legend()
    plt.show()

    # Report final validation and test accuracy
    val_acc = sparse_matrix_evaluate(val_data, matrix)
    test_acc = sparse_matrix_evaluate(test_data, matrix)
    print("Final Validation Accuracy: ", val_acc)
    print("Final Test Accuracy: ", test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
