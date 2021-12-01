from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # initialize some variables
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    log_lklihood = 0
    for i in range(len(is_correct)):
        log_lklihood += \
            is_correct[i] * (theta[user_id[i]] - beta[question_id[i]]) - \
            np.logaddexp(0, theta[user_id[i]] - beta[question_id[i]])
        # # if c_ij = 1
        # if is_correct[i] == 1:
        #     log_lklihood += theta[user_id[i]] - beta[question_id[i]] \
        #                     - np.logaddexp(0, theta[user_id[i]] -
        #                                    beta[question_id[i]])
        # # if c_ij = 0
        # else:
        #     log_lklihood += - np.logaddexp(0, theta[user_id[i]] -
        #                                    beta[question_id[i]])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # initialize some variables
    user_id = data["user_id"]
    question_id = data["question_id"]
    is_correct = data["is_correct"]

    theta_gradient = np.zeros(len(theta))
    beta_gradient = np.zeros(len(beta))

    for i in range(len(is_correct)):
        # compute the derivatives
        temp = sigmoid(theta[user_id[i]] - beta[question_id[i]])
        grad_theta = is_correct[i] - temp
        grad_beta = - grad_theta
        # apply the update to theta_gradient/beta_gradient
        theta_gradient[user_id[i]] += grad_theta
        beta_gradient[question_id[i]] += grad_beta
    # apply the update
    theta += lr * theta_gradient
    beta += lr * beta_gradient

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)  # num of students
    beta = np.random.rand(1774)  # num of questions

    val_acc_lst = []
    data_acc_lst = []
    val_like_lst = []
    train_like_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta=theta, beta=beta)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score_data = evaluate(data=data, theta=theta, beta=beta)
        score_val = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score_val)
        data_acc_lst.append(score_data)
        val_like_lst.append(-neg_lld_val)
        train_like_lst.append(-neg_lld_train)
        # print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, data_acc_lst, val_like_lst, train_like_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    learning_rate = 0.001
    number_of_iteration = 150
    iteration_list = [*range(1, number_of_iteration + 1, 1)]
    theta, beta, val_acc_list, train_acc_list, val_like_lst, train_like_lst = irt(train_data, val_data, learning_rate, number_of_iteration)

    # report the validation and test accuracy
    plt.plot(val_acc_list, label="validation accuracy")
    plt.plot(train_acc_list, label="training accuracy")
    plt.title("Training Curve")
    plt.xlabel("num of iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    # plot showing the training and valid log-likelihoods

    plt.plot(iteration_list, val_like_lst, label="validation likelihood")
    plt.plot(iteration_list, train_like_lst, label="train likelihood")

    plt.xlabel("iteration number")
    plt.ylabel("likelihood")
    plt.title('log-likelihood of training and validation VS num of iteration')
    plt.legend()
    plt.show()

    # part c
    # report the final validation and test accuracies
    valid_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("The final accuracy for validation set is {}".format(valid_acc))
    print("The final accuracy for test set is {}".format(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    for j in range(1, 4):
        sorted_theta_list = np.sort(theta)
        p_correct = np.exp(-np.logaddexp(0, beta[j] - sorted_theta_list))
        plt.plot(sorted_theta_list, p_correct, label="j_{}".format(j))
    plt.title("correct response vs theta")
    plt.ylabel("probability")
    plt.xlabel("theta")
    plt.legend()
    plt.show()
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
