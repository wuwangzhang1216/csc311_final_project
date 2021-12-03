import os
import sys

# Adding parent directory to path (for importing utils)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# change current working directory to part_a to access data folder
os.chdir("./part_a")
from utils import *

import numpy as np
import matplotlib.pyplot as plt

N_STUDENTS = 542
N_QUESTIONS = 1774


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, C_mat, data_mask):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param C_mat        matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise 
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # initialize some variables
    capability = np.subtract.outer(theta, beta)  # (theta_i - beta_j) matrix
    
    log_lklihood_mat = (C_mat * capability - np.logaddexp(0, capability)) * data_mask
    log_lklihood = np.sum(log_lklihood_mat)
    return -log_lklihood

    # user_id = data["user_id"]
    # question_id = data["question_id"]
    # is_correct = data["is_correct"]

    # log_lklihood = 0
    # for x in range(len(is_correct)):
    #     i, j, c = user_id[x], question_id[x], is_correct[x]
    #     capability = theta[i] - beta[j]
    #     log_lklihood += c * capability - np.logaddexp(0, capability)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, C_mat, data_mask):
    """Update theta and beta using gradient descent.

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
    :param C_mat        matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise 
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # initialize some variables
    capability = np.subtract.outer(theta, beta)  # (theta_i - beta_j) matrix
    sigmoid_cap = sigmoid(capability)
    d_theta_mat = C_mat - sigmoid_cap
    d_beta = (-d_theta_mat * data_mask).T @ np.ones((N_STUDENTS))
    d_theta = (d_theta_mat * data_mask) @ np.ones((N_QUESTIONS))

    theta += lr * d_theta
    beta += lr * d_beta
    return theta, beta

    # theta_gradient = np.zeros(len(theta))
    # beta_gradient = np.zeros(len(beta))

    # for i in range(len(is_correct)):
    #     # compute the derivatives
    #     temp = sigmoid(theta[user_id[i]] - beta[question_id[i]])
    #     grad_theta = is_correct[i] - temp
    #     grad_beta = -grad_theta
    #     # apply the update to theta_gradient/beta_gradient
    #     theta_gradient[user_id[i]] += grad_theta
    #     beta_gradient[question_id[i]] += grad_beta
    # # apply the update
    # theta += lr * theta_gradient
    # beta += lr * beta_gradient

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations, C_mat, data_mask, C_mat_val, data_mask_val):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param C_mat            matrix containing all c_ij values for training data
    :param data_mask        matrix where ij is 1 if there is data and 0 otherwise for training data
    :param C_mat_val        matrix containing all c_ij values for validation data
    :param data_mask_val    matrix where ij is 1 if there is data and 0 otherwise for validation data
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(N_STUDENTS)
    beta = np.random.rand(N_QUESTIONS)


    val_acc_lst = []
    data_acc_lst = []
    val_like_lst = []
    train_like_lst = []

    for i in range(iterations):
        theta = theta.reshape(-1, )
        beta = beta.reshape(-1, )
        neg_lld_train = neg_log_likelihood(data, theta, beta, C_mat, data_mask)
        neg_lld_val = neg_log_likelihood(data, theta, beta, C_mat_val, data_mask_val)
        score_data = evaluate(data, theta, beta, C_mat, data_mask)
        score_val = evaluate(val_data, theta, beta, C_mat_val, data_mask_val)
        val_acc_lst.append(score_val)
        data_acc_lst.append(score_data)
        val_like_lst.append(-neg_lld_val)
        train_like_lst.append(-neg_lld_train)
        print(f"NLLK iter={i}:\t {neg_lld_train} \t Score: {score_data}")
        theta, beta = update_theta_beta(data, lr, theta, beta, C_mat, data_mask)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, data_acc_lst, val_like_lst, train_like_lst


def evaluate(data, theta, beta,C_mat, data_mask):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param C_mat        matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise 
    :return: float
    """
    capability = np.subtract.outer(theta, beta)
    one_parameter_logistic = sigmoid(capability) 
    pred_mat = (one_parameter_logistic >= 0.5) 
    pred_correct = np.sum((pred_mat == C_mat) * data_mask)

    return pred_correct / np.sum(data_mask)
    # pred = []
    # for i, q in enumerate(data["question_id"]):
    #     u = data["user_id"][i]
    #     x = (theta[u] - beta[q]).sum()
    #     p_a = sigmoid(x)
    #     pred.append(p_a >= 0.5)
    # return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def build_data_mat(data):
    """Build C_mat and datamask for the given data

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    C_mat = np.zeros((N_STUDENTS, N_QUESTIONS))
    data_mask = np.zeros((N_STUDENTS, N_QUESTIONS))

    users = data["user_id"]
    questions = data["question_id"]
    is_correct = data["is_correct"]
    for i in range(len(users)):
        u = users[i]
        q = questions[i]
        c = is_correct[i]

        C_mat[u, q] = c
        data_mask[u, q] = 1
    return C_mat, data_mask

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

    C_mat, data_mask = build_data_mat(train_data)
    C_mat_val, data_mask_val = build_data_mat(val_data)
    C_mat_test, data_mask_test = build_data_mat(test_data)

    theta, beta, val_acc_list, train_acc_list, val_like_lst, train_like_lst = irt(
        train_data, val_data, learning_rate, number_of_iteration, C_mat, data_mask, C_mat_val, data_mask_val
    )

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
    plt.title("log-likelihood of training and validation VS num of iteration")
    plt.legend()
    plt.show()

    # part c
    # report the final validation and test accuracies
    valid_acc = evaluate(val_data, theta, beta, C_mat_val, data_mask_val)
    test_acc = evaluate(test_data, theta, beta, C_mat_test, data_mask_test)
    print(f"The final accuracy for validation set is {valid_acc}")
    print(f"The final accuracy for test set is {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    for j in range(1, 4):
        sorted_theta_list = np.sort(theta)
        p_correct = np.exp(-np.logaddexp(0, beta[j] - sorted_theta_list))
        plt.plot(sorted_theta_list, p_correct, label=f"j_{j}")
    plt.title("correct response vs theta")
    plt.ylabel("probability")
    plt.xlabel("theta")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
