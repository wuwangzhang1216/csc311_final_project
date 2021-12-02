from utils import *

import numpy as np
import matplotlib.pyplot as plt
import csv

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def initialize_theta_beta(student_mata_data_path, question_meta_data_path):
    # theta = np.full((542, 1), 1)
    # beta = np.full((1774, 1), 1)
    theta = [1] * 542
    beta = [1] * 1774
    elder = []
    younger = []
    with_pre = []
    without_pre = []
    with open(student_mata_data_path) as student_mata_data:
        csv_reader = csv.reader(student_mata_data, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            student_id = row[0]
            birthday = row[2]
            if birthday == "":
                continue
            premium = row[3]
            if premium == "":
                continue
            if int(premium[:1]) == 1:
                with_pre.append(student_id)
            else:
                without_pre.append(student_id)
            if int(birthday[:4]) < 2006:
                elder.append(student_id)
            else:
                younger.append(student_id)

    with open(student_mata_data_path) as student_mata_data:
        csv_reader = csv.reader(student_mata_data, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            student_id = row[0]
            if student_id in elder:
                theta[int(student_id)] += 0.5
            else:
                theta[int(student_id)] -= 0.5
            if student_id in with_pre:
                theta[int(student_id)] -= 0.2
            else:
                theta[int(student_id)] += 0.2
    counter = {}
    with open(question_meta_data_path) as question_meta_data:
        csv_reader = csv.reader(question_meta_data, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            subject_id = row[1]
            if subject_id in counter:
                counter[subject_id] += 1
            else:
                counter[subject_id] = 1
    with open(question_meta_data_path) as question_meta_data:
        csv_reader = csv.reader(question_meta_data, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            question_id = row[0]
            subject_id = row[1]
            if counter[subject_id] > 5:
                beta[int(question_id)] += 0.5
            else:
                beta[int(question_id)] -= 0.5

    return np.array(theta), np.array(beta)


def neg_log_likelihood(data, theta, beta, alpha, k):
    """Compute the negative log-likelihood.

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

    # initialize some variabl；s
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_correct = data["is_correct"]

    log_lklihood = 0
    for x in range(len(is_correct)):
        # i-th student and j-th question
        i, j, c = user_ids[x], question_ids[x], is_correct[x]
        rate = alpha[j] * (theta[i] - beta[j])
        log_lklihood += (
            c * (np.log(k + np.exp(rate)) - np.log(1 - k))
            + np.log(1 - k)
            - np.log(1 + np.exp(rate))
        )

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update(data, lr, theta, beta, alpha, k, c_matrix, in_data_matrix):
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
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    t = theta.copy()
    b = beta.copy()
    k2 = k
    a = alpha.copy()
    diff = (np.subtract.outer(t[:, 0], b[:, 0]).T * a).T
    tb_diff = np.subtract.outer(t[:, 0], b[:, 0]) * np.exp(diff)
    r_diff = k2 + np.exp(diff)
    k_diff = (np.exp(diff).T * a).T

    d_theta = c_matrix * (k_diff / r_diff) - k_diff / (1 + np.exp(diff))
    d_beta = np.dot((-1 * d_theta * in_data_matrix).T, np.ones((542, 1)))
    d_theta = np.dot(d_theta * in_data_matrix, np.ones((1774, 1)))
    d_k = np.dot(
        (
            (-1 * tb_diff / (1 + np.exp(diff)) + c_matrix * tb_diff / r_diff)
            * in_data_matrix
        ).T,
        np.ones((542, 1)),
    )

    theta += lr * d_theta
    beta += lr * d_beta
    alpha = alpha + lr * d_k
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, k, alpha


def irt(data, val_data, lr, iterations, c_matrix, in_data_matrix):
    """Train IRT model.

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
    # theta = np.full((542, 1), 0.5)
    # beta = np.full((1774, 1), 0.5)
    theta, beta = initialize_theta_beta("../data/student_meta.csv", "../data/question_meta.csv")
    k = 0.3
    alpha = np.full((1774, 1), 1)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(
            data, theta=theta[:, 0], beta=beta[:, 0], k=k, alpha=alpha[:, 0]
        )
        score = evaluate(
            data=val_data, theta=theta[:, 0], beta=beta[:, 0], k=k, alpha=alpha[:, 0]
        )
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, k, alpha = update(
            data, lr, theta, beta, alpha, k, c_matrix, in_data_matrix
        )

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, k, alpha, val_acc_lst


def evaluate(data, theta, beta, alpha, k):
    """Evaluate the model given data and return the accuracy.
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
        p_a = k + (1 - k) * sigmoid(alpha[q] * x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


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

    c_matrix = np.zeros((542, 1774))
    in_data_matrix = np.zeros((542, 1774))

    users = train_data["user_id"]
    questions = train_data["question_id"]
    is_correct = train_data["is_correct"]
    for i in range(len(users)):
        u = users[i]
        q = questions[i]
        c = is_correct[i]

        c_matrix[u, q] = c
        in_data_matrix[u, q] = 1

    num_iterations = 26
    lr = 0.01
    theta, beta, k, alpha, val_acc_lst = irt(
        train_data, val_data, lr, num_iterations, c_matrix, in_data_matrix
    )

    fig1 = plt.figure()
    ax = fig1.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    plt.plot(
        [i + 1 for i in range(num_iterations)], val_acc_lst, "-g", label="validation"
    )
    plt.legend(loc="upper right")
    plt.show()

    max_i = np.argmax(np.array(val_acc_lst))
    print(
        "The iteration value with the highest validation accuracy is "
        + str(max_i + 1)
        + " with an accuracy of "
        + str(val_acc_lst[max_i])
    )
    print(
        "The train accuracy is "
        + str(evaluate(train_data, theta[:, 0], beta[:, 0], alpha[:, 0], k))
    )
    print(
        "The val accuracy is "
        + str(evaluate(val_data, theta[:, 0], beta[:, 0], alpha[:, 0], k))
    )
    print(
        "The test accuracy is "
        + str(evaluate(test_data, theta[:, 0], beta[:, 0], alpha[:, 0], k))
    )


if __name__ == "__main__":
    main()
