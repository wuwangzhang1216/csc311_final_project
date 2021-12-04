import os
import sys

# Adding parent directory to path (for importing utils)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# change current working directory to part_a to access data folder
os.chdir("./part_a")
from utils import *
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import csv

N_STUDENTS = 542
N_QUESTIONS = 1774
VARIATION = 0.1

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


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


def initialize_theta_beta(student_mata_data_path, question_meta_data_path):
    # theta = np.full((542, 1), 1)
    # beta = np.full((1774, 1), 1)
    theta = [0.5] * N_STUDENTS
    beta = [0.5] * N_QUESTIONS
    older = []
    younger = []
    with_pre = []
    without_pre = []
    with open(student_mata_data_path) as student_mata_data:
        csv_reader = csv.reader(student_mata_data, delimiter=",")
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
                older.append(student_id)
            else:
                younger.append(student_id)

    with open(student_mata_data_path) as student_mata_data:
        csv_reader = csv.reader(student_mata_data, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            student_id = row[0]
            if student_id in older:
                theta[int(student_id)] += VARIATION
            else:
                theta[int(student_id)] -= VARIATION
            if student_id in with_pre:
                theta[int(student_id)] -= VARIATION
            else:
                theta[int(student_id)] += VARIATION
    counter = {}
    with open(question_meta_data_path) as question_meta_data:
        csv_reader = csv.reader(question_meta_data, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            subject_id = row[1]
            if subject_id in counter:
                counter[subject_id] += 1
            else:
                counter[subject_id] = 1
    with open(question_meta_data_path) as question_meta_data:
        csv_reader = csv.reader(question_meta_data, delimiter=",")
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            question_id = row[0]
            subject_id = row[1]
            if counter[subject_id] > 5:
                beta[int(question_id)] += VARIATION
            else:
                beta[int(question_id)] -= VARIATION

    return np.array([theta]).T, np.array([beta]).T


def neg_log_likelihood(data, theta, beta, alpha, k, C_mat, data_mask):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param C_mat:       matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise
    :return: float
    """
    t, b = theta.reshape(-1,), beta.reshape(
        -1,
    )
    capability = np.subtract.outer(t, b)  # (theta_i - beta_j) matrix
    difference = (capability.T * alpha).T  # alpha * (theta - beta) matrix
    exp_diff = np.exp(difference)  # exp{ alpha * (theta - beta)}
    log_lklihood_mat = (
        C_mat * (np.log(k + exp_diff) - np.log(1 - k))
        + np.log(1 - k)
        - np.log(1 + exp_diff)
    ) * data_mask
    return -np.sum(log_lklihood_mat)

    # user_ids = data["user_id"]
    # question_ids = data["question_id"]
    # is_correct = data["is_correct"]

    # log_lklihood = 0
    # for x in range(len(is_correct)):
    #     # i-th student and j-th question
    #     i, j, c = user_ids[x], question_ids[x], is_correct[x]
    #     rate = alpha[j] * (theta[i] - beta[j])
    #     log_lklihood += (
    #         c * (np.log(k + np.exp(rate)) - np.log(1 - k))
    #         + np.log(1 - k)
    #         - np.log(1 + np.exp(rate))
    #     )
    # return -log_lklihood


def update(lr, theta, beta, alpha, k, C_mat, data_mask):
    """Update theta and beta using alternating gradient descent.

    :param lr: float
    :param theta:       Vector shape (N_STUDENT, 1)   - students ability
    :param beta:        Vector shape (N_QUESTIONS, 1) - questions difficulty
    :param alpha:       Vector shape (N_QUESTIONS, 1) - discrimination ability of questions
    :param k:           Scalar - psuedo-guessing parameter
    :param C_mat:       matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise
    :return: tuples (theta, beta, alpha, k) of item response 3PL parameters
    """
    # refer to report for derivative of parameter dl/d_theta, dl/d_beta, dl/d_alpha
    # capability: matrix of student capability - (theta_i - beta_j)
    # difference: matrix of student capability scaled by discrimination - alpha_j * (theta_i - beta_j)
    t, b = theta.reshape(-1,), beta.reshape(
        -1,
    )

    capability = np.subtract.outer(t, b)  # (theta_i - beta_j) matrix
    difference = (capability.T * alpha).T  # alpha * (theta - beta) matrix
    exp_diff = np.exp(difference)  # exp{ alpha * (theta - beta)}

    cap_exp = capability * exp_diff  # (theta - beta) * exp{ alpha (theta - beta)}
    a_exp = (exp_diff.T * alpha).T  # alpha * exp{ alpha (theta - beta)}

    # the matrices have values for all i, j and depends on C_ij values and only
    # counting data from training set

    # - So we populate matrices values depending on c_ij using C-mat
    # - then mask the matrices to count only the data we have
    # - sum for each over row-wise or column-wise (depending on the parameter) to
    # get derivative by @ np.ones()

    d_theta_mat = C_mat * (a_exp / (k + exp_diff)) - (a_exp / (1 + exp_diff))
    d_beta = (-d_theta_mat * data_mask).T @ np.ones((N_STUDENTS, 1))
    d_theta = d_theta_mat * data_mask @ np.ones((N_QUESTIONS, 1))
    d_alpha = (
        (C_mat * cap_exp / (k + exp_diff) - cap_exp / (1 + exp_diff)) * data_mask
    ).T @ np.ones((N_STUDENTS, 1))

    theta += lr * d_theta
    beta += lr * d_beta
    alpha += lr * d_alpha
    return theta, beta, alpha, k


def irt(
    data,
    val_data,
    lr,
    iterations,
    theta,
    beta,
    k,
    C_mat,
    data_mask,
    C_mat_val,
    data_mask_val,
):
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

    # no discrimination of questions' ability to differentiate between how
    # competent the student is and how diffucult the question is thus a_i = 1
    # initially a_j (theta_i - beta_j) = 1 * (theta_i - beta_j)
    alpha = np.ones((1774, 1))

    val_acc_lst = []
    data_acc_lst = []
    val_like_lst = []
    train_like_lst = []

    for i in range(iterations):
        neg_lld_train = neg_log_likelihood(data, theta, beta, alpha, k, C_mat, data_mask)
        neg_lld_val = neg_log_likelihood(val_data, theta, beta, alpha, k, C_mat_val, data_mask_val)
        score_data = evaluate(data, theta, beta, alpha, k, C_mat, data_mask)
        score_val = evaluate(val_data, theta, beta, alpha, k, C_mat_val, data_mask_val)
        val_acc_lst.append(score_val)
        data_acc_lst.append(score_data)
        val_like_lst.append(-neg_lld_val)
        train_like_lst.append(-neg_lld_train)
        print(f"NLLK iter={i + 1}:\t {neg_lld_train} \t Score: {score_val}")
        theta, beta, alpha, k = update(lr, theta, beta, alpha, k, C_mat, data_mask)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, k, val_acc_lst, data_acc_lst, val_like_lst, train_like_lst


def evaluate(data, theta, beta, alpha, k, C_mat, data_mask):
    """Evaluate the model given data and return the accuracy.
    based on the item response theory 3PL formula

    p(c=1 | theta) = k + (1-k) * sigmoid(alpha * (theta - beta))

    :param data: Adictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param C_mat:       matrix containing all c_ij values
    :param data_mask    matrix where ij is 1 if there is data and 0 otherwise
    :return: float
    """
    theta, beta = theta.reshape(-1,), beta.reshape(
        -1,
    )
    capability = np.subtract.outer(theta, beta)

    sigmoid_cap = sigmoid(capability)
    pred_mat = sigmoid_cap >= 0.5
    pred_correct = np.sum((pred_mat == C_mat) * data_mask)

    difference = (
        np.subtract.outer(theta, beta).T * alpha
    ).T  # alpha * (theta - beta) matrix

    three_parameter_logistic = k + (1 - k) * sigmoid(difference)
    pred_mat = three_parameter_logistic >= 0.5
    pred_correct = np.sum((pred_mat == C_mat) * data_mask)

    return pred_correct / np.sum(data_mask)
    # alpha = alpha.reshape(-1,)
    # pred = []
    # for i, q in enumerate(data["question_id"]):
    #     u = data["user_id"][i]
    #     x = (theta[u] - beta[q]).sum()
    #     p_a = k + (1 - k) * sigmoid(alpha[q] * x)
    #     pred.append(p_a >= 0.5)
    # n = np.sum((data["is_correct"] == np.array(pred)))
    # return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def predictions(data, theta, beta, alpha, k)->List[int]:
    """ Return the IRT predictions given the 4 parameters
    
    :param theta:       Vector shape (N_STUDENT, 1)   - students ability
    :param beta:        Vector shape (N_QUESTIONS, 1) - questions difficulty
    :param alpha:       Vector shape (N_QUESTIONS, 1) - discrimination ability of questions
    :param k:           Scalar - psuedo-guessing parameter
    """
    theta, beta, alpha = alpha.reshape(-1,), beta.reshape(-1,), alpha.reshape(-1,)
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = k + (1 - k) * sigmoid(alpha[q] * x)
        pred.append(p_a >= 0.5)
    pred = list(map(int, pred))
    return pred

def competition_csv(theta, beta, alpha, k):
    private_test_data = load_private_test_csv("../data")
    pred = predictions(private_test_data, theta, beta, alpha, k)
    ans = private_test_data
    ans["is_correct"] = pred
    save_private_test_csv(ans)


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

    # ======================================================== #
    # initialization

    # theta = np.full((542, 1), 0.5)
    # beta = np.full((1774, 1), 0.5)

    # theta = np.random.rand(N_STUDENTS).reshape(-1,1)  # num of students
    # beta = np.random.rand(N_QUESTIONS).reshape(-1,1)  # num of questions

    # heuristically assign ability and difficulty based on student/question metadata
    theta, beta = initialize_theta_beta(
        "../data/student_meta.csv", "../data/question_meta.csv"
    )

    # alpha is best suited to have initial value of 1 (no discrimonation)

    # psuedo-guessing parameter, since we have 4 possible answer to a question
    # and assuming none of the answer are trivially false, then 1/4 = 0.25 is
    # the psuedo-guessing parameter
    k = 0.25
    # ======================================================== #
    # hyperparameters
    num_iterations = 80
    lr = 0.003
    # ======================================================== #

    C_mat, data_mask = build_data_mat(train_data)
    C_mat_val, data_mask_val = build_data_mat(val_data)
    C_mat_test, data_mask_test = build_data_mat(test_data)

    theta, beta, alpha, k, val_acc_list, train_acc_list, val_like_lst, train_like_lst= irt(
        train_data,
        val_data,
        lr,
        num_iterations,
        theta,
        beta,
        k,
        C_mat,
        data_mask,
        C_mat_val,
        data_mask_val,
    )
    
    

    max_i = np.argmax((val_acc_list))
    print(
        f"The iteration value with the highest validation accuracy is {max_i + 1} with an accuracy of {val_acc_list[max_i]}"
    )
    print(
        f"The train accuracy is {evaluate(train_data, theta, beta, alpha, k, C_mat, data_mask)}"
    )
    print(
        f"The val accuracy is {evaluate(val_data, theta, beta, alpha, k, C_mat_val, data_mask_val)}"
    )
    print(
        f"The test accuracy is {evaluate(test_data, theta, beta, alpha, k, C_mat_test, data_mask_test)}"
    )
    competition_csv(theta, beta, alpha, k)

    # report the validation and test accuracy
    plt.plot(val_acc_list, label="validation accuracy")
    plt.plot(train_acc_list, label="training accuracy")
    plt.title("Training Curve")
    plt.xlabel("num of iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


    # plot showing the training and valid log-likelihoods
    iteration_list = [*range(1, num_iterations + 1, 1)]
    plt.plot(iteration_list, val_like_lst, label="validation likelihood")
    plt.plot(iteration_list, train_like_lst, label="train likelihood")

    plt.xlabel("iteration number")
    plt.ylabel("likelihood")
    plt.title("log-likelihood of training and validation VS num of iteration")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
