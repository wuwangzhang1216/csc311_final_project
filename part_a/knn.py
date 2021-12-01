# import os
# import sys
# # Adding parent directory to path (for importing utils)
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
#
# # change current working directory to part_a to access data folder
# os.chdir("./part_a")

from sklearn.impute import KNNImputer
from utils import *
from tabulate import tabulate
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print(f"Validation Accuracy by user k={k}: {acc}")
    return acc


def sparse_matrix_evaluate_item(data, matrix, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrix: 2D matrix
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        if matrix[cur_question_id, cur_user_id] >= threshold and data["is_correct"][i]:
            total_accurate += 1
        if matrix[cur_question_id, cur_user_id] < threshold and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate_item(valid_data, mat)
    print(f"Validation Accuracy by item k={k}: {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Note that:
    # Please make sure to comment out other parts when running a single part
    # part a and b should be ran at the same time

    # part a - plot KNN impute by user for different values of K
    k_list = [1, 6, 11, 16, 21, 26]
    accuracy_list = []  # used to store the acc for each k
    for k in k_list:
        accuracy_list.append(knn_impute_by_user(sparse_matrix, val_data, k))

    table = [["K value", "Accuracy"]]
    for i in range(len(accuracy_list)):
        table.append([k_list[i], accuracy_list[i]])
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    plt.xlabel("Values for k")
    plt.ylabel("Accuracy")
    plt.plot(k_list, accuracy_list)
    plt.show()

    # part b -  use the kNN model with the best hyperparameter performance
    # on validation set and output performance on test set
    best_k = k_list[accuracy_list.index(max(accuracy_list))]
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("The best performance k value for valid data is {}\n"
          "Its performance on test data is {}".format(best_k, test_accuracy))
    #####################################################################
    # part c - repeat the above two part, but for kNN impute by item/questions
    k_list = [1, 6, 11, 16, 21, 26]
    accuracy_list = []  # used to store the acc for each k
    for k in k_list:
        accuracy_list.append(knn_impute_by_item(sparse_matrix, val_data, k))

    table = [["K value", "Accuracy"]]
    for i in range(len(accuracy_list)):
        table.append([k_list[i], accuracy_list[i]])
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    plt.xlabel("Values for k")
    plt.ylabel("Accuracy")
    plt.plot(k_list, accuracy_list)
    plt.show()

    best_k = k_list[accuracy_list.index(max(accuracy_list))]
    test_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("The best performance k value for valid data is {}\n"
          "Its performance on test data is {}".format(best_k, test_accuracy))
    #####################################################################
    # part d - compares the test performance between user- and item- based kNN
    k_list = [1, 6, 11, 16, 21, 26]
    _user = []
    _item = []
    for k in k_list:
        _user.append(knn_impute_by_user(sparse_matrix, test_data, k))
        _item.append(knn_impute_by_item(sparse_matrix, test_data, k))
    plt.plot(k_list, _user, label="user-based filtering")
    plt.plot(k_list, _item, label="item-based filtering")
    plt.xlabel("Values for k")
    plt.ylabel("Accuracy")
    plt.title('Compare the test performance between user- and item- based')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
