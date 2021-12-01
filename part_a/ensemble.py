from random import randint
from knn import *

import numpy as np


def resemble_data(data, sample_num):
    """ Resemble the given data into a sample_num size.

    :param data: 2D sparse matrix
    :param sample_num: int represents the size of the resembled data
    :return: the data been resembled
    """
    # store the resembled data
    resembled_data = []
    for i in range(sample_num):
        random_pick = randint(0, data.shape[0] - 1)
        resembled_data.append(data[random_pick])
    return np.array(resembled_data)


def bagging(data, sample_num):
    """ bagging the given data into a sample_num size.

    :param data: 2D sparse matrix
    :param sample_num: int represents the size of the resembled data
    :return: three bags
    """
    return resemble_data(data, sample_num), resemble_data(data, sample_num), \
           resemble_data(data, sample_num)


def main():

    # load data as needed
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    bag1, bag2, bag3 = bagging(sparse_matrix, len(sparse_matrix))

    # using three knn models based on three different bags
    # recall from part a the best k value for user_based is k = 11
    # so we use that k value for each knn models

    best_k = 11
    nbrs = KNNImputer(n_neighbors=best_k)
    # collect the prediction for each model and compute the avg

    # bag1
    mat1 = nbrs.fit_transform(bag1.T)
    print("[bag1] Validation Accuracy: {}".
          format(sparse_matrix_evaluate_item(val_data, mat1)))
    # bag2
    mat2 = nbrs.fit_transform(bag2.T)
    print("[bag2] Validation Accuracy: {}".
          format(sparse_matrix_evaluate_item(val_data, mat1)))
    # bag3
    mat3 = nbrs.fit_transform(bag3.T)
    print("[bag3] Validation Accuracy: {}".
          format(sparse_matrix_evaluate_item(val_data, mat1)))

    avg = (mat1 + mat2 + mat3) / 3

    print("Final Validation Accuracy: {}".
          format(sparse_matrix_evaluate_item(val_data, avg)))

    print("Final Test Accuracy: {}".
          format(sparse_matrix_evaluate_item(test_data, avg)))


if __name__ == "__main__":
    main()
