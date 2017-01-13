#! /usr/bin/python

'''
Vidya Sagar Kalvakunta (vkalvaku)
Subhash Bylaiah (sybylaiah)
'''

import numpy as np
import math
import random
import sys
from collections import Counter, defaultdict


def readfile(dataset):
    data_array = []
    with open(dataset, 'r') as readfile:
        next(readfile)  # Skip header row
        for line in readfile:
            data = line.strip('\n').split(",")
            data.pop(21)
            # Moving class attribute to last column and removing the unique Id
            data.append(-1 if data.pop(20) == '0' else +1)
            data_array.append(np.asarray(data).astype(np.int))
    return data_array


class Node(object):
    """Node is a data structure represent a node in the decision tree"""

    def __init__(self, attribute_idx=-1,
                 attribute_val=None, isleaf=None, depth=0):
        self.children = defaultdict()
        self.depth = depth
        self.isleaf = isleaf
        self.attribute_idx = attribute_idx
        if attribute_val:
            for key, value in attribute_val:
                self.children[key] = value

            # The max depth of the tree


def calc_entropy(data_array, data_weights):
    # Counter function returns the count of each class as a dictionary
    class_count = set(data_array)
    data_array = data_array.reshape(len(data_array), 1)
    N = data_array.shape[0]

    # initialize to uniform weights if weigts are not provided
    if len(data_weights) == 0:
        print "here"
        data_weights = 1.0 / N * np.ones(data_array.shape)

    labels_and_weights = np.hstack((data_weights, data_array))
    entropy_val = 0
    total_sum = sum(np.asarray(labels_and_weights[:, 0], dtype=float))
    for c in class_count:
        filtered_data = labels_and_weights[labels_and_weights[:, 1] == c]
        class_sum = sum(np.asarray(filtered_data[:, 0], dtype=float))
        p_x = float(class_sum) / total_sum
        entropy_val -= p_x * math.log(p_x, 2)  # Calculating the entropy
    return entropy_val


def calc_entr(data_array, data_weights):
    # Counter function returns the count of each class as a dictionary
    class_count = set(data_array)

    entropy_val = 0
    total_sum = sum(data_weights)
    for c in class_count:
        class_sum = 0
        for cl, wt in zip(data_array, data_weights):
            if cl == c:
                class_sum += wt
        p_x = float(class_sum) / total_sum
        entropy_val -= p_x * math.log(p_x, 2)  # Calculating the entropy
    return entropy_val


def maxclass_count(data_array, data_weights=[]):
    # Function to return the class with the maximum count of the given data array
    # Counter function returns the count of each class as a dictionary
    class_count = set(data_array)
    data_array = data_array.reshape(len(data_array), 1)
    N = data_array.shape[0]

    # initialize to uniform weights if weigts are not provided
    if len(data_weights) == 0:
        data_weights = np.ones(data_array.shape[0])

    labels_and_weights = np.hstack((data_weights, data_array))

    max_class_weight = -sys.maxint
    max_class = None

    for c in class_count:
        filtered_data = labels_and_weights[labels_and_weights[:, 1] == c]
        class_sum = sum(np.asarray(filtered_data[:, 0], dtype=float))
        if class_sum > max_class_weight:
            max_class_weight = class_sum
            max_class = c
    return max_class


def classifytree(data_array, max_depth=2, weights=None, depth=1):  # Function to build the classification tree
    if len(data_array) is None:  # If the there is no data to classify, return empty node
        return Node()
    if weights is None:
        N = data_array.shape[0]
        weights = np.ones((N, 1), dtype=np.int)

    tot_entropy = calc_entr(data_array[:, -1], weights)  # Call function to calculate the entropy

    cum_info_gain = -9999

    for column in xrange(len(data_array[
                                 0]) - 1):  # Loop through all the columns in the data array except the last column, The last column contains the class information
        unique_values = set(data_array[:, column])  # Set of al the unique values in the present column
        entropy = 0
        for value in unique_values:
            vfilter = data_array[:, column] == value
            filtered_array = data_array[vfilter]  # Split the column based on the value
            x2 = float(filtered_array.shape[0]) / data_array.shape[0]
            entropy += x2 * calc_entr(filtered_array[:, -1], weights[vfilter])

        info_gain = tot_entropy - entropy
        if info_gain > cum_info_gain:
            cum_info_gain = info_gain
            values = unique_values
            attribute = column

    if (
        cum_info_gain > -10) and depth <= max_depth:  # A flag to check if the there is any significant info gain on classfying the present array

        children = []
        for value in values:
            vfilter = data_array[:, attribute] == value
            if depth == max_depth:  # If the required depth is reached just return Node without increasing the tree further
                max_class = maxclass_count(data_array[vfilter][:, -1], weights[vfilter])
                child_node = Node(isleaf=max_class, depth=depth + 1)
            else:
                child_node = classifytree(data_array[vfilter], max_depth, weights[vfilter], depth + 1)
            children.append((value, child_node))
        return Node(attribute_idx=attribute, attribute_val=children, depth=depth)

    # If there is no info gain obtained, just return the class with max count
    return Node(isleaf=maxclass_count(data_array[:, -1]), depth=depth)


def test(tree_list, test_array, voting_weights=None):
    tp = tn = fp = fn = 0
    predictions = []
    for row in test_array:
        predicted_class = []

        for ctree in tree_list:
            while ctree.isleaf is None:  # Pass down the tree until a leaf node is reached
                ctree = ctree.children.get(row[ctree.attribute_idx], Node(isleaf=3))
            predicted_class.append(ctree.isleaf)

        # We are storing the class value in the last column, so comparing the predicted class value in isleaf variable
        # with the actual to get a count of the positive matches.
        if not voting_weights:
            max_prediction = Counter(predicted_class).most_common(1)[0][0]
        else:
            N = len(predicted_class)
            max_prediction = int(np.sign(
                float(np.dot(np.asarray(predicted_class).reshape((1, N)), np.asarray(voting_weights).reshape(N, 1)))))

        # target = 1 if row[-1] == 'Yes' else -1
        target = row[-1]
        if target == max_prediction:
            if target == 1:
                tp += 1
            else:
                tn += 1
        else:
            if target == 1:
                fn += 1
            else:
                fp += 1
        predictions.append(max_prediction)
    return ((tp, tn, fp, fn), predictions)


def predict(tree, input):
    while tree.isleaf is None:  # Pass down the tree until a leaf node is reached
        tree = tree.children.get(input[tree.attribute_idx], Node(isleaf=3))
    return tree.isleaf


def confusionMatrix(pn):
    p = pn[0] + pn[1]
    n = pn[2] + pn[3]

    print  "=" * 12 + "\t\tConfusion matrix\t" + "=" * 12
    print "\n {:^16} {:^20} {:^20} ".format("", "Actual False", "Actual True")
    print "\n {:^16} {:^20} {:^20} ".format("Predicted False", str(pn[1]), str(pn[2]))
    print "\n {:^16} {:^20} {:^20} ".format("Predicted True", str(pn[3]), str(pn[0]))

    print "\nCorrectly classified examples:", p, "\tMisclassified examples:", n
    print "\nAccuracy on test set", (float(p) / (p + n)) * 100, "%"

    print "TPR: ", (float(pn[0]) / (pn[0] + pn[3])) * 100, "%"
    print "FPR: ", (float(pn[2]) / (pn[2] + pn[1])) * 100, "%"

    return (float(p) / (p + n)) * 100
