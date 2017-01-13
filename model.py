
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from matplotlib.pylab import rcParams
from sklearn import metrics, model_selection
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import TreeClassifier as tc
from math import log, exp
from collections import Counter
import time
from multiprocessing import Pool
from itertools import repeat


rcParams['figure.figsize'] = 20, 20
result_datadir = 'data'


def printFeatureImportance(xgb1):
    # Print feature importance
    feature_fscores = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
    # print "total fetures: ",len(feature_fscores)
    plt.gcf().subplots_adjust(bottom=0.25)
    feature_fscores.iloc[:10].plot(color='OrangeRed', alpha=0.6, kind='bar', title='Top 10 Features')
    plt.ylabel('Feature Importance Score')
    plt.show(block=False)
    plt.savefig('myfig')


# Calculate Accuracy, log_loss
def calculatePerformance(actual_values, test_predictions, test_pred_prob, scoring_metric):

    actual_values_list = list(actual_values)
    classification_Accuracy = metrics.accuracy_score(actual_values, test_predictions)
    if len(test_pred_prob) > 0:
        roc_auc_score = metrics.roc_auc_score(actual_values, test_pred_prob[:, 1])
    else:
        roc_auc_score = metrics.roc_auc_score(actual_values, test_predictions)

    true_positives = 0; false_positives = 0; true_negatives = 0; false_negatives = 0

    for i in range(len(test_predictions)):
        if actual_values_list[i] == test_predictions[i] and actual_values_list[i] == 1:
            true_positives += 1
        if test_predictions[i] == 1 and actual_values_list[i] != test_predictions[i]:
            false_positives += 1
        if actual_values_list[i] == test_predictions[i] and actual_values_list[i] == 0:
            true_negatives += 1
        if test_predictions[i] == 0 and actual_values_list[i] != test_predictions[i]:
            false_negatives += 1


    tpr = float(true_positives) / (true_positives + false_negatives)
    fpr = float(false_positives) / (false_positives + true_negatives)

    print "\n############-Model Report-################"
    print "Confusion Matrix:"
    print pd.crosstab(actual_values, test_predictions)
    print 'Note: rows - actual; col - predicted'
    print "\n###########-Classification Report-########"
    print metrics.classification_report(y_true=list(actual_values), y_pred=test_predictions)
    print "Accuracy  : %s" % "{0:.3%}".format(classification_Accuracy)
    print "ROC Score:" + str(roc_auc_score)
    print "True Positive Rate: %f, False Positive Rate: %f" % (tpr, fpr)


def learn_xgboost(train, predictors, class_target):

    xgb_train = xgb.DMatrix(train[predictors].values, label=train[class_target].values)
    xgb1 = XGBClassifier()

    # Tunning parameters
    cv_folds = 10
    early_stopping_rounds = 20
    show_progress = True

    params = {'reg_alpha': 0, 'colsample_bytree': 0.6, 'silent': 1, 'colsample_bylevel': 1,
              'scale_pos_weight': 1, 'learning_rate': 0.1, 'missing': -1, 'max_delta_step': 0,
              'nthread': 4, 'base_score': 0.5, 'n_estimators': 500, 'subsample': 0.7,
              'reg_lambda': 1, 'seed': 0, 'min_child_weight': 1, 'objective': 'multi:softprob',
              'max_depth': 6, 'gamma': 0, 'num_class': 2}
    # Cross-Validation
    cvresult = xgb.cv(params, xgb_train, num_boost_round=params['n_estimators'], nfold=cv_folds,
                      metrics=['mlogloss'], early_stopping_rounds=early_stopping_rounds)
    xgb1.set_params(n_estimators=cvresult.shape[0])
    xgb1.fit(train[predictors], train[class_target], eval_metric=['mlogloss'])
    return xgb1

def predict_xgboost(model, test, predictors):
    # Get test predictions:
    test_predictions = model.predict(test[predictors])
    test_pred_prob = model.predict_proba(test[predictors])
    return (test_predictions, test_pred_prob)


def learn_svm(train, predictors, class_target):

    svm = SVC(probability=True, kernel='rbf')
    # Cross-Validation
    model = svm.fit(train[predictors], train[class_target])
    return model

def predict_svm(model, test, predictors):
    # Get test predictions:
    test_predictions = model.predict(test[predictors])
    test_pred_prob = model.predict_proba(test[predictors])
    return (test_predictions, test_pred_prob)



def learn_randomforest(train, predictors, class_target):
    params = {'n_estimators': 500, 'random_state': 0, 'max_depth': 6}
    rf = RandomForestClassifier(n_estimators=500, max_depth=6)
    rf.fit(train[predictors], train[class_target])
    return rf

def predict_randomforest(model, test, predictors):
    # Get test predictions:
    test_predictions = model.predict(test[predictors])
    test_pred_prob = model.predict_proba(test[predictors])
    return (test_predictions, test_pred_prob)


def main():
    train = pd.read_csv(os.path.join(result_datadir, 'processed_train1.csv'))
    test = pd.read_csv(os.path.join(result_datadir, 'processed_test1.csv'))

    # train = train.sample(1000)
    # test = test.sample(100)

    class_target = 'fault_severity'
    IDcol = 'id'
    predictors = [x for x in train.columns if x not in [class_target, IDcol]]

    print "###################################################################################################################"
    print "Learning XGBoost Model"
    c = time.time()
    xgb1 = learn_xgboost(train, predictors, class_target)
    (test_predictions, test_pred_prob) = predict_xgboost(xgb1, test, predictors)
    calculatePerformance(test[class_target], test_predictions, test_pred_prob, 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"

    feature_fscores = pd.Series(xgb1.booster().get_fscore()).sort_values(ascending=False)
    top_features = list(feature_fscores[:10].index)
    printFeatureImportance(xgb1)

    print "###################################################################################################################"
    print "Learning Random Forest Model"
    c = time.time()
    rf = learn_randomforest(train, predictors, class_target)
    (test_predictions, test_pred_prob) = predict_xgboost(rf, test, predictors)
    calculatePerformance(test[class_target], test_predictions, test_pred_prob, 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"


    print "###################################################################################################################"
    print "Learning SVM Model"
    c = time.time()
    svm = learn_svm(train, top_features, class_target)
    (test_predictions, test_pred_prob) = predict_svm(svm, test, top_features)
    calculatePerformance(test[class_target], test_predictions, test_pred_prob, 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"


    c = time.time()
    print "###################################################################################################################"
    print "Simple Decision Tree (All Features), Depth: 5"
    trainset = train.as_matrix(columns=list(feature_fscores[:].index) + [class_target])
    testset = test.as_matrix(columns=list(feature_fscores[:].index) + [class_target])

    (boost_trees, voting_weights) = learn_boosted(5, 1, (trainset, testset))
    (pn, predictions) = tc.test(boost_trees, testset, voting_weights)
    print(tc.confusionMatrix(pn))

    calculatePerformance(test[class_target], pd.Series(predictions), [], 'log_loss')
    print "Time taken: ", time.time()-c

    print "###################################################################################################################"
    print "Ada Boosted Decision Tree (Top selected Features), Depth: 10, Numbed Trees:1"
    c = time.time()
    print "############ Top selected features ################"
    trainset = train.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    testset = test.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    (boost_trees, voting_weights) = learn_boosted(10, 1, (trainset, testset))
    (pn, predictions) = tc.test(boost_trees, testset, voting_weights)
    print(tc.confusionMatrix(pn))
    calculatePerformance(test[class_target], pd.Series(predictions), [], 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"


    print "###################################################################################################################"
    print "Learning Adaboost Trees (Top selected Features), Depth: 5, NumTrees: 5"
    c = time.time()
    print "############ Top selected features ################"
    trainset = train.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    testset = test.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    (boost_trees, voting_weights) = learn_boosted(5, 5, (trainset, testset))
    (pn, predictions) = tc.test(boost_trees, testset, voting_weights)
    print(tc.confusionMatrix(pn))
    calculatePerformance(test[class_target], pd.Series(predictions), [], 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"

    print "###################################################################################################################"
    print "Learning Adaboost Trees (Top selected Features), Depth:1, NumTrees: 10"
    c = time.time()
    print "############ Top selected features ################"
    trainset = train.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    testset = test.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    (boost_trees, voting_weights) = learn_boosted(1, 10, (trainset, testset))
    (pn, predictions) = tc.test(boost_trees, testset, voting_weights)
    print(tc.confusionMatrix(pn))
    calculatePerformance(test[class_target], pd.Series(predictions), [], 'log_loss')
    print "Time taken: ", time.time()-c
    print "###################################################################################################################"


    # print "###################################################################################################################"
    # print "Learning Bagged Ensemble Trees (Top selected Features), Depth:1, NumTrees: 10"
    # c = time.time()
    # print "############ Top selected features ################"
    # trainset = train.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    # testset = test.as_matrix(columns=list(feature_fscores[:10].index) + [class_target])
    # bag_trees = learn_bagged(3, 10, (trainset, testset))
    # (pn, predictions) = tc.test(bag_trees, testset)
    # print(tc.confusionMatrix(pn))
    # calculatePerformance(test[class_target], pd.Series(predictions), [], 'log_loss')
    # print "Time taken: ", time.time()-c
    # print "###################################################################################################################"




    print "done"


'''
Function: learn_boosted(tdepth, numtrees, datapath)
tdepth: (Integer) depths to which to grow the decision trees
numtrees: (Integer) the number of boosted trees to learn
datapath: (String) the location in memory where the data set is stored

This function wil manage coordinating the learning of the boosted ensemble.

Nothing is returned, but the accuracy of the learned ensemble model is printed
to the screen.
'''


def learn_boosted(tdepth, numtrees, dataset):
    train_data, test_data = dataset

    boost_trees = []
    voting_weights = []

    labels = train_data[:, [-1]]
    N = labels.shape[0]

    # initialize the data weights uniformly
    data_weights = (1.0 / N) * np.ones(labels.shape)

    for i in range(numtrees):
        # print '======================= Iteration %d =============================' % i

        tree = tc.classifytree(train_data, tdepth, data_weights)
        # make predictions using the learnt tree
        predictions = np.asarray([tc.predict(tree, row) for row in train_data])
        predictions = predictions.reshape(len(predictions), 1)

        # identify correct and wrong predictions
        correct_predictions = predictions == labels
        wrong_predictions = predictions != labels

        # calculate error and apply laplace correction
        error = (float(np.sum(np.multiply(wrong_predictions, data_weights), axis=0)) + 1) / (
        float(np.sum(data_weights, axis=0)) + 2)

        # print 'Error %f ' % error


        voting_weight = 0.5 * log((1 - error) / error)

        weight_update_factor = np.apply_along_axis(lambda x: exp(voting_weight
                                                                 ) if x else exp(-voting_weight), 1,
                                                   correct_predictions)
        weight_update_factor = weight_update_factor.reshape((len(weight_update_factor), 1))

        # update and normalize the dataweights
        data_weights = np.multiply(data_weights, weight_update_factor)
        data_weights = (1.0 / np.sum(data_weights)) * data_weights

        boost_trees.append(tree)
        voting_weights.append(voting_weight)

    max_class = maxclass_count(train_data[:, [-1]])
    # print "Voting Weights: %s" % str(voting_weights)
    pn = tc.test(boost_trees, test_data, voting_weights)
    # print "Adaboost with tree depth {} and num of iterations {} ".format(str(tdepth), str(numtrees))
    # print(tc.confusionMatrix(pn))
    return (boost_trees,voting_weights)


def learn_bagged(tdepth, numbags, dataset):
    train_data, test_data = (dataset)

    bag_trees = []
    sample_arrays = []

    row_len = len(train_data)

    for i in range(numbags):
        # Selecting n random indices from the in the range of the row len
        rand_idx = np.random.choice(row_len, row_len, replace=True)

        # print "Bag", i, "is", float(len(set(rand_idx)))/len(rand_idx) , "unique."

        # Creating a new train set using the random indices
        sample_arrays.append(np.vstack((train_data[idx] for idx in rand_idx)))

    # using subprocesses to parallelize building trees
    pool = Pool()
    bag_trees = pool.map(multi_run_wrapper, [(a, b) for a, b in zip(sample_arrays, repeat(tdepth))])
    pool.close()
    pool.join()

    # test the dataset using the tree and test data
    return bag_trees


def multi_run_wrapper(args):
   return tc.classifytree(*args)

def maxclass_count(data_array):
    max_class, class_count = Counter(data_array[:, -1]).most_common(1)[
        0]  # Function to return the class with the maximum count of the given data array
    return max_class

