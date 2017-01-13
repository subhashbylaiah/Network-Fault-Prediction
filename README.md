**Predicting Network Fault Severity**

This project is build classifier models on the
Telstras Network Disruption data to predict fault severity of
Telstras network disruptions. Telstra, Australias largest telecommunications
network made this data available via Kaggle.

The class field has categories:
• 0 means that there is no fault
• 1 means that there are faults

Here we try multiple classification algorithms XGBoost, RandomForest,
DecisionTree, Adaboost and SVM and compare their performance
on this dataset.
We are interested in evaluating the algorithms and analyzing practical
difficulties and limitations in processing high-dimension, high-volume datasets
and identifying techniques to help mitigate issues in Applied machine learning.


**About the data**
More details about the competition and the data can be obtained here
https://www.kaggle.com/c/telstra-recruiting-network

**Data Preprocessing**
We started with some data cleaning to remove data points that had missing features, and then merged the data across
multiple files using the common identifier attribute.
The final processed data matrix is stored as feature_Extracted_train_data.csv and feature_Extracted_test_data.csv

**Performance**

We started with a simple decision tree as our baseline and compated the performance of various classification approaches.

**SVM**: We tested with rbf and linear kernels. RBF kernel was able to quickly converge, it provided a slight
improvement over the simple decision tree. linear kernel was not able to converge in reasonable time.
This shows that the data is not linearly separable. Choosing the right kernel for SVMs is the most difficult part and
its problem specific.

**Adaboost**: We tested adaboost with varying depths and iterations, we achieved the best performance with tree
depth of 1 and 10 iterations. We observed that adaboost increased its performance significantly when the depth of
the tree was decreased and the number of iterations were increased.

**RandomForest**: We tested random forest, with varying number of trees and observed better performance as the number of
trees increased. The multitude of trees help reduce variance. Every tree tries to overfit a part of the training data
and when combined they normalize the error.

**XGBoost**: We have observed very good performance with XGBoost ensemble models which is due to advantages of an
ensemble method along with the robustness of the algorithm, which includes a vast array of tunable parameters to
create the best set of ensembles.