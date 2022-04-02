import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn import metrics
from mlxtend.evaluate import bias_variance_decomp

#read the data
data_set = pd.read_csv("COMP4388-Dataset1.csv")
#drop ID feature becouse its dummy sequence feature has no correlation with target class
data_set=data_set.drop(columns=['ID'])
#chooseng the input features as x and target feature as y
x=data_set[['x','y']]
y=data_set['label']
#split the data 20% test and 80% train
#remark: the splitting will be randomly.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#generate model
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
#set predicted values of test data
y_pred = clf.predict(x_test)
#get confusion matrix of test
cm = metrics.confusion_matrix(y_test, y_pred)
print("------------------------------------confusion_matrix------------------------------------")
print(f'*DecisionTree_confusion_matrix:\n{cm}')
print("---------------------------Accuracy,Precision,Recall,F-score---------------------------")
TP = cm[0][0];FN = cm[0][1];FP = cm[1][0];TN = cm[1][1]
print("Accuracy: ", (TP+TN)/(TP+FP+FN+TN) )# or we can use this function (metrics.accuracy_score(y_pred,y_test,))
print("Precision: ", TP/(TP+FP))# or we can use built in function (metrics.precision_score() )
print("Recall: ", TP/(TP+FN))# or we can use built in function (metrics.recall_score() )
print("F-score: ", (2*TP/(2*TP+FP+FN) ) )
print("------------------------------------Bias,Variance,MSE-----------------------------------")
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf, x_train.to_numpy(), y_train.to_numpy(),x_test.to_numpy(),
                                      y_test.to_numpy(), loss='0-1_loss', random_seed=1)
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)

#show the tree by plot
plot_tree(clf)
plt.show()
