import datetime
import graphviz
import numpy as np
import pandas as pd
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split


def handle_data(data):
    del data['id']

    # today = datetime.datetime.now()
    # today.strftime('%Y-%m-%d')
    today = '2019-03-02'
    today = datetime.datetime.strptime(today, '%Y-%m-%d')

    data['deadline'] = pd.to_datetime(data['deadline'])
    data['deadline'] = data['deadline'] - today
    data['deadline'] = data['deadline'].astype('int')//(24*2600*10**9)
    data['model_number'] = -data['model_number']

    min_max_scaler = preprocessing.MinMaxScaler()
    if 'importance' in data:
        feature_name = data.drop('importance', axis=1).columns
        
        new_data = pd.DataFrame(min_max_scaler.fit_transform(
            data.drop('importance', axis=1)), columns=feature_name)
        new_data['importance'] = data['importance']
        return new_data
    else:
        feature_name = data.columns
        new_data = pd.DataFrame(min_max_scaler.fit_transform(
            data), columns=feature_name)
        return new_data


# read file
datafile = "/media/congee/1CD870D4D870AE20/Documents/homework_4th/APS/code/OrderHandle/order_train.csv"
data = pd.read_csv(datafile, header=0, sep=',')

# handle the data
train_data = handle_data(data)

# train the classifier
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    train_data.drop('importance', axis=1), train_data['importance'], test_size=0.3, random_state=30)
feature_name = train_data.drop('importance', axis=1).columns
importance_name = ['significant', 'important',
                   'noteworthy', 'ordinary', 'soft']

clf = tree.DecisionTreeClassifier(criterion="entropy",
                                  # random_state=30,
                                  # splitter="random",
                                  # max_depth=3,
                                  # min_samples_leaf=10,
                                  # in_samples_split=10,
                                  )
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
print("score: ", score)

# draw the decision tree and list the ratio of feature_importance
dot_data = tree.export_graphviz(
    clf, feature_names=feature_name, class_names=importance_name, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph

clf.feature_importances_
print([*zip(feature_name, clf.feature_importances_)])


# read order_data to be predicted
datafile = "/media/congee/1CD870D4D870AE20/Documents/homework_4th/APS/code/OrderHandle/order.csv"
order_data = pd.read_csv(datafile, header=0, sep=',')

# handle the data
ans = pd.DataFrame.copy(order_data)
predict_data = handle_data(order_data)

# predict the importance
ans['importance'] = clf.predict(predict_data)
ans.to_csv(datafile, sep=',')