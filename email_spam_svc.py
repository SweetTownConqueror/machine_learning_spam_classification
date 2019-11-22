import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#Load dataset
dataframe = pd.read_csv("spam.csv")
#print(dataframe.head())


X = dataframe["EmailText"]
y = dataframe["Label"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)
#print(X_train.head())

#extract features
cv = CountVectorizer()
features = cv.fit_transform(X_train)
#print(cv.get_feature_names())
#print(features.toarray())
#print(features)

#use a model
model = svm.SVC()
model.fit(features, y_train)

#test accuracy
features_test = cv.transform(X_test)

##if you want to test on one value
# k = X_test.keys()[0]
# tst = cv.transform([X_test[k]])
# print(tst)
# print(model.predict(tst))
# print(y_test[k])

print("Accuracy of the model is:",model.score(features_test, y_test))