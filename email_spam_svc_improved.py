import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import pickle

#Load dataset
dataframe = pd.read_csv("spam.csv")
#print(dataframe.head())


X = dataframe["EmailText"]
y = dataframe["Label"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#extract features
cv = CountVectorizer()
features = cv.fit_transform(X_train)

#tuned_parameters = {'kernel':['linear', 'rbf'], 'gamma':[1e-3, 1e-4], 'C':[1,10,100,1000]}
tuned_parameters = {'kernel':['rbf'], 'gamma':[1e-4], 'C':[1000]}

#use a model
model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(features, y_train)

#test accuracy
features_test = cv.transform(X_test)
##if you want to test on one value
# k = X_test.keys()[0]
# tst = cv.transform([X_test[k]])
# print(tst)
# print(model.predict(tst))
# print(y_test[k])

#print(model.best_params_)
print("Accuracy of the model is:",model.score(features_test, y_test))
##To Export the model:
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.dump(cv, open("vector.pickel", "wb"))