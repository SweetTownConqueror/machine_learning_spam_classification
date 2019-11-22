This is a little python script used to classify mail as spam or not.
I Highly inspired myself from this turorial:
https://www.youtube.com/watch?v=exHwwy9kVcg

First please install theses libraries if you don't already have them:
pip install pandas scikit-learn

Description:
This programm load a dataset conainting 2 columns : 
A colomn with the mail classication as spam or not spam
A column containing the email content
1)Model 85% accuracy: python3  email_spam_svc.py
2)Model 98% accuracy: python3  email_spam_svc_improved.py
3)I exported the model improved thanks to pickle (vector.pickel and finalized_model.sav), and load it in is_your_mail_a_spam.py
If you want to test if one of your mail is a spam or not, put your mail content into mymail.txt
Execute the script: python3  is_your_mail_a_spam.py mymail.txt
It will output "spam" if a spam, "ham" if not



Detailed description:
I use train_test_split to get a bunch of training data and testing data.
Training data will be use to train the programm
Testing data will be use to make prediction and see their accuracy

CountVectorizer.fit_transform(data).toarray() allows to transform the data of words
into an array containing the repetition of these words.
This array can then be used to make prediction using a model.
(Note that CountVectorizer can take many argument which can be usefull in some cases to choose some word that may be more pertinent, but for the moment we don't give it any arguments)

First let's use the simple vector machine model because it's very efficient when there are many features
We fit it with the training data
We test its efficiency by calculating the precision of the classification on the test data on which we know the result (calculating the score)
Without adding any parameters to the SVC function, the accuracy is about 85%

In the improved script, with different parameter, the accuracy is about 98%.
The script automately export the model and the CountVectorizer() so it can be reusable easely
(vector.pickel and finalized_model.sav)




