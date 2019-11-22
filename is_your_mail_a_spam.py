import pickle
import sys


def file_get_contents(filename):
    with open(filename) as f:
        return f.read()

# load the model and the vector from disk
model_filename = 'finalized_model.sav'
loaded_model = pickle.load(open(model_filename, 'rb'))
vector_filename = 'vector.pickel'
loaded_vector = pickle.load(open(vector_filename, 'rb'))

#make a classification of the mail content
X_test = file_get_contents(sys.argv[1])
features_test = loaded_vector.transform([X_test])
print(loaded_model.predict(features_test))