import pickle
def save_to_pickle(filename, X, y, X_label, y_label):
    dist_pickle_full_set = {}
    dist_pickle_full_set[X_label] = X
    dist_pickle_full_set[y_label] = y
    pickle.dump( dist_pickle_full_set, open( "../data/pickles/"+filename+".p", "wb" ) )

def load_pickle(filename):
    with open(filename + ".p", mode='rb') as f:
        return pickle.load(f)

def save_history_to_pickle(filename, history):
    dist_pickle_full_set = {}
    dist_pickle_full_set[filename] = history
    pickle.dump( dist_pickle_full_set, open( "../data/pickles/"+filename+".p", "wb" ) )
