from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pandas as pd

# features_df = parsing.main()
features_df = pd.read_csv("data/features.csv")
features_df.fillna(0, inplace=True)
features_df = features_df.drop('Unnamed: 0', axis=1)


# Train/test split
X = features_df.drop('label', axis=1)
y = features_df['label']

# Normalizing the features leads to slightly worse performance
# X=(X-X.mean())/X.std()

features_imp = features_df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def find_best_model_with_params(params = None):
    mlp = MLPClassifier(max_iter=1000, random_state=42)
    # Parameter space to choose from and try out all possible combinations
    parameter_space = {
        'hidden_layer_sizes': [(200, 200, 200), (100, 100, 100, 100), (10, 10, 10, 10, 10, 10, 10), (70, 60, 50, 40), (100, 300, 100), (200, 250, 40)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    if params == None:
        params = parameter_space
    # Run GridSearch
    # n_jobs = -1 uses all CPU cores
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
    clf.fit(X_train, y_train)

    # Show the results
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


    y_pred = clf.predict(X_test)

    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_test, y_pred))
    return clf

def safe_model(model, path):

    # Create file for saving trained model as a onnx file
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

    # Convert the model
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Safe the trained model to the file
    with open("model_neuralnet.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

def train_neuralnet():

    # Best layer structure until now [120, 40, 40, 40]
    params = { 'hidden_layer_sizes' : [120, 40,40,40],
        'activation' : 'relu', 'solver' : 'adam',
        'alpha' : 0.0001, 'batch_size' : 'auto',
        'learning_rate':'constant',
        'learning_rate_init' : 0.001,
        'power_t':0.5, 'max_iter' : 2000,
        'shuffle' : True,
        'random_state' : 42, 'tol' : 0.0001,
        'verbose':False, 'warm_start':False, 'momentum':0.9,
        'nesterovs_momentum' : True, 'beta_1':0.9,'beta_2':0.999,
        'epsilon':1e-08,
        'n_iter_no_change' : 10, 'max_fun':15000}

    clf = MLPClassifier(**params)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    # Use k fold cross validation
    model = MLPClassifier(**params)
    scores = cross_val_score(model, X, y, cv=10)
    print(scores)
    print("%0.2f accuracy for non balanced out dataset with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    return model

    