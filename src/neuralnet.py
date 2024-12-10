from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# features_df = parsing.main()
features_df = pd.read_csv("data/features.csv")
features_df.fillna(0, inplace=True)
features_df = features_df.drop('Unnamed: 0', axis=1)
# Train/test split
X = features_df.drop('label', axis=1)
y = features_df['label']
features_imp = features_df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=1e-4,
                    hidden_layer_sizes=(13, 13), random_state=1)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
