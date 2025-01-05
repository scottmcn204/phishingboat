# import parsing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def imp_features(model, features_df):
    feature_names = features_df.columns
    # Built-in feature importance (Gini Importance) to determine the most important features for our model
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
    print(feature_imp_df)

def balance_dataset(X, y):
    # Apply oversampling using RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

# features_df = parsing.main()
features_df = pd.read_csv("data/features.csv")
features_df = features_df.loc[:, ~features_df.columns.str.contains('^Unnamed')]

# Train/test split
X = features_df.drop('label', axis=1)
y = features_df['label']
features_imp = features_df.drop('label', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
imp_features(model,features_imp)

# Use k fold cross validation
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=10)
print(scores)
print("%0.2f accuracy for non balanced out dataset with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# # Balancing out the dataset --> artificially sampling the phishing websites to increase their occurences
# X,y = balance_dataset(X,y)

# # Use k fold cross validation
# model = RandomForestClassifier()
# scores = cross_val_score(model, X, y, cv=10)
# print("%0.2f accuracy for balanced dataset with a standard deviation of %0.2f" % (scores.mean(), scores.std()))




initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

# Convert the model
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
