# import parsing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

def imp_features(model, features_df):
    feature_names = features_df.columns
    # Built-in feature importance (Gini Importance) to determine the most important features for our model
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
    print(feature_imp_df)


# features_df = parsing.main()
features_df = pd.read_csv("data/features.csv")

# Train/test split
X = features_df.drop('label', axis=1)
y = features_df['label']
features_imp = features_df.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

imp_features(model,features_imp)

