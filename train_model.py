import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

train = pd.read_csv("train_house1.csv", parse_dates=["timestamp"], index_col="timestamp")
test = pd.read_csv("test_house1.csv", parse_dates=["timestamp"], index_col="timestamp")

feature_cols = ["agg_roll_mean_5min", "agg_roll_mean_1min", "agg_roll_mean_30s", "agg_diff", "agg_roll_std_5min"]
label_col = "fridge_on"

X_train = train[feature_cols]
y_train = train[label_col]

X_test = test[feature_cols]
y_test = test[label_col]

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1 score", f1)
print("Confusion Matrix")
print(cm)

model_xgb = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

print("\n--- XGBoost Results ---")
print("Accuracy:", accuracy_xgb)
print("F1 Score:", f1_xgb)
print("Confusion Matrix:")
print(cm_xgb)