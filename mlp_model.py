#phase-3 feature extraction model
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


num_df2=pd.read_csv("export_csv.csv")

X=num_df2[['URL', 'EMAIL', 'PHONE', 'special', 'symbol', 'misspelled','common_words']]
y=num_df2['LABEL']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    max_iter=1000, random_state=42)


mlp.fit(X_train,list(y_train.values))

# Make predictions on the test data
y_pred = mlp.predict(X_test)

# Calculate the evalution metrics of the model
def evalution_metrics(test_val, y_pred):
    accuracy = accuracy_score(test_val, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    precision = precision_score(test_val, y_pred)
    print('Precision: %f' % precision)
    recall = recall_score(test_val, y_pred)
    print('Recall: %f' % recall)
    f1 = f1_score(test_val, y_pred)
    print('F1_score: %f' % f1)
    return [accuracy, precision, recall, f1]

eval_result = evalution_metrics(list(y_test.values), y_pred)
print(eval_result)

CM = confusion_matrix(list(y_test.values), y_pred)
print(CM)
print(classification_report(list(y_test.values), y_pred))


joblib.dump(mlp, 'mlp_model.pkl')