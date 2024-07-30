
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
import numpy as np
import os
import joblib   
import argparse
import pathlib
from io import StringIO
import boto3

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()


    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--random-state', type=int, default=0)

    #data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")

    args, _ = parser.parse_known_args()


    print("[INFO] Reading data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)

    print("building training and testing datasets")
    print()
    x_train = train_df[features]
    y_train = train_df[label]
    x_test = test_df[features]
    y_test = test_df[label]

    print("Column order:")
    print(features)
    print()

    print("Label column is:",label)
    print()
    print("Data Shape: ")
    print()
    print("--SHAPE OF TRAINING DATA--")
    print("X_train:", x_train.shape)
    print("Y_train:", y_train.shape)
    print()
    print("--SHAPE OF TESTING DATA--")
    print("X_test:", x_test.shape)
    print("Y_test:", y_test.shape)
    print()

    print("Training Random Forest Model")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(x_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model saved at:", model_path)
    print()

    y_pred = model.predict(x_test)
    testaccuracy = accuracy_score(y_test, y_pred)
    testreport = classification_report(y_test, y_pred)

    print()
    print("--TESTING REPORT--")
    print()
    print("Total Rows are: ", x_test.shape[0])
    print("Accuracy: ", testaccuracy)
    print('Testing report: ', testreport)
