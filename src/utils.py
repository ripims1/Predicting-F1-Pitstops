import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Reads data files
# x: input dataframe, y: target value for that data, x_test: the test data without existing associated values
def load_processed_data():
    x = pd.read_csv("../data/train_encoded.csv")
    y = pd.read_csv("../data/target.csv").squeeze()
    x_test = pd.read_csv("../data/test_encoded.csv")

    return x, y, x_test

# Self explanatory
def generate_submission(predictions, output_path,raw_test_path="../data/test.csv", target_column="PitNextLap"):

    raw_test = pd.read_csv(raw_test_path)
    submission = pd.DataFrame({

    "id": raw_test["id"],
    target_column: predictions
    })

    submission.to_csv(output_path, index=False)
    return submission

# Generates a simulated auc value given the y_val and the predictions made
def generate_auc(y_val, predictions, decimals=4):
    auc = roc_auc_score(y_val, predictions)
    print(f"AUC: {auc:.{decimals}f}")

# Generates the probability prediction given x_val and model
def generate_probability_predict(model, x):
    return model.predict_proba(x)[:,1]