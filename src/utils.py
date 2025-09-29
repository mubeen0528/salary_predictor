import pickle
import os

def save_model(model, model_path="model/salary_model.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

def load_model(model_path="model/salary_model.pkl"):
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    return None
