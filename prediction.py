import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def get_prediction(data, model):
  val = model.predict(data)
  return val[0]