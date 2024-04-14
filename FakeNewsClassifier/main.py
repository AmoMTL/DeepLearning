import pandas as pd
from models import LSTMBinaryClassifier

data = pd.read_csv("fake-news-dataset.csv")

model = LSTMBinaryClassifier()

model.Learn(data)
