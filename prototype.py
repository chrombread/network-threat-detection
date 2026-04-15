import sklearn
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy as np
import math
from ipaddress import ip_address
import matplotlib.pyplot as plt

def get_time(X):
    return pd.to_datetime(X.iloc[:,0]).apply(lambda x: math.sin(
                              math.tau / 24 * (3600 * x.hour + 60 * x.minute + x.second)
                          )).values.reshape(-1, 1)

# Define steps used for preprocessing
preprocessor = sklearn.compose.ColumnTransformer(
    transformers=[
        ('timestamp', preprocessing.FunctionTransformer(get_time), ['timestamp']),
        ('port', 'passthrough', ['src_port', 'dst_port']),
        ('protocol', preprocessing.OneHotEncoder(drop=None), ['protocol']),
        ('bytes_sent', 'passthrough', ['bytes_sent', 'bytes_received']),
        ('internal', 'passthrough', ['is_internal_traffic']),
    ]
)

# Define the model
model = sklearn.ensemble.RandomForestClassifier(class_weight='balanced', random_state=0)

# Define data pipeline (steps applied to dataset)
pipeline = sklearn.pipeline.make_pipeline(preprocessor, model)

# Read data
data = pd.read_csv('traffic.csv')
X = data[data.columns.drop('attack_type')]
y = data.attack_type

# Run model
pipeline.fit(X, y)
y_p = pipeline.predict(X)

# Evaluate performance
cm = metrics.confusion_matrix(y, y_p)
confusion_matrix = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                   display_labels=pipeline.classes_)
confusion_matrix.plot()
plt.show()