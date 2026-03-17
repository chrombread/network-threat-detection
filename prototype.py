import sklearn
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from ipaddress import ip_address
'''
I started with this tutorial:
www.kaggle.com/code/dansbecker/your-first-machine-learning-model/tutorial
'''

def preprocess(df: pd.DataFrame):
    '''
    scikit-learn needs every column to have numbers. This function makes it so.
    
    Right now the URL and user agent fields can't be used in the model.
    I don't know how to turn them into numbers in a way that preserves their
    meaning.
    '''
    df['timestamp'] = [pd.to_datetime(ts).timestamp() for ts in df['timestamp']]
    df['src_ip'] = df['src_ip'].apply(
        lambda x: int(ip_address(x))
    )
    df['dst_ip'] = df['dst_ip'].apply(
        lambda x: int(ip_address(x))
    )
    protocols = {
        'ICMP': 0,
        'TCP': 1,
        'UDP': 2
    }
    df['protocol'] = df['protocol'].map(protocols)
    df['is_internal_traffic'] = df['is_internal_traffic'].apply(
        lambda x: int(x)
    )
    attack_types = {
        'benign': 0,
        'xss': 1,
        'exploit-attempt': 2,
        'sql-injection': 3,
        'c2': 4,
        'brute-force': 5,
        'ddos': 6,
        'command-injection': 7,
        'credential-stuffing': 8,
        'port-scan': 9
    }
    df['attack_type'] = df['attack_type'].map(attack_types)
    return df


# Get DataFrame from CSV
data = preprocess(pd.read_csv('traffic.csv'))

# Select Prediction Target (exactly what it sounds like)
y = data.attack_type

# Select 'features' (columns used to predict target)
features = ['timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'bytes_sent', 'bytes_received', 'is_internal_traffic']

# By convention, features data is called X
X = data[features]

# Create model
model = sklearn.linear_model.LogisticRegression(max_iter=1000)

# Fit model
model.fit(X, y)

# Predict attack types
y_p = model.predict(X)

# There are other metrics we'll end up using like confusion matrices, recall,
# F1 score.
print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y, y_p))
