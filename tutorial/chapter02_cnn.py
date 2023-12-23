



# 1/ make classification data and get it ready

import sklearn
from sklearn.datasets import make_circles

# 
n_samples = 1000
# 
X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(len(X), len(y))

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of X:\n {y[:5]}")


# make dataframe of circle data

import pandas as pd
circles = pd.DataFrame({"X1":X[:, 0],
                        "X2":X[:, 1],
                        "label":y})
print(circles.head(10))

