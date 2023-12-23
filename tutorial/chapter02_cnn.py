



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
# First 5 samples of X:
#  [[ 0.75424625  0.23148074]
#  [-0.75615888  0.15325888]
#  [-0.81539193  0.17328203]
#  [-0.39373073  0.69288277]
#  [ 0.44220765 -0.89672343]]
# First 5 samples of X:
#  [1 1 1 1 0]

print(f"First 5 samples of X:\n {X[:5]}")
print(f"First 5 samples of X:\n {y[:5]}")


# make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1":X[:, 0],
                        "X2":X[:, 1],
                        "label":y})
print(circles.head(10))


# visualize
import matplotlib.pyplot as plt
plt.scatter(x=X[:, 0],
            y=X[:, 1],
            c=y,
            cmap=plt.cm.RdYlBu)
plt.show()