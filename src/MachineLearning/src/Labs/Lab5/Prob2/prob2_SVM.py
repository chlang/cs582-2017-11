import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pcn
from sklearn import svm
import cvxopt
import seaborn as sns;

sns.set(font_scale=1.2)

"""
SVM
"""
data1 = pd.read_csv("prob2_version2.csv")
# , hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s":70})

sns.lmplot('X', 'Y', data=data1, palette='Set1', fit_reg=False, scatter_kws={"s": 70})
model = svm.SVC(kernel='linear')
model.fit(x_training, y_training)

# get separating hyperplane 

model_coef = model.coef_[0]

a = -model_coef[0] / model_coef[1]
xx = np.linspace(-2, 2)
yy = a * xx - (model.intercept_[0]) / model_coef[1]

# plot two supporting hyperplanes
b = model.support_vectors_[0]
yy_down = a * xx + [b[1] - a * b[0]]
b = model.support_vectors_[-1]
yy_up = a * xx + [b[1] - a * b[0]]

sns.lmplot('X', 'Y', data=data1, palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=80, facecolors='green')
plt.show()

# hyperplane
sns.lmplot('X', 'Y', data=data1, palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.show()
