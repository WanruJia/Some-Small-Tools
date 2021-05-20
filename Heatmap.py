import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt

diff_data = pd.read_excel ("C:/DalMasterCourses/Medical Image Analysis/Xray Grades/program for X-ray images/20201212/Gradient.xlsx", header=None)

ax = sns.heatmap(diff_data)
plt.show()
