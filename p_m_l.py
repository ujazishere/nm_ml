# Python Machine Learning

import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

desk = r'C:\Users\ujasv\OneDrive\Desktop\codes\test_data.csv'

# Converts raw digits column to a clear float only column using regex
# df[df.columns[-1]] = df[df.columns[-1]].replace('\D','',regex=True).astype(float)

df = pd.read_csv(desk)

# Getting unique nationalities and associating numbers to each items.
nat_uniq = pd.unique(df['Nationality'])
nat_nums_dict = dict(zip(nat_uniq, range(len(nat_uniq))))

# Replacing Nationality with the associated number
df['Nationality'] = df['Nationality'].map(nat_nums_dict)
# print(df)

# Getting unique of 'Go' items then making dict of items and associating number
Go_uniq = pd.unique(df['Go'])
Go_uniq_dict = dict(zip(Go_uniq, range(len(Go_uniq))))

# Replacing those 'Go' items with associated numbers
df['Go'] = df['Go'].map(Go_uniq_dict)

# print(df)

feature = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[feature]
y = df['Go']

print(df)

dtree = DecisionTreeClassifier()

dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=feature)
# plt.show()

#  arguments(comedians age, his years of experience and comedy ranking
print(dtree.predict([[40, 10, 7, 1]]))

actual = np.random.binomial(1, 0.9, size=1000)
predicted = np.random.binomial(1, 0.9, size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels= [False, True])

cm_display.plot()
plt.show()

accuracy = metrics.accuracy_score(actual, predicted)
precision = metrics.precision_score(actual, predicted)
sensitivity_recall = metrics.recall_score(actual, predicted)
specificity = metrics.recall_score(actual, predicted, pos_label=0)
f1_score = metrics.f1_score(actual, predicted)

print(f'accuracy: {accuracy}, precision: {precision}, sensitivity_recall: {sensitivity_recall}, specificity: {specificity}, f1_score: {f1_score}')















