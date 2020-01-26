from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from graphviz import Digraph
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv', sep='\t')
print(df)
#df.fillna(df.mean())
df2 = df.dropna(subset=['Age'])
#print(df)
df2['Sex'] = df2['Sex'].str.replace('female', '1')
df2['Sex'] = df2['Sex'].str.replace('male', '0')
df2['Sex'].astype(int)
target = df2['Survived']
#print(target)
data = df2[['Pclass', 'Sex', 'Age', 'SibSp']]
#print(data)
#print(df2.dtypes)


clf = tree.DecisionTreeClassifier(max_depth=3)  # limit depth of tree
clf.fit(data, target)

"""
dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=['Pclass', 'Sex', 'Age', 'SibSp'],
    class_names=['Survived', 'not Survived'],
    filled=True,
    proportion=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
"""
tree.plot_tree(clf,
              feature_names=['Pclass', 'Sex', 'Age', 'SibSp'],
              class_names=['Survived', 'not Survived'],
              filled=True, rounded=True, proportion=True)
plt.savefig("tree1.4.png")