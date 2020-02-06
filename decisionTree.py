from sklearn import tree
import pandas as pd
import pydotplus
from IPython.display import Image
from graphviz import Digraph
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
#print(df)
df_test = pd.read_csv('test.csv')
print(df_test)


#df.fillna(df.mean())
#df2 = df.dropna(subset=['Age'])
df2 = df.fillna(df.median())
#df_test2 = df_test.dropna(subset=['Age'])
df_test2 = df_test.fillna(df_test.median())
#print(df)
df2['Sex'] = df2['Sex'].str.replace('female', '1')
df2['Sex'] = df2['Sex'].str.replace('male', '0')
df2['Sex'].astype(int)
target = df2['Survived']
df_test2['Sex'] = df_test2['Sex'].str.replace('female', '1')
df_test2['Sex'] = df_test2['Sex'].str.replace('male', '0')
df_test2['Sex'].astype(int)
#print(target)
data = df2[['Pclass', 'Sex', 'Age', 'SibSp']]
test = df_test2[['Pclass', 'Sex', 'Age', 'SibSp']]
#print(data)
#print(df2.dtypes)


clf = tree.DecisionTreeClassifier(max_depth=3)  # limit depth of tree
clf.fit(data, target)

tree.plot_tree(clf,
              feature_names=['Pclass', 'Sex', 'Age', 'SibSp'],
              class_names=['Survived', 'not Survived'],
              filled=True, rounded=True, proportion=True)
plt.savefig("tree1.4.png")

#print(clf.predict(test))

df_test2['Survived'] = clf.predict(test)

df_test2.to_csv('predicted.csv', columns=['PassengerId', 'Survived'])