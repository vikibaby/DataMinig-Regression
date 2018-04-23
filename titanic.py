import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")
y = df.Survived.astype(bool)

#Pre-processing the data
df['Sex'].replace(['male','female'],[0,1],inplace = True)
df.Embarked.fillna(-1,inplace = True)
df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace = True)
df.Age.fillna(-1,inplace = True)
x = df[['Pclass','SibSp','Fare','Age','Parch','Sex','Embarked']]
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size = 0.40, random_state=0)

#analysis Sex and PClass
ct=pd.crosstab([df.Pclass, df.Sex],y)
ct.plot(kind='bar', stacked=False, color=['red','blue'], grid=False,figsize=(20,16))
plt.savefig('titanic_Sex_Class.png')
plt.show()

#analysis Age
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(df['Age'], bins = 10, range = (max(0,df['Age'].min()),df['Age'].max()))
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Count of Passengers')
plt.savefig('titanic_Age_distribution.png')
plt.show()
plt.close()



# DecisionTreeClassifier
tree_clf = tree.DecisionTreeClassifier(max_depth = 6,random_state = 99)
tree_clf = tree_clf.fit(x_train, y_train)
tree_pre = tree_clf.predict(x_test)
print confusion_matrix(y_test,tree_pre)
#An alternative options for the confusion matrix
print pd.crosstab(y_test,tree_pre, rownames=['True'], colnames=['Predicted'], margins=True)

# Logical Regression
from sklearn.linear_model import LogisticRegression
LR_clf = LogisticRegression()
LR_clf = LR_clf.fit(x_train, y_train)
LR_pre = LR_clf.predict(x_test)
print pd.crosstab(y_test,LR_pre, rownames=['True'], colnames=['Predicted'], margins=True)

#ROC
from sklearn.metrics import roc_curve, auc
LR_false_positive_rate, LR_true_positive_rate, LR_thresholds = roc_curve(y_test,LR_pre)
LR_roc_auc=auc(LR_false_positive_rate, LR_true_positive_rate)
tree_false_positive_rate, tree_true_positive_rate, tree_thresholds = roc_curve(y_test,tree_pre)
tree_roc_auc=auc(tree_false_positive_rate, tree_true_positive_rate)
plt.title('ROC-comparing Decision Tree to LR')
plt.plot(LR_false_positive_rate, LR_true_positive_rate, 'b-', label='Logical Regression')
plt.plot(tree_false_positive_rate, tree_true_positive_rate, 'g', label='DecisionTree')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('titanic_ROC.png')
plt.show()
plt.close()
