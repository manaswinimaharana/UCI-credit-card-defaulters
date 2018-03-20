#import statsmodels.formula.api as sm


df = pd.read_csv("resources/data/UCI_Credit_Card.csv.zip")

# summarize the data
print(df.describe())
  

# take a look at the standard deviation of each column
print(df.std())


# frequency table cutting presitge and whether or not someone was admitted
#print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
# prestige   1   2   3   4
# admit                   
# 0         28  97  93  55
# 1         33  54  28  12

# plot all of the columns
#df.hist()


####sklearn 


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('resources/data/UCI_Credit_Card.csv.zip', index_col='ID')
df.head()
df.dtypes
df.shape
df.isnull().sum()
# Categorical variables description
df[['SEX', 'EDUCATION', 'MARRIAGE']].describe()


# rename column for easy reference

df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
df.columns=df.columns.str.lower()

print(df.columns)


# check the benchmark

float(df.default.sum())/len(df.default)

# by looking at the column names, there might be multicollinearity issues here,
# so check the correlation matrix to confirm

df.corr()

# plot columns with similar names to check the correlation

sns.pairplot(df, vars=df.columns[11:17], kind='scatter')
sns.pairplot(df, vars=df.columns[17:23])

# manually standardize numeric columns

col_to_norm = ['limit_bal', 'age', 'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
              'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

df[col_to_norm]=df[col_to_norm].apply(lambda x: (x-np.mean(x))/np.std(x))


# create dummies for categorical features.
# add 2 to all the values because OneHotEncoder can only handle non-negative values

col_pay = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
df[col_pay] = df[col_pay].apply(lambda x: x+2)

X = df.iloc[:, 0:23]
y = df.default
enc = OneHotEncoder(categorical_features=[1,2,3,5,6,7,8,9,10])
X = enc.fit_transform(X)

# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=df.default, random_state=1)




# Define function to optimize model based on roc_auc 
# (for unbalanced classes, roc_auc makes more sence since the accuracy score can be fooled by only predicting 0)

def gridsearch(model, params):
    gs = GridSearchCV(model, params, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, y_train)
    print ('Best params: ', gs.best_params_)
    print ('Best auc on training set: ', gs.best_score_)
    print ('Best auc on test set: ', gs.score(X_test, y_test))
    return gs.predict(X_test), gs.decision_function(X_test)
  


# Define function to generate confusion matrix

def plot_confusion(prediction):
    conmat = np.array(confusion_matrix(y_test, prediction, labels=[1,0]))
    confusion = pd.DataFrame(conmat, index=['default', 'not default'], 
                             columns=['predicted default', 'predicted not default'])
    print (confusion)
    


# Define function to plot roc curve

def plot_roc(prob):
    y_score = prob
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    fpr[1], tpr[1], _ = roc_curve(y_test, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])

    plt.figure(figsize=[9,7])
    plt.plot(fpr[1], tpr[1], label='Roc curve (area=%0.2f)' %roc_auc[1], linewidth=4)
    plt.plot([1,0], [1,0], 'k--', linewidth=4)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('false positive rate', fontsize=18)
    plt.ylabel('true positive rate', fontsize=18)
    plt.title('ROC curve for credit default', fontsize=18)
    plt.legend(loc='lower right')
    plt.show()
    
#Logistic Regression with manual feature selection

# logistic regression with grid search

lr = LogisticRegression(solver='liblinear')
lr_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced'], 'penalty': ['l1', 'l2']}

lr_pred, lr_prob = gridsearch(lr, lr_params)

# feature selection with the best model from grid search
# roc_auc same as sgdclassifier

lr = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', class_weight='balanced')
rfecv = RFECV(estimator=lr, scoring='roc_auc')
model = rfecv.fit(X_train, y_train)
lr_pred = model.predict(X_test)
lr_prob = model.decision_function(X_test)
print( 'Test score: ', model.score(X_test, y_test))

# Check the accuracy score, much worse than sgdclassifier

print ('accuracy score on training set: ', cross_val_score(lr, X_train, y_train, n_jobs=-1).mean())
print ('accuracy score on testing set: ', accuracy_score(lr_pred, y_test))

# print confusion matrix
# the model catches more default, but made more mistake in identifying non default at the same time
# So if the bank cares more about identifying people who are going to default (more conservative), 
# this model may be a better choice. If the bank is more aggressive, go for the first one.
plot_confusion(lr_pred)


# plot roc curve

plot_roc(lr_prob)

