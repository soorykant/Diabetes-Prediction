# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import statsmodels.api as sm
import matplotlib.collections


dataset = pd.read_csv("diabetes.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

labelEnco_y = LE()
y = labelEnco_y.fit_transform(y)


x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.25, random_state = 0)


sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


# Logistic Regression Starts from here.

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)

# Prediction part

pred01 = classifier.predict(x_train)


# Confusion Matrix Part

cm = confusion_matrix(y_train, pred01)
cm

# True Positive = 874
# True Negative = 299
# False Negative = 219
# False Positive = 108

accuracy_tr = (cm[0][0]+cm[1][1])/1500

accuracy_tr

############### End of General Model ##############


# Training more optimised model with stats model package.

x_train_s, x_test_s, y_train_s, y_test_s = tts(x, y, test_size = 0.25, random_state = 0)

# Feature scaling

sc_x_s = StandardScaler()

x_train_s = sc_x_s.fit_transform(x_train_s)

x_test_s = sc_x_s.transform(x_test_s)

# Stats model work

x_train_s = sm.add_constant(x_train_s)
x_test_s = sm.add_constant(x_test_s)

# Main part of the model

classifier02 = sm.Logit(endog = y_train_s, exog = x_train_s).fit()
classifier02.summary()

# Prediction 

pred02 = classifier02.predict(x_test_s)

# Stats model gives probability

pred02 = (pred02 > 0.5).astype(int)

# Confusion matrix for statsmodels

cm02 = confusion_matrix(y_test_s, pred02)

accuracy02 = (cm02[0][0]+cm02[1][1])/500

accuracy02

###### Working on optimal model creation by using backward elimination approach.

classifier02.summary()

aic1 = (2*722.20 + 2*8)

# Removing some veriables and makin the model optimal
x_opt = x_train_s[:,[0,1,2,3,5,6,7,8]]

classifier02_b = sm.Logit(endog = y_train_s, exog = x_opt).fit()

classifier02_b.summary()

aic2 = 2*722.22 + 2*7
aic2

pred03 = classifier02_b.predict(x_test_s[:,[0,1,2,3,5,6,7,8]])

pred03 = (pred03>0.5).astype(int)

cm03 = confusion_matrix(y_test_s,pred03)

cm03

accuracy03 = (cm03[0][0]+cm03[1][1])/500

accuracy03

# (1) When all veriables are included aic1 = 1460.4 and misclassification is 113. 36+77
# (2) After removing 1 veriable indexed at 4, aic2 = 1458.44 and misclassification is 111. 36+75

# Missclassification value is 111 in the optimised model, but in the original normal model it has value of 113 misclassifications.

classifier02_b.summary()

pd.crosstab(y_test_s, pred03, rownames = ["True"], colnames = ["Predicted"], margins = True)

pred_r = classifier02_b.predict(x_train_s[:,[0,1,2,3,5,6,7,8]])

# curve b/w FPR and TPR

fpr, tpr, threshold = roc_curve(y_true = y_train_s, y_score = pred_r, drop_intermediate = True)


# Plot ROC 


plt.figure()

plt.plot(fpr, tpr, lw = 2, color="red")
plt.plot([0,1], [0,1], lw = 2, color="blue")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")

plt.show()


# Area under the ROC curve.


roc_auc_score(y_true = y_train_s, y_score = pred_r)

# 0.84


# We'll change the threshold value in order to better model 

# TPR must be high
# FPR must be less


# Reducing the threshold

pred04 = classifier02_b.predict(x_test_s[:,[0,1,2,3,5,6,7,8]])

pred04 = (pred04 > 0.35).astype(int)

cm04 = confusion_matrix(y_test_s,pred04)

cm04

accuracy04 = (cm04[0][0]+cm04[1][1])/500

accuracy04

roc_auc_score(y_true = y_test_s, y_score = pred04)

# So I was able to create a optimal model with only 45 patients are charecterised as no-diabetes but actually 
# they're diabetes patients.

