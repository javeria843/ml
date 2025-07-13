from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_diabetes
X,y=load_diabetes(return_X_y=True)
lr=LogisticRegression(max_iter=1000)
lr_scores=cross_val_score(lr,X,y,cv=5)
print("logistic regression avg Accuracy",lr_scores.mean())
svm=SVC()
svm_scores=cross_val_score(svm,X,y,cv=5)
print("SVM AVG ACCURACY", svm_scores.mean())