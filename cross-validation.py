from sklearn.model_selection import cross_val_scores
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
cv_scores = cross_val_scores(model, X, y, cv=5)
print("scores from 5 folds:",cv_scores)
print("average Accuracy",round(cv_scores.mean()*100,2), "%")

