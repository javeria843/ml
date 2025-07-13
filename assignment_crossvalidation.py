from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
def evaluate_models(models, X,y):
    for model in models:
         name = model.__class__.__name__        
         scores=cross_val_score(model,X,y,cv=5)
         mean_accuracy = round(scores.mean() * 100, 2)  # convert to percentage
         print(f"{name}:{mean_accuracy}%accuracy")

data=load_iris()
X,y=data.data,data.target
models = [
    LogisticRegression(max_iter=1000),
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier()
]
evaluate_models(models, X,y)
