import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

import streamlit as slt 



class logistic_regression():
    model=None
    def __init__(self, degree, L2_penalty):
        self.degree = degree
        self.L2_penalty = L2_penalty
    
    def fit(self,X,y):
        
        self.model = LogisticRegression(penalty='l2', C=self.L2_penalty)
        
        # add polynomial features
        X_old=X
        for i in range(2,degree+1):
            X=np.concatenate((X,X_old**i),axis=1)
        
        self.model.fit(X, y)

        
    
    def predict(self,X):
        X_old=X
        for i in range(2,degree+1):
            X=np.concatenate((X,X_old**i),axis=1)
        return self.model.predict(X)
        
    def plot_decision_surface(self,X):
        x1_min = np.min(X[:,0])
        x1_max = np.max(X[:,0])

        x2_min = np.min(X[:,1])
        x2_max = np.max(X[:,1])

        x1, x2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))

        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)

        X_pred=np.concatenate((x1,x2),axis=1)
        y_pred=self.predict(X_pred)
        fig,ax=plt.subplots()
        
        plt.scatter(x1,x2,c=y_pred)
        plt.scatter(X[:,0], X[:,1], cmap=plt.cm.Paired, c=y)
        slt.pyplot(fig)

    def plot_hist_weights(self):
        fig,ax=plt.subplots()
        plt.hist(self.model.coef_[0])
        slt.pyplot(fig)

   



slt.title("Logistic Rgeression")


datasets=['Simple Classfication','Circles','Squares']
selected_dataset=slt.selectbox('Select Box', options=datasets)

degree=slt.select_slider('Degree', options=[1,2,3,4,5,6,7,8,9,10])

L2_penalty=slt.slider('L2 Penanlty', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

L2_penalty=float(L2_penalty)
X,y=None,None

if(selected_dataset=='Simple Classfication'):
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
    
    # fit the model and show the decision boundary
    # fig,ax=plt.subplots()
    # model=logistic_regression(degree,L2_penalty)
    # model.fit(X,y)
    # model.plot_decision_surface(X)
    




elif(selected_dataset=='Circles'):
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    # fig,ax=plt.subplots()
    # plt.scatter(X[:,0], X[:,1], c=y)
    # slt.pyplot(fig)

elif(selected_dataset=='Squares'):
    n_samples = 1000  # Number of samples
    centers = [(1,1), (-1, 1), (-1, -1), (1,-1)]
    cluster_std = [0.5, 0.5, 0.5, 0.5]
    random_state = 42  # Random state for reproducibility

    # Generate blob data
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=random_state)

    # combine the two classes in the first and third quadrant
    y[y == 0] = 2
    y[y == 3] = 1
    # fig,ax=plt.subplots()
    # # Plot the generated dataset
    # colors = ['r', 'b', 'r', 'b']  # Color for each class
    # for i in range(len(centers)):
    #     plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i])
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Dataset with Two Classes in Quadrants')
    # plt.legend()
    # slt.pyplot(fig)


model = LogisticRegression(penalty='l2', C=L2_penalty)
model.fit(X, y)

fig,ax=plt.subplots()
model=logistic_regression(degree,L2_penalty)
model.fit(X,y)
model.plot_decision_surface(X)
model.plot_hist_weights()