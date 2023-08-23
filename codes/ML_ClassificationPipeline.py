import wget
import subprocess
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import VotingClassifier
import plotly.express as px

#=========================================================================================================================================================
#=========================================================================================================================================================

def Pipeline(dataset,labels,splitting_rate,threshold):
    
    def BarPlot(scores,names):
        df_scores=pd.DataFrame()
        df_scores['name']=names
        df_scores['score']=scores
        
        cm = sns.light_palette("green", as_cmap=True)
        s = df_scores.style.background_gradient(cmap=cm)
        sns.set(style="whitegrid")
        
        ax = sns.barplot(y="name", x="score", data=df_scores)
        fig = ax.get_figure()
        fig.savefig("BarPlot.pdf")

    def HyperparameterTunning(name,classifier,parameters,x_train,y_train,x_test,y_test):
        grid = GridSearchCV(estimator=clf, param_grid = param, cv=5, scoring="accuracy")
        grid.fit(X_train, Y_train)
        grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
        
        if grid_results.shape[1]==2   and name!="Naive_Bayes":
            Plot2D(grid_results,name)
        
        elif grid_results.shape[1]>2 and name!="Naive_Bayes":
            Plot3D(grid_results,name)

        print("{}:".format(name))
        print("The best parameters are {} with a score of {}".format(grid.best_params_, grid.best_score_))
        tunned_clf = grid.best_estimator_
        tunned_clf.fit(x_train,y_train)

        return tunned_clf, tunned_clf.score(x_test,y_test), grid.best_score_

    def MaximumVoting(final_names,final_classifiers):
        max_vot_classifier = VotingClassifier(estimators = list(zip(final_names,final_classifiers)), voting ='hard')
        max_vot_classifier.fit(X_train, Y_train)
        score = max_vot_classifier.score(X_test,Y_test)
        return max_vot_classifier, score
    
    def Plot3D(data,name):
        if name == "Neural_Network":
            k = 2
        else:
            k=1
        
        fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[k], z=data.columns[-1],color=data.columns[-1])
        fig.show()
        fig.write_html("3dPlot_{}.html".format(name)) 

    def Plot2D(data,name):
        fig, ax = plt.subplots(figsize = (12,9))
        sns.set()
        plt.plot(data[data.columns[0]],data[data.columns[-1]])
        plt.xlabel("Parameters for {}".format(name), fontsize=12)
        plt.ylabel("Accuracy",fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.savefig("Accuracy of {}.pdf".format(name))
        plt.show()

    #Classification
    names = ["Nearest_Neighbors", "SVM", 
    "Decision_Tree", "Random_Forest", 
    "Neural_Network","Naive_Bayes"]

    classifiers = [
        KNeighborsClassifier(),
        SVC(probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(),
        GaussianNB()]
    
    #Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(dataset,labels,test_size = splitting_rate)

    print("train set shape:{},labels:{}".format(X_train.shape,Y_train.shape))

    print("train set shape:{},labels:{}".format(X_test.shape,Y_test.shape))


    fig, ax = plt.subplots(figsize = (12,9))
    plt.plot([0,1],[0,1], linestyle="--", color= "grey")
    colors = ["red","blue","yellow","green","orange","purple"]
    
    scores=[]

    for name,clf,c in zip(names, classifiers,colors):
        clf.fit(X_train, Y_train)
        print(clf)
        score=clf.score(X_test, Y_test)

        Y_pred = clf.predict(X_test)
        print("The confusion matrix for {} is:".format(name))
        print(classification_report(Y_test, Y_pred))

        y_score = clf.predict_proba(X_test)
        scores.append(score)
        fpr, tpr, _ = roc_curve(Y_test, y_score[:, 1], pos_label=clf.classes_[1])
        plt.plot(fpr,tpr, color=c, linestyle = "-")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(["Random"]+names)
    plt.savefig("ROC_Curve.pdf")
    plt.show()
    
    BarPlot(scores,names)

    # Initialize the parameters' range for each algorithm that will be used in hyperparameter tunning process
    #=========================================================================================================================================================
    Nearest_Neighbors_params = {"n_neighbors": np.arange(1,21), "p":[1,2], "weights":["distance"]}

    SVM_params = {"C": np.arange(0.001, 0.11, 0.01), "degree": np.arange(1,6),
    "kernel": ["linear","poly","rbf","sigmoid"]}

    Decision_Tree_params = {"max_depth": np.arange(2,21)}

    Random_Forest_params = {"max_depth": np.arange(2,21)}

    Neural_Netwrok_params = {"alpha": np.arange(0.0001,0.01, 0.001), 
    "max_iter": np.arange(200,1100,100), "solver": ["lbfgs","sgd", "adam"], 
    "hidden_layer_sizes": [(5,),(10,),(50,),(100,), (50,50),(100,100)]}

    GaussianNB_params = {}

    Classifiers_Parameters = [Nearest_Neighbors_params, SVM_params, 
    Decision_Tree_params, Random_Forest_params, Neural_Netwrok_params, 
    GaussianNB_params]
    
    
    final_classifiers = []
    final_names = []
    scores_ = []
    predictions = []

    for name, clf, param in zip(names, classifiers, Classifiers_Parameters):
        final_clf, prediction_score, accuracy = HyperparameterTunning(name,clf,param,X_train,Y_train,X_test,Y_test)

        scores_.append(accuracy)
        predictions.append(prediction_score)
        if accuracy >= threshold:
            final_names.append(name)
            final_classifiers.append(final_clf)

    BarPlot(predictions,names)
    
    return MaximumVoting(final_names,final_classifiers)

#=========================================================================================================================================================
#=========================================================================================================================================================