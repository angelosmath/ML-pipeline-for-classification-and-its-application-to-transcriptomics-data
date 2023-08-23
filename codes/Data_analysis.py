#%%
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

# Physical exercise related genes from REVIEW article: "A COMPENDIUM OF PHYSICAL EXERCISE-RELATED HUMAN GENES: AN 'OMIC SCALE ANALYSIS"
#=========================================================================================================================================================
def ResistanceExercise_GenesSelection (file_path):
    if ".csv" in file_path:
        ExerciseGenes = pd.read_csv(file_path, header = 1, index_col = 0, sep = ",")
    else:
        raise Exception("Check file format! Need to be a CSV file")

    ResistanceGenes = ExerciseGenes[ExerciseGenes["Linked to what type of exercise"].fillna("missing").str.contains("resistance")] #pick genes related to resistance exercise

    return ResistanceGenes["Gene symbol"].to_list()

resistance_genes = ResistanceExercise_GenesSelection("PhysicalExercise_RelatedGenes.csv")

#already known genes related to exercise adaptations, as candidates
#=========================================================================================================================================================
extras =  ["STAT3", "MSTN", "HK2", "ACTN3", "ACE", "MYOD1", "MYF5", "MYF6", "CKMT2", "GAPDH"]

# Download and clean the initial dataset about transcriptomics signature
#=========================================================================================================================================================
def file_process(link):
    url = link
    print("Downloading file")
    filename = wget.download(url)

    if (".gz" in filename) and (".tar" not in filename):
        subprocess.run("gzip -d {}".format(filename), shell=True)
        print("File unzipped with gunzip")
        unzipped_file = filename.replace(".gz", "")
    if ".tar.gz" in filename:
        subprocess.run("tar -xvf {}".format(filename), shell=True)
        print("File unzipped with tar")
        unzipped_file = filename.replace(".tar.gz", "")

    filename_final = unzipped_file+".clean"
    subprocess.run("grep -v '^[\!\#\^]' {} > {}".format(unzipped_file, filename_final), shell=True)
    
    clean_data = pd.read_csv(filename_final, sep = "\t", index_col=1)
    return clean_data

Main_Dataset = file_process("https://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS5nnn/GDS5218/soft/GDS5218.soft.gz")
Main_Dataset.head()

# Further clean of DS to select the set of genes in it
#=========================================================================================================================================================
#Main_Dataset = pd.read_csv("GDS5218.soft.clean", sep="\t", index_col= 1)

def Cleaning_Dataset(DS):
    
    data_clean = DS.loc[:,DS.columns != "ID_REF"]

    final = data_clean.drop(["--Control"], axis=0)
    return final

Main_Dataset = Cleaning_Dataset(Main_Dataset)
Main_Dataset

# Take a list of related genes proposed by authors behind the main dataset
#=========================================================================================================================================================
def CandidateGenes(file):
    if ".csv" in file:
        df = pd.read_csv(file, sep=",")
    else:
        raise Exception("Check file format! Need to be a CSV file")
    
    Candidate_genes = [x.strip(" ") for x in df["Gene Symbol"].to_list()]
    return Candidate_genes

candidate_genes = CandidateGenes("Transcriptome_signature.csv")

# Make a list of common genes between two supplementary files coming from papers
#=========================================================================================================================================================
def CommonGenes(list1,list2,extra):
    common_genes = list(sorted(set(list1).intersection(list2)))
    return common_genes + extra

profile_genes = CommonGenes(candidate_genes, resistance_genes, extras)
len(profile_genes)

# Rotate Main dataset and keep only the columns of profile genes
#=========================================================================================================================================================
Main_Final = (Main_Dataset.T)[list(set(Main_Dataset.T.columns.to_list()).intersection(profile_genes))]
Main_Final.head()

#Drop duplicate columns by highest mean
#=========================================================================================================================================================
def DuplicateDrop_byMean(X):
    genes_f = list(set(X.columns))
    means = X.mean(axis=0)
    means = list(means.iteritems())
    l=[]
    for i in range(len(genes_f)):
        S=[]
        for j in range(len(means)):
            if genes_f[i]==means[j][0]:
                S.append(means[j])
        l.append(means.index(max(S,key=lambda item:item[1])))
    return l

Main_Final_DS = Main_Final.iloc[:,DuplicateDrop_byMean(Main_Final)]

#Select labels from supplementary file
#=========================================================================================================================================================
Labels = pd.read_csv('Samples_Labels.csv',header=0,index_col=0,sep=',')
Train_Labels = Labels["label"]
Train_Labels2 = Labels['Label 2']

Main_Final_DS.head()

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
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels,test_size=splitting_rate)

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


print(Pipeline(Main_Final_DS, Train_Labels2, 0.2, 0.75))
