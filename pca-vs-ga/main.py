#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn import decomposition
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt
import time

def main():
    # Carrega os dados...
    tr = np.loadtxt('./dataset/treino_baseIAM_301_Filter_1x1_new2.txt')
    ts = np.loadtxt('./dataset/teste_baseIAM_301_Filter_1x1_new2.txt')
    y_test  = ts[:, -1]
    y_train = tr[:, -1]
    X_train = tr[:, 0 : -1]
    X_test  = ts[:, 0 : -1]
    
    #Count Number of Unique Values
    print('\nY_train - classes uniques: ', len(np.unique(y_train)))
    print('Y_test - classes uniques: ', len(np.unique(y_test)))
    
    # Normaliza os dados...
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #genetico(X_train, X_test, y_train, y_test)
    combinacoes(X_train, X_test, y_train, y_test)
    
    # n_components = [16, 32, 64, 128, 256, 512]
    # for n in n_components:
    #     pca = decomposition.PCA(n)
    #     pca.fit(X_train)
    #     X_train_pca = pca.transform(X_train)
    #     X_test_pca = pca.transform(X_test)
        
    #     print('\n-------------------- PCA: ', n)
    #     print('X_train_pca: ', X_train_pca.shape)
    #     print('X_train: ', X_train.shape)
    #     print('X_test_pca: ', X_test_pca.shape)
    #     print('X_test: ', X_test.shape)
    #     classificadores(X_train_pca, X_test_pca, y_train, y_test)
def combinacoes(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import VotingClassifier
    pca = decomposition.PCA(512)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('\n********************** combine *************************')
    start_time = time.time()
    
    clf1 = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf2 = tree.DecisionTreeClassifier()
 
    voting_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft')
    voting_clf.fit(X_train_pca, y_train)
    predictions = voting_clf.predict(X_test_pca)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, predictions, zero_division=0))

def genetico(X_train, X_test, y_train, y_test):
    pca = decomposition.PCA(512)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print('\n********************** alg. genetico *************************')
    start_time = time.time()

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')

    evolved_estimator = GAFeatureSelectionCV(
        estimator=clf,
        cv=3,
        scoring="accuracy",
        population_size=512,
        generations=500,
        n_jobs=-1,
        # verbose=True,
        keep_top_k=4,
        elitism=True,
    )

    evolved_estimator.fit(X_train_pca, y_train)

    print(evolved_estimator.best_features_)

    # Predict only with the subset of selected features
    y_predict_ga = evolved_estimator.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_predict_ga)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #plot
    plot_fitness_evolution(evolved_estimator)
    plt.savefig("genetico_512.png") 
    #plt.show()
    

# def pca(n_components, nameTr, nameTs, X_train, X_test):   
#     for n in n_components:
  
#         pca = decomposition.PCA(n)
#         pca.fit(X_train)
#         X_train_pca = pca.transform(X_train)
#         X_test_pca = pca.transform(X_test)
        
#         # file_tr = nameTr + str(n)
#         # file_ts = nameTs + str(n)
#         # a_fileTr = open(file_tr, "w")
#         # a_fileTs = open(file_ts, "w")
#         # np.savetxt(a_fileTr, X_train_pca)
#         # np.savetxt(a_fileTs, X_test_pca)
        
#         print('PCA: ', n)
#         print('X_train_pca: ', X_train_pca.shape)
#         print('X_test_pca: ', X_test_pca.shape)
#         print('X_train: ', X_train.shape)
#         print('X_test: ', X_test.shape)

def classificadores(X_train, X_test, y_train, y_test):

# ------------------------------------------------------------
# Decision Tree classifier
    print('\n********************** DT *************************')
    start_time = time.time()
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))
    #tree.plot_tree(clf)

# ------------------------------------------------------------
# k-NN classifier
    print('\n********************* k-NN ************************')
    start_time = time.time()
    neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    neigh.fit(X_train, y_train)
    #neigh.score(X_test, y_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, neigh.predict(X_test), zero_division=0))
    
# ------------------------------------------------------------
# SVM classifier
    print('\n********************* SVM ************************')
    C_range = 2. ** np.arange(-5,15,2)
    gamma_range = 2. ** np.arange(3,-15,-2)
    #k = [ 'rbf']
    # instancia o classificador, gerando probabilidades
    start_time = time.time()
    srv = svm.SVC(probability=True, kernel='rbf')
    ss = StandardScaler()
    pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
    
    param_grid = {
        'svm__C' : C_range,
        'svm__gamma' : gamma_range
    }     
    
    #Faz a busca por melhores parâmetros...
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)
    
    # recupera o melhor modelo
    model = grid.best_estimator_
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

# ------------------------------------------------------------
# Multi Layer Perceptron classifier
    print('\n********************** MLP ************************')
    start_time = time.time()
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20), random_state=1)
    clf.fit(X_train, y_train)
    print(clf.predict(X_test))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))

# ------------------------------------------------------------
# Random Forest classifier
    print('\n***************** Random Forest *******************')
    start_time = time.time()
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier(n_estimators=10000, max_depth=30, random_state=1)
    clf.fit(X_train, y_train)  
    #print(clf.feature_importances_)
    print(clf.predict(X_test))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(classification_report(y_test, clf.predict(X_test), zero_division=0))


if __name__ == "__main__":
    orig_stdout = sys.stdout
    
    file_stdout = 'out-512-'+datetime.now().strftime("%d_%m_%Y_%H_%M_%S")+'.txt'
    f = open(file_stdout, 'w')
    sys.stdout = f
    
    main()
    
    sys.stdout = orig_stdout
    f.close()