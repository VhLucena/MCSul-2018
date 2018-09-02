import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn import tree
import matplotlib.pyplot as plt
#import scikitplot as skplt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold


def carregar_dados():
    """ Carregar o dataset

    Retorno:
        X: Dataset sem o atributo rótulo
        y: Somento o atributo rótulo
    """

    df = pd.read_csv("dados_UnderSampling_OneHot.csv")
    X = df.values[:,:30] # Atributos
    y = df.values[:,30]  # Rotulo
    return X,y


def criar_classificador():
    """ Cria o classificador Árvore de Decisão

    Retorno:
        Objeto classificador
    """

    classifier = DecisionTreeClassifier(criterion = "entropy",
                                        random_state = 100,
                                        max_depth=3,
                                        min_samples_leaf=5)
    return classifier


def validar_modelo(X, classifier):
    """ Validacao do Modelo com K-Fold


    """

    kf = KFold(n_splits=10)

    acuracia = []
    precision = []
    recall = []
    f1score = []

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        """
        print("Acurácia : ", metrics.accuracy_score(y_test, y_pred))
        print("Recall   : ", metrics.recall_score(y_test, y_pred))
        print("Precision: ", metrics.precision_score(y_test, y_pred))
        print("F1 Score : ", metrics.f1_score(y_test, y_pred))
        print("\n\n")
        """

        acuracia.append(metrics.accuracy_score(y_test, y_pred))
        recall.append(metrics.recall_score(y_test, y_pred))
        precision.append(metrics.precision_score(y_test, y_pred))
        f1score.append(metrics.f1_score(y_test, y_pred))


X, y = carregar_dados()
clf = criar_classificador()
validar_modelo(X, clf)
