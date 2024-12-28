import pandas as pd
import pickle as pk

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Decision model creation
class DecisionModel:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
    def create_decision_model(self):
        dataset = self.dataset
        X = dataset.drop('churn', axis=1)
        y = dataset['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        dtree = DecisionTreeClassifier()
        dtree.fit(X_train, y_train)
        with open('Decision_tree_model.pkl', 'wb') as path:
            pk.dump(dtree, path)
        print("Decision tree model created successfully!")

# Logistic Model Creation
class LogisticModel:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)
    def creation_logistic_model(self):
        dataset = self.dataset
        X = dataset.drop('churn', axis=1)
        y = dataset['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        logistic = LogisticRegression()
        logistic.fit(X_train, y_train)
        with open('logistic_model.pkl', 'wb') as path:
            pk.dump(logistic, path)
        print("Logistic model created successfully!")

# KNN model creations
class KNNModel:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def creation_knn_model(self):
        dataset = self.dataset
        X = dataset.drop('churn', axis=1)
        y = dataset['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        with open('knn_model.pkl', 'wb') as path:
            pk.dump(knn, path)
        print("KNN model created successfully!")

# Random model creations
class RandomModel:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def creation_random_model(self):
        dataset = self.dataset
        X = dataset.drop('churn', axis=1)
        y = dataset['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        random = RandomForestClassifier()
        random.fit(X_train, y_train)
        with open('random_model.pkl', 'wb') as path:
            pk.dump(random, path)
        print("Random model created successfully!")

# Gausian model
class NaiveModel:
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def creation_naive_model(self):
        dataset = self.dataset
        X = dataset.drop('churn', axis=1)
        y = dataset['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_test = X_test
        self.y_test = y_test

        naive = GaussianNB()
        naive.fit(X_train, y_train)
        with open('naive_model.pkl', 'wb') as path:
            pk.dump(naive, path)
        print("Naive model created successfully!")

file_path = r"C:\Users\kdeepak_new\Downloads\preprocessd_data.csv"

dt_obj = DecisionModel(file_path)
dt_obj.create_decision_model()

logistic_obj = LogisticModel(file_path)
logistic_obj.creation_logistic_model()

knn_obj = KNNModel(file_path)
knn_obj.creation_knn_model()

random_obj = RandomModel(file_path)
random_obj.creation_random_model()

naive_obj = NaiveModel(file_path)
naive_obj.creation_naive_model()

