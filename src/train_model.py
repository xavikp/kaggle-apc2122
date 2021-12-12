import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def train_SVM(df, test):
    X = df.drop(labels=['results', 'home Score'], axis=1)
    Y = df['results']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.85, random_state=42, stratify=Y)
    poly = sk.svm.SVC(decision_function_shape='ovo',kernel='poly',degree=6, C=10, probability=True).fit(x_train, y_train)
    rbf = sk.svm.SVC(decision_function_shape='ovo',kernel='rbf',degree=6, C=10, probability=True).fit(x_train, y_train)

    if (test != True):
        return poly, rbf, x_test, y_test, y_train
    else:
        return poly, rbf, x_test, y_test, y_train, X, Y

def train_RF(df, est):
    X = df.drop(labels=['results', 'home Score'], axis=1)
    Y = df['results']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)
    random_forest = RandomForestClassifier(n_estimators=est,max_depth=50, max_features=0.6, random_state=42)
    random_forest.fit(x_train,y_train)
           
    return random_forest, x_test, y_test, y_train

def train_KNN(df, neigh):
    X = df.drop(labels=['results', 'home Score'], axis=1)
    Y = df['results']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)
    knn_ = KNeighborsClassifier(n_neighbors=neigh)
    knn_.fit(x_train,y_train)

    return knn_, x_test, y_test, y_train
