import argparse
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from generate_features import generate_features
from clean_dataset import prune_attr

import train_model as tm

parser = argparse.ArgumentParser()
parser.add_argument("-svm", help="Use SVM with poly kernel", default=0)
parser.add_argument("-rf", help="Use Random Forest classifier", default=0)
parser.add_argument("-knn", help="User KNN method", default=0)
args = parser.parse_args()

df = generate_features()
df = prune_attr(df)

def test_SVM():
    print('\nStarting SVM with poly kernel')
    poly, rbf, x_test, y_test, y_train, X , Y = tm.train_SVM(df, True)
    y_pred = poly.predict(x_test)
    acc = sk.metrics.accuracy_score(y_test, y_pred)
    f1 = sk.metrics.f1_score(y_test, y_pred, average='weighted')
    mr = sk.metrics.recall_score(y_test, y_pred, average='macro')

    print("Accuracy:",acc)
    print("F1:",f1)
    print("Recall:", mr)

    plt.figure(0)
    plt.title("Scatter between the first two attributes")
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=Y, s=50, cmap='autumn')
    plt.scatter(poly.support_vectors_[:,0],poly.support_vectors_[:,1])
    plt.show()

    "\nNext up is the ROC Curve of the SVM with Poly kernel"
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = poly.predict_proba(x_test)
    y_train = label_binarize(y_train, classes=[1, 2])
    y_test = label_binarize(y_test, classes=[1, 2])
    n_classes = y_test.shape[1]
    NSPFR=dict() 
    NSTPR=dict()
    LRFPR=dict()
    LRTPR=dict()

    for i in range(n_classes):
        NSPFR[i], NSTPR[i], _ = roc_curve(y_test[:,i], ns_probs)
        LRFPR[i], LRTPR[i], _ = roc_curve(y_test[:,i], lr_probs[:, i])
        plt.plot(LRFPR[i], LRTPR[i], marker='.', label=i)
        
    plt.figure(1)
    print('SVM KERNEL POLY')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(title='Classes')
    # show the plot
    plt.title(label='Roc curve')
    plt.show()

def test_RF():
    print('\nStarting Random Forest classifier')
    n_estim = [5,10,20,50,75,100,150,300,500, 1000]
    res_acc = []
    res_F1 = []
    res_rec = []

    for est in n_estim:
        print("\n Estimating metrics with", est, "estimators.")
        random_forest, x_test, y_test, y_train = tm.train_RF(df, est, True)
        y_pred = random_forest.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mr = recall_score(y_test, y_pred, average='macro')
        res_acc.append(acc)
        res_F1.append(f1)
        res_rec.append(mr)

    max_v = max(res_rec)
    max_pos = res_rec.index(max_v)

    plt.xlabel(n_estim)
    plt.title("Comparison")
    plt.plot(res_rec, label="Recall")
    plt.plot(res_acc, label="Accuracy") 
    plt.plot(res_F1, label="F1")
    plt.legend(["Recall", "Accuracy", "F1"])
    plt.show()

    print("Best recall value: ", round(max_v * 100, 3),"%", "with", n_estim[max_pos], "max_depth.")

    #RANDOM FOREST
    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = random_forest.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    y_train = label_binarize(y_train, classes=[1, 2])
    y_test = label_binarize(y_test, classes=[1, 2])
    n_classes = y_test.shape[1]


    NSPFR=dict() 
    NSTPR=dict()
    LRFPR=dict()
    LRTPR=dict()

    for i in range(n_classes):
        NSPFR[i], NSTPR[i], _ = roc_curve(y_test[:,i], ns_probs)
        LRFPR[i], LRTPR[i], _ = roc_curve(y_test[:,i], lr_probs[:, i])
        plt.plot(LRFPR[i], LRTPR[i], marker='.', label=i)
        

    print('Random Forest')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(title='Classes')
    # show the plot
    plt.title(label='Roc curve')
    plt.show()
    
def test_KNN():
    print('\nStarting KNN classifier')
    X = df.drop(labels=['results', 'home Score'], axis=1)
    Y = df['results']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)
    
    neig = [1,2,3,4,5,10,15,25,50,75,90]

    res_acc = []
    res_F1 = []
    res_rec = []

    for el in neig:
        print("\n Estimating metrics with", el, "neighbours.")
        knn_, x_test, y_test, y_train = tm.train_KNN(df, el)
        y_pred = knn_.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        mr = recall_score(y_test, y_pred, average='macro')
        res_acc.append(acc)
        res_F1.append(f1)
        res_rec.append(mr)
    

    max_v = max(res_rec)
    max_pos = res_rec.index(max_v)

    plt.xlabel(neig)
    plt.title("Comparison")
    plt.plot(res_rec, label="Recall")
    plt.plot(res_acc, label="Accuracy") 
    plt.plot(res_F1, label="F1")
    plt.legend(["Recall", "Accuracy", "F1"])
    plt.show()
    print("Best recall value: ", round(max_v * 100, 3),"%", "with", neig[max_pos], "n_neighbors.")

    ns_probs = [0 for _ in range(len(y_test))]
    # predict probabilities
    lr_probs = knn_.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    y_train = label_binarize(y_train, classes=[1, 2])
    y_test = label_binarize(y_test, classes=[1, 2])
    n_classes = y_test.shape[1]

    NSPFR = dict() 
    NSTPR = dict()
    LRFPR = dict()
    LRTPR = dict()

    for i in range(n_classes):
        NSPFR[i], NSTPR[i], _ = roc_curve(y_test[:,i], ns_probs)
        LRFPR[i], LRTPR[i], _ = roc_curve(y_test[:,i], lr_probs[:, i])

        plt.plot(LRFPR[i], LRTPR[i], marker='.', label=i)
        
    print('KNN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend(title='Classes')
    # show the plot
    plt.title(label='Roc curve')
    plt.show()

if (args.svm != 0):
    test_SVM()
if (args.rf != 0):
    test_RF()
if (args.knn != 0):
    test_KNN()