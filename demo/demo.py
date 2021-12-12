import argparse
import sys
from pathlib import Path

p = str(Path(__file__).parent.parent)
p = p + '\src'
sys.path.insert(0, p)

from generate_features import generate_features


parser = argparse.ArgumentParser()
parser.add_argument("-svm", help="Use SVM with poly kernel")
parser.add_argument("-rf", help="Use Random Forest classifier")
parser.add_argument("-knn", help="User KNN method")
args = parser.parse_args()

print(args)


df = generate_features()
print(df.head(5))