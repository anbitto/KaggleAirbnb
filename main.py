import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Carreguem dataset d'exemple
dataset = pd.read_csv('train.csv')
y_dataset = pd.read_csv('y_train.csv')

print(dataset.shape)
print(y_dataset.shape)

"""
# Per veure totes les columnes amb valors inexistents
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum( ))
print("Total de valors no existents:", dataset.isnull().sum().sum())


null_columns = y_dataset.columns[y_dataset.isnull().any()]
print(null_columns)
print(y_dataset[null_columns].isnull().sum( ))
print("Total de valors no existents:", y_dataset.isnull().sum().sum())
"""

dataset = dataset.join(y_dataset, lsuffix='_caller', rsuffix='_other')

print(dataset.shape)

data = dataset.values


# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.show()

dataset = dataset[['accommodates', 'bathrooms', 'bedrooms',
       'calculated_host_listings_count', 'guests_included',
       'host_listings_count', 'minimum_nights',
       'number_of_reviews', 'd_centre',
       'instant_bookable_t', 'room_type_Entire home/apt',
       'room_type_Private room', 'room_type_Shared room', 'price']]





#relacio2 = sns.pairplot(dataset)
#plt.show()