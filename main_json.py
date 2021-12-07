import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Python 3 program to calculate Distance Between Two Points on Earth
from math import radians, cos, sin, asin, sqrt

latitude = 52.379189
longitude = 4.899431

# Función para calcular distancias entre dos puntos definidos con longitud y latitud

def distance(lat1, lat2, lon1, lon2):
    # The math module contains a function named
    # radians which convLongitude=erts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return (c * r)

def categorise(row):
    return distance(row['latitude'], latitude, row['longitude'], longitude)

# Carreguem dataset d'exemple
dataset = pd.read_json("amsterdam.json")

print(dataset.shape)

# Per veure totes les columnes amb valors inexistents
null_columns = dataset.columns[dataset.isnull().any()]
print(null_columns)
print(dataset[null_columns].isnull().sum( ))
print("Total de valors no existents:", dataset.isnull().sum().sum())

# Eliminem valors inexistents
dataset = dataset.dropna()

print("Total de valors no existents:", dataset.isnull().sum().sum())

data = dataset.values

# Creación y modificación de datos
dataset['distance_to_center'] = dataset.apply(lambda row: categorise(row), axis=1)

le = LabelEncoder()
dataset['room_type'] = le.fit_transform(dataset['room_type'])
dataset['instant_bookable'] = le.fit_transform(dataset['instant_bookable'])

dataset['price'] = dataset['price'].str.replace('$','')
dataset['price'] = dataset['price'].str.replace(',','')
dataset['price'] = pd.to_numeric(dataset['price'])


# dataset = dataset.convert_objects(convert_numeric=True)

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades


dataset = dataset[['accommodates', 'bathrooms', 'bedrooms', 'guests_included', 'distance_to_center']]
correlacio = dataset.corr()

plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.show()

print(dataset.dtypes)

relacio2 = sns.pairplot(dataset)
plt.show()