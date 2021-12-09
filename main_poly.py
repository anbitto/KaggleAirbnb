import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from math import radians, cos, sin, asin, sqrt
from scipy.stats import shapiro
from sklearn import metrics

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression


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

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)
    # Retornem el model entrenat
    return regr

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


dataset = dataset[['accommodates', 'bathrooms', 'bedrooms', 'guests_included', 'distance_to_center', 'price']]
correlacio = dataset.corr()

plt.figure(figsize=(16, 5))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
plt.show()

print(dataset.dtypes)

# relacio = sns.pairplot(dataset)
# plt.show()

dataset = dataset.drop(dataset[dataset.price > 350].index)
dataset = dataset.drop(dataset[dataset.price < 50].index)

dataset = dataset.drop(dataset[dataset.distance_to_center > 10].index)

dataset = dataset.drop(dataset[dataset.accommodates > 10].index)

dataset = dataset.drop(dataset[dataset.bathrooms > 6].index)

# revisamos si los datos siguen una distribución normal
# Apliquem el test de Shapiro per veure si es segueix una distribució Gaussiana
columns = dataset.columns
for column in columns:
    normal, value = (shapiro(dataset[column]))
    rounded_value = round(value, 5)
    if rounded_value > 0.05:
        print('Probably Gaussian')
    else:
        print("Probably not Gaussian")

dataset.describe()

x = dataset[["distance_to_center"]]
y = dataset[["price"]]

x_t = x
y_t = y

x_train, x_test, y_train, y_test = train_test_split(x_t, y_t, test_size=0.3)

model = regression(x_train, y_train)
y_hat = model.predict(x_test)

x_train = x_train.values
y_train = y_train.values

x_test = x_test.values
y_test = y_test.values

x = x.values
y = y.values

poly_model = make_pipeline(PolynomialFeatures(degree=3), linear_model.LinearRegression())
poly_model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

linear_model_1 = linear_model.LinearRegression()
linear_model_1.fit(x_train, y_train)

fig = plt.figure()
ax = plt.axes()

x_test.sort(axis=0)

ax.set(xlabel='distance_to_center', ylabel='price', title='distance_to_center vs price')
ax.scatter(x,y, alpha=0.5, cmap='viridis')
ax.plot(x_test, linear_model_1.predict(x_test), color='green', label='linear')
ax.plot(x_test, poly_model.predict(x_test), color='red', label='poly')
plt.show()


print("MSE:", mean_squared_error(poly_model.predict(x_test),y_test))
print("R^2:", r2_score(poly_model.predict(x_test), y_test))