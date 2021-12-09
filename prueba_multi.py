from sklearn.preprocessing import LabelEncoder
from math import radians, cos, sin, asin, sqrt
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

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
plt.draw()

print(dataset.dtypes)

#dataset = dataset.drop(dataset[dataset.price > 350].index)
#dataset = dataset.drop(dataset[dataset.price < 50].index)
#dataset = dataset.drop(dataset[dataset.distance_to_center > 10].index)
#dataset = dataset.drop(dataset[dataset.accommodates > 10].index)
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

x = dataset[['distance_to_center', 'accommodates']]
y = dataset[['price']]


poly_var_train, poly_var_test, res_train, res_test = train_test_split(x, y, test_size=0.3, random_state=4)
for index in range(2,10):

    poly = PolynomialFeatures(degree=index)

    X_train = poly.fit_transform(poly_var_train)
    X_test = poly.fit_transform(poly_var_test)

    model = linear_model.LinearRegression()
    model.fit(X_train, res_train)

    print(model.score(X_train, res_train))

plt.show()