#APC: Practica Kaggle 2021 - 2022
###Andrés Bitto Castro

###Dataset: Amsterdam - AirBnb (https://www.kaggle.com/adityadeshpande23/amsterdam-airbnb)

##Resum

El dataset està format per dades recollides de diferents ofertes d'allotjaments ubicats a Amsterdam extrets de la pàgina d'Airbnb. 

Aquest estudi del dataset esmentat consistirà en els següents apartats:

- Anàlisi del dataset i selecció dels atributs rellevants per a la predicció.
- Aplicació dels models de regressió sobre els atributs.
- Resultats i conclusions del treball.


##Objectius del dataset

L'objectiu d'aquesta práctica és predir els preus d'allotjaments Airbnb per mig d'atributs que ofereix el dataset com ara el nombre d'habitacions, nombre de lavabos, ubicació, nombre de reviews, etc.

Les dimensions del dataset són:
- 10498 mostres
- 17 atributs (inclosa la variable objectiu)


##Preprocessat

Abans de realitzar les proves, s'ha intentat ajustar les dades per tal d'extreure resultats més fiables. Primer, s'ha afegit un atribut nou anomenat distance_to_center. En base als atributs longitude i latitude, aquest camp calula la distància de l'allotjament respecte al centre de la ciutat en kilòmetres.

També s'ha dut a terme la conversió dels atributs categòrics a nombres, concretament sobre 'room_type' i 'room_type'. Després de retirar les mostres amb valors nuls, s'han suprès possibles outliers. El criteris principals han sigut terure els preus extrems (tant alts com baixos en excés).

Comprovant la correlació dels atributs, s'arriba a la conclusió de que els atributs més rellevants són: 'accommodates', 'bathrooms', 'bedrooms', 'guests_included', 'distance_to_center', 'price'.

##Models

Els models que s'han fet servir en aquest treball son els que es mostren a continuació:

 - Regressió Lineal (simple i multivariable)
 - Regressió Lineal Polinòmica (simple i multivariable)

A l'hora d'analitzar les dades i comprovar-ho amb els propis models, s'aprecia que és complicat extreure prediccions fiables sobre aquest dataset. És per això que en general no s'ha fet ús d'una varietat més ampla de models al arribar a resultats molt propers en nivell d'insatisfacció.


| Model | Atributs | MSE | Score |
| --- | --- | --- | --- | 
| Regressió Lineal | accommodates | 3506,939 | -0,525 |
| Regressió Lineal | accommodates | -1.899 | -2.513 |
| Regressió Lineal Polinòmica | accommodates, grau 2 | 7147.629 | -2.783 |
| Regressió Lineal Polinòmica | accommodates, grau 8 | 7284.545 | -2.782 |
| Regressió Lineal Polinòmica | bedrooms, grau 2 | 6533.191 | -3.364 |
| Regressió Lineal Polinòmica | bedrooms, grau 8 | 7284.545 | -2.782 |
| Regressió Lineal Polinòmica (multivariada) | accommodates, bedrooms, grau 2 | - | 0.331 |
| Regressió Lineal Polinòmica (multivariada) | accommodates, bedrooms, grau 8 | - | 0.341 |
| Regressió Lineal Polinòmica (multivariada) | accommodates, bedrooms, grau 11 | - | 0.351 |

##Conclusions

Els resultats de l'experiment ens demostren que les dades reflexades al dataset no defineixen una tendència clara entre els propietaris dels allotjaments a l'hora de definir els preus. És molt probable que aquestes decisions es prenguin tenint en consideració dades més privades/personals dels propietaris, com ara el cost de mantenir l'allotjament, la situació socio-econòmica en que es troba el propietari en qüestió, el limit de preu fins al que es troben disposats a baixar (per tal d'oferir preus més atractius, els quals es podrien considerar fora de la norma en alguns casos). A aquestes consideracions podem afegir el factor del coneixement que tingui el propietari sobre la resta del mercat, dada difícilment quantificable.

Es per això que, malgrat tenir dades aparentment determinants, no és posible treure prediccions fiables a partir dels models entrenats amb aquest dataset.






