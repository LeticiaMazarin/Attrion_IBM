# Importación de las librerías necesarias para la realización del análisis de los datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
from scipy import stats


path_raw = f'{os.getcwd()}/DATA/raw/'
path_processed = f'{os.getcwd()}/DATA/processed/'
path_graph = f'{os.getcwd()}/UTILS/Images/'

# Función para importar archivo
def import_df(path_raw, file_name):
    '''
   Objetivo: importar un dataset en formato DataFrame.

   args.
   ----
   data: path + nombre del archivo

   return.
   ----
   dataset

    '''
    file = file_name
    x = pd.read_csv(path_raw+file)

    return x


# Función para guardar nuevo archivo
def save_df(x, path_clean, file_name):
    '''
   Objetivo: guardar un DataFrame en formato csv.

   args.
   ----
   data: path + nombre del archivo

   return.
   ----
   dataset

    ''' 
    x.to_csv()

    return x


# Función para ver los valores nulos
def missing_data(x):
    '''
   Objetivo: encontrar una relación de los valores nulos de un dataset.

   args.
   ----
   data: dataset

   return.
   ----
   tabla con la relación de NaN
   '''
    total_null = x.isnull().sum().sort_values(ascending=False)
    percentage_null = (x.isnull().sum()/x.isnull().count()).sort_values(ascending=False)
    return pd.concat([total_null,percentage_null], axis=1, keys=['Total', 'Percent'])



path_graph = f'{os.getcwd()}/UTILS/Images/'
def plot_distribution(x, col_name, var1, var2, name_file):
    '''
    Objetivo: Construir un gráfico con la distribución en porcentaje de la cantidad de películas y series en Netflix.

    Args:
    ----
    data: dataset, nombre de la columna utilizada, 'Movie', 'TV Show', nombre del archivo del gráfico para guardarlo
    '''
    # Preparamos los datos - porcentajes de películas x series:
    type_ = x.groupby([col_name])[col_name].count()
    ration_attrition=pd.DataFrame(((type_/len(x))).round(2)).T

    # Definimos el tamaño del gráfico:
    fig, ax = plt.subplots(1,1,figsize=(10, 3))

    # Pintamos las barras
    ax.barh(ration_attrition.index, ration_attrition[var1], alpha=1,
        color='#001d6c')
    ax.barh(ration_attrition.index, ration_attrition[var2], left=ration_attrition[var1], alpha=1,
    color='#4589ff')

    # movie percentage
    for i in ration_attrition.index:
        ax.annotate(f"{int(ration_attrition[var1][i]*100)}%", 
                    xy=(ration_attrition[var1][i]/2, i),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')

        ax.annotate("Yes", 
                    xy=(ration_attrition[var1][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif', color='white')
    
    
    for i in ration_attrition.index:
        ax.annotate(f"{int(ration_attrition[var2][i]*100)}%", 
                    xy=(ration_attrition[var1][i]+ration_attrition[var2][i]/2, i),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')
        ax.annotate("No", 
                    xy=(ration_attrition[var1][i]+ration_attrition[var2][i]/2, -0.25),
                    va = 'center', ha='center',fontsize=25, fontfamily='serif',color='white')

    ax.set_xticklabels('', fontfamily='serif', rotation=0, color='white')
    ax.set_yticklabels('', fontfamily='serif', rotation=0, color='white')

    plt.savefig(path_graph + name_file+'.png',transparent=False)

    return plt.show();


def attrition_cat(x, i, attrition, name_file, color_='black'):
        # Preparamos los datos:
        # Seleccionamos los 10 países con más títulos en netflix.
        var1 = x[i].value_counts()[:11].index
        # Ahora sacamos un nuevo dataset que nos enseña la cantidad de títulos que son 'movies' o 'show' en cada país de los 10 que hemos seleccionado anteriormente:
        data_1 = x[[attrition, i]].groupby(i)[attrition].value_counts().unstack().loc[var1]
        # Creamos una nueva columna que suma las dos columnas de 'type' Así podremos sacar el porcentaje:
        data_1['sum'] = data_1.sum(axis=1)
        # Sacamos los porcentajes:
        data_ratio = (data_1.T / data_1['sum']).T[['Yes', 'No']].sort_values(by='Yes')

        # Definimos el tamaño del gráfico:
        fig, ax = plt.subplots(1,1,figsize=(10, 6))

        # Dibujamos las barras 
        ax.barh(data_ratio.index, data_ratio['Yes'], color='#001d6c', alpha=0.9, label='Yes')
        ax.barh(data_ratio.index, data_ratio['No'], left=data_ratio['Yes'], color='#4589ff', alpha=0.9, label='No')

        ax.set_yticklabels(data_ratio.index, fontsize=11, color=color_)

        # Anotación de porcentajes
        for i in data_ratio.index:
                ax.annotate(f"{data_ratio['Yes'][i]*100:.3}%", xy=(data_ratio['Yes'][i]/2, i), va = 'center', ha='center',fontsize=10, color=color_)

        #for i in data_ratio.index:
        #        ax.annotate(f"{data_ratio['No'][i]*100:.3}%", xy=(data_ratio['No'][i]/2, i), va = 'center', ha='center',fontsize=10, color=color_)
        
        # Definimos la leyenda
        ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3))
        ax.set_title(f'{name_file}')

        plt.savefig(path_graph + name_file+'.png',transparent=False)

        return plt.show();


def save_ml_model(filename, model):
    import pickle
    with open(f'{os.getcwd()}/ML/SRC/MODEL/{filename}.pkl', 'wb') as file: 
        pickle.dump(model, file)


def open_ml_model(filename):
    import pickle
    with open(f'{os.getcwd()}/MODEL/{filename}.pkl', 'rb') as file:  
        pickled_model = pickle.load(file)
    
    return pickled_model