# Importación de las librerías necesarias para la realización del análisis de los datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
from scipy import stats
from matplotlib import rc
from scipy.stats import chisquare
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC #SVC es de clasificación, SVR es de regresión

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, classification_report

from sklearn.utils import _to_object_array
from imblearn.over_sampling import SMOTE

import pickle