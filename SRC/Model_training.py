from UTILS.functions import *
from UTILS.librerias import *

try:
    os.mkdir(f'{os.getcwd()}/MODEL/')
except:
    print('Directory already exists')


# Importamos el dataset
file = 'Attrition_processed.csv'
df_processed = pd.read_csv(f'{os.getcwd()}/ML/SRC/DATA/processed/{file}')

# Separamos el dataset
X = df_processed.drop('Attrition', axis = 1)
y = df_processed['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# Realizamos el resampling del dataset para mejorar la distribución y tratar los sesgos del modelo
sm = SMOTE(sampling_strategy=0.40)
X_train, y_train = sm.fit_resample(X, y)

# Creamos el modelo de Logistic Regression
model_log = LogisticRegression(random_state=1234)
model_log_fit = model_log.fit(X_train,y_train)
# Guardamos el modelo
save_ml_model('pickle_model_log', model_log_fit)


# Creamos el modelo de Decision Tree
tree = DecisionTreeClassifier(max_depth=15, random_state=1234)
tree_fit = tree.fit(X_train,y_train)
# Guardamos el modelo
save_ml_model('pickle_model_tree', tree_fit)


# Creamos el modelo de Random Forest
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=100, criterion='entropy',random_state=1234)
model_rf_fit = model_rf.fit(X_train, y_train)
# Guardamos el modelo
save_ml_model('pickle_model_rf', model_rf_fit)


# Creamos el modelo de KNN
model_knn = KNeighborsClassifier(4)
model_knn_fit = model_knn.fit(X_train, y_train)
# Guardamos el modelo
save_ml_model('pickle_model_rknn', model_knn_fit)
