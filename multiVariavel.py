import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

def replace_outliers(column, threshold=30):
    processed_column = column.copy()
    for i in range(len(processed_column)):
        if processed_column[i] > threshold:
            non_outliers = processed_column[:i][processed_column[:i] <= threshold]
            if len(non_outliers) > 0:
                processed_column[i] = np.median(non_outliers)
            else:
                processed_column[i] = np.nan
    return processed_column

# Load dataset
dataset = pd.read_csv("C:/Users/TallesMarcelo/.spyder-py3/dataSet/dataMulti.csv")
dataset = dataset.dropna()
X = dataset.iloc[:, :-2]
y = dataset.iloc[:, -1].values

# Encode categorical variables
X_encoded = pd.get_dummies(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = ms.train_test_split(X_encoded, y, test_size=1/5, random_state=0)

# Train the model
regressor = lm.LinearRegression()
regressor.fit(X_train, y_train)
#coeficiDeterminacao = regressor.score(X, y)

# Prediction
y_pred = regressor.predict(X_test)

# Visualizing Training set results
def plot_basic(X_train, y_train):
    # Select a column for visualization
    X_train_col = X_train.iloc[:, 0].values
    X_train_col = replace_outliers(X_train_col)

    # Sort the data for plotting
    sorted_indices = np.argsort(X_train_col)
    X_train_col = X_train_col[sorted_indices]
    y_train = y_train[sorted_indices]

    # Plotting
    plt.scatter(X_train_col, y_train, color='red')
    plt.plot(X_train_col, regressor.predict(X_train), color='blue')
    plt.title('Training set')
    plt.xlabel('Anos de Trabalho')
    plt.ylabel('Salario - USD')
    plt.show()

def criar_indice(dataframe):
    # Criar um dicionário para rastrear os índices
    indices = {}

    # Função para criar ou repetir o índice
    def gerar_indice(nome):
        if nome in indices:
            indices[nome] += 1
            return indices[nome]
        else:
            indices[nome] = 1
            return 1

    # Aplicar a função aos dados
    dataframe['indice'] = dataframe['CountryColumns'].apply(gerar_indice)

    return dataframe

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def calcular_media_por_pais(df):
    # Extrair lista de países únicos
    paises_unicos = df['Country'].unique()
    
    # Inicializar dicionário para armazenar médias
    medias_por_pais = {}

    # Calcular a média para cada país
    for pais in paises_unicos:
        medias_por_pais[pais] = int(df[df['Country'] == pais]['Salario'].mean())
    
    return medias_por_pais

# Visualizing Test set results
def plot_3d(X_train, y_train):

    
    
    undimmifyData = undummify(X_train)
    undimmifyData['Salario'] = y_train.tolist()
    mergeCountrySalario = undimmifyData.iloc[:, [2, -1]]
    dicionarioCountryMedia = calcular_media_por_pais(mergeCountrySalario)

    

plot_basic(X_train, y_train)
plot_3d(X_train, y_train)

