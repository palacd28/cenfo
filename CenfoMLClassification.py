import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from tld import get_tld, is_tld

#Cargar el data set  despues de ser exportado de la base de datos
data = pd.read_csv('C:\\Users\\palac\\OneDrive\\Documentos\\dataset_analisis2.csv')
#data = pd.read_csv('C:\\Users\\palac\\OneDrive\\Documentos\\dataset_analisis2.csv', encoding='utf-8', error_bad_lines=False)
data.head()
print(data)
print("")

#obteniendo la información del data set para revisar valores nulos
data.info()
data.isnull().sum()
print("")

#Agrupando los valores segun su tipo
contar_valores = data.type.value_counts()
print(contar_valores)
print("")

#Visualizando graficámente para una mejor comprensión
sns.barplot(x=contar_valores.index, y=contar_valores, width = 0.5)
plt.xlabel('Tipo de registro', color='blue', size=12, fontdict={'fontweight': 'bold'})
plt.ylabel('Cantidad', color='blue', size=12, fontdict={'fontweight': 'bold'});
plt.title("Clasificación simple", color='blue', size=18, fontdict={'fontweight': 'bold'})
plt.legend(title = 'Número total de registros: 651,191')
plt.show()
print("")

#Visualizando graficámente para una mejor comprensión
sns.lineplot(x=contar_valores.index, y=contar_valores)
plt.xlabel('Tipo de registro', color='blue', size=12, fontdict={'fontweight': 'bold'})
plt.ylabel('Cantidad', color='blue', size=12, fontdict={'fontweight': 'bold'});
plt.legend(title = 'Número total de registros: 651,191')
plt.title("Clasificación simple")
plt.show()

#Limpiar los datos de valores que no son necesarios para que el modelo sea mas efectivo
data['url'] = data['url'].replace('www.', '', regex=True)
print(data)
print("")

#Impresión rapida de los datos para verificar que todo esta bien 
data.head()
print("")

#Etiquetar los datos en un rango de [0-4] este rango sera utilizado para que el modelo
#tenga conocimiento sobre lo que debe recibir, esto se añade en una nueva columna 
add = {"Category": {"benign": 0, "defacement": 1, "phishing":2, "malware":3}}
data['Category'] = data['type']
data = data.replace(add)
print(data)

#Extraccion de caracteristicas del dataset: este código crea una nueva serie que contiene la longitud de cada cadena 
data['url_len'] = data['url'].apply(lambda x: len(str(x)))
#Función para extraer el dominio primario o dominio de alto nivel
def obtener_alto_nivel(url):
    try:
        #Intenta obtener el dominio primario de la URL utilizando la función get_tld
        alto_nivel = get_tld(url, as_object=True, fail_silently=False, fix_protocol=True)
        #Extrae el dominio primario de la URL analizada
        es_primario = alto_nivel.parsed_url.netloc
    except:
        #Si hay un error, establece es_primario en None
        es_primario = None
        #Devuelve el dominio primario
    return es_primario

#Extraccion de caracteristicas del dataset: La función es una expresión lambda que toma cada URL (i) y llama a process_tld(i) para obtener el dominio primario
data['domain'] = data['url'].apply(lambda i: obtener_alto_nivel(i))
data.head()
print(data)

#Extraccion de caracteristicas del dataset: Conteo de caracteres especiales dentro de cada registro
caracter = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for a in caracter:
    data[a] = data['url'].apply(lambda i: i.count(a))
#Impresión de los datos para visualizar los cambios
data.head()
print(data)

def url_maliciosa(valor):
    #Extraer el nombre del registro de la URL utilizando urlparse
    registro = urlparse(valor).hostname
    registro = str(registro)
    #Buscar el nombre del host en la URL
    acierto = re.search(registro, valor)
    #Si hay una coincidencia, se considera un registro malicioso (devuelve 1), de lo contrario, no es maliciosa (devuelve 0)
    if acierto:
        return 1
    else:
        return 0

#Se obtiene la función creada anteriormente y por parametrós se pasan los valores    
data['abnormal_url'] = data['url'].apply(lambda i: url_maliciosa(i))
sns.countplot(x='abnormal_url', data=data);
print(data)

#Cfuncion para determinar si el extracto http es válido o no
def http_confiable(valor_http):
    #Extraer la parte del protocolo del registro utilizando urlparse
    extracto_http = urlparse(valor_http).scheme
    #Convertir el resultado a una cadena
    coincidencia = str(extracto_http)
    #Si el protocolo es 'https', se considera seguro (devuelve 1), de lo contrario, no es seguro (devuelve 0)
    if coincidencia == 'https':
        return 1
    else:
        return 0
    
#Se obtiene la función creada anteriormente y por parametrós se pasan los valores   
data['https'] = data['url'].apply(lambda i: http_confiable(i))
sns.countplot(x='https', data=data);
print(data)    

#Extrayendo la cantidad de digitos que contiene un registro
def contador_digitos(ingreso_valor):
    #variable digito iniciada en 0
    digitos = 0
    for i in ingreso_valor:
        #si es numero 
        if i.isnumeric():
            #la variable digito suma 1
            digitos = digitos + 1
            #devolver la cantidad total despues del recorrido del for
    return digitos

#Se obtiene la función creada anteriormente y por parametrós se pasan los valores
data['digits']= data['url'].apply(lambda i: contador_digitos(i))
print(data)

#Función para contar el número de letras que contiene una dirección
def contador_letras(direccion):
    #contador de letras iniciado en 0
    numero_letras = 0
    for i in direccion:
        #si es una letra del alfabeto
        if i.isalpha():
            #numero de letras suma 1
            numero_letras = numero_letras + 1
            #devuelve el numero de letras total
    return numero_letras

#Se obtiene la función creada anteriormente y por parametrós se pasan los valores
data['letters']= data['url'].apply(lambda i: contador_letras(i))
print(data)

#Preparando el ambiente para los modelos de ML
X = data.drop(['url','type','Category','domain'],axis=1)
y = data['Category']
data.head()
print(data)
print("")

#Entrenamiento y division de los datos a evaluar en los modelos de machine learning
#La cantidad de % de prueba será de 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#Se definen e inicializan los modelos 
lista_modelos = [DecisionTreeClassifier,RandomForestClassifier,ExtraTreesClassifier,GaussianNB]
#Inicializa una lista vacía llamada precision_de_la_prueba, que se utilizará para almacenar la precisión de cada modelo en el conjunto de prueba
precision_de_la_prueba=[]
#Itera sobre cada modelo en la lista de modelos
for modelo in lista_modelos:
    print('----------------------------------------------------------------------------------')
    #Inicializa el modelo.
    print(f'Modelo en ejecución: {lista_modelos}')
    modelo_ = modelo()
    #Entrena el modelo utilizando los datos de entrenamiento (X_train y y_train).
    modelo_.fit(X_train, y_train)
    #Realiza predicciones en el conjunto de prueba (X_test).
    prediccion = modelo_.predict(X_test)
    #Calcula la precisión de las predicciones utilizando la función accuracy_score
    precision = accuracy_score(prediccion, y_test)
    #Agrega la precisión calculada a la lista precision_de_la_prueba.
    precision_de_la_prueba.append(precision)
    #Imprime la precisión del modelo en el conjunto de prueba.
    print('Test Accuracy:{:.2f}%'.format(precision*100))
    #Imprime un informe de clasificación (classification_report) que incluye métricas como precisión, recuperación y puntuación F1.
    print('Reporte de clasificación')
    print(classification_report(y_test, prediccion))
    print('Matriz de Confusión')
    #Muestra una matriz de confusión (confusion_matrix) como un mapa de calor (heatmap) utilizando Seaborn.
    matrix = confusion_matrix(y_test, prediccion)
    grafico_ = sns.heatmap(matrix/np.sum(matrix), annot=True,fmt= '0.2%')
    plt.show()
    #Imprime un mensaje indicando que la ejecución del modelo ha sido completada.
    print('La ejecución ha sido completada')
    
#Data Frame que almacena los resultados por modelo    
resultados = pd.DataFrame({"Model":['Decision Tree Classifier','Random Forest Classifier','Extra Trees Classifier','GaussianNB'],"Accuracy":precision_de_la_prueba})
#Imprimir los resultados almacenados    
print(resultados)

#Gráfico de barras que muestra la precisión de cada modelo de aprendizaje automático, con etiquetas y título personalizados    
visualizar = sns.barplot(x='Model', y='Accuracy', data=resultados, width=0.5)  
plt.xlabel('Modelos', color='blue', size=12, fontdict={'fontweight': 'bold'}, ha='center', va='center')
plt.ylabel('Precisión', color='blue', size=12, fontdict={'fontweight': 'bold'})
plt.title("Resumen de la ejecución de los algoritmos de ML", color='blue', size=12, fontdict={'fontweight': 'bold'})
plt.legend(title = 'Número total de registros: 651,191')
plt.show()
print("")