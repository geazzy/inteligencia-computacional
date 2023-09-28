# imports
import cv2
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

def hu_moments(image, quadrants=(1,1)):
    # Tamanho dos quadrantes
    width, height = image.shape[0], image.shape[1]
    part_width = width // quadrants[0]
    part_height = height // quadrants[1]
    
    # Calculate
    huMoments = list()
    for i in range(quadrants[0]):
        for j in range(quadrants[1]):
            #coordenadas da parte da imagem
            left = i * part_width
            upper = j * part_height
            right = left + part_width
            lower = upper + part_height

            # Retorna o quadrante da imagem
            quad = image[left:right, upper:lower]

            # Calculate Moments 
            moments = cv2.moments(quad)
            # Calculate Hu Moments
            huMoments.extend(cv2.HuMoments(moments).flatten())
            
    return huMoments

def black_pixels(image, quadrants=(1,1)):
    # Tamanho dos quadrantes
    width, height = image.shape[0], image.shape[1]
    part_width = width // quadrants[0]
    part_height = height // quadrants[1]

    counts = []
    for i in range(quadrants[0]):
        for j in range(quadrants[1]):
            #coordenadas da parte da imagem
            left = i * part_width
            upper = j * part_height
            right = left + part_width
            lower = upper + part_height

            # Retorna o quadrante da imagem
            part = image[left:right, upper:lower]

            # Contar os pixels pretos (assumindo que menores que 128 são pretos)
            black_pixels = np.sum(np.array(part) < 128)

            # Adicionar a contagem à lista
            counts.append(black_pixels)

    return counts

def get_files(caminho_p):
    file_dict = {}
    for pasta in Path(caminho_p).iterdir():
        if pasta.is_dir():
            file_dict[pasta.name] = []
            
            for file in Path(pasta).iterdir():
                if file.is_file() and (file.name.endswith(".png") or file.name.endswith(".jpg")):
                    file_dict[pasta.name].append(file)
    return file_dict

def createXy(database, quadrantes):
    X_hu, X_bp, y = list(), list(), list()

    for classe in database:
        for item in database[classe]:        
            image = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE) #open img

        #hu
            X_hu.append(hu_moments(image, quadrantes))
            X_bp.append(black_pixels(image, quadrantes))
            y.append(classe)

    X_hu, X_bp, y = np.array(X_hu), np.array(X_bp), np.array(y)
    return X_hu,X_bp,y

def treino_teste_normalizado(X, y):
    StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    return X_train, X_test, y_train, y_test


def knn(neighbors, X_train, y_train, X_test):

    knn = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

# def get_report(y_test, y_pred):

#     return f1_score(y_test, y_pred, average='weighted')


def print_report(y_test, y_pred):
    # print('Accuracy: {:.4f}'.format(accuracy_score(y_test, y_pred)))
    # print('Weighted - ponderada')
    print('F1 Score: {:.4f}'.format(f1_score(y_test, y_pred, average='weighted')))
    # print('Precision Score: {:.4f}'.format(precision_score(y_test, y_pred, average='weighted')))
    # print('Recall Score: {:.4f}'.format(recall_score(y_test, y_pred, average='weighted')))


database = get_files('../fourShapes/')

params = {'neighbors': [1, 3, 5], 'quadrantes': [1, 2, 3, 4, 5] }
classes = ['circle', 'square', 'triangle']
f1_results_hu = {}
f1_results_bp = {}


for neighbor in params['neighbors']:
    for quadrante in params['quadrantes']:
        X_hu, X_bp, y = createXy(database, (quadrante, quadrante))
        X_hu_train, X_hu_test, y_hu_train, y_hu_test = treino_teste_normalizado(X_hu, y)
        X_bp_train, X_bp_test, y_bp_train, y_bp_test = treino_teste_normalizado(X_bp, y)
        y_hu_pred = knn(neighbor, X_hu_train, y_hu_train, X_hu_test)
        y_bp_pred = knn(neighbor, X_bp_train, y_bp_train, X_bp_test)
        
        # f1_results_hu.append(f1_score(y_hu_test, y_hu_pred, average='weighted'))
        # f1_results_bp.append(f1_score(y_bp_test, y_bp_pred, average='weighted'))
        f1_results_hu[(neighbor, quadrante)] = f1_score(y_hu_test, y_hu_pred, average='weighted')
        f1_results_bp[(neighbor, quadrante)] = f1_score(y_bp_test, y_bp_pred, average='weighted')
            
        print("===============================================================")
        print("KNN com {} vizinhos e {} quadrantes".format(neighbor, quadrante))
        print_report(y_hu_test,y_hu_pred)
        print()
        print_report(y_bp_test,y_bp_pred)
