# imports
import cv2
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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



def createX_huX_bp(database, quadrantes):
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

    knn_hu = KNeighborsClassifier(n_neighbors=neighbors, metric='euclidean')
    knn_hu.fit(X_train, y_train)
    y_pred = knn_hu.predict(X_test)
    return y_pred

################## MAIN ##################

database = get_files('../fourShapes/')
quadrantes = (2,2)
neighbors = 3

X_hu, X_bp, y = createX_huX_bp(database, quadrantes)

X_hu_train, X_hu_test, y_hu_train, y_hu_test = treino_teste_normalizado(X_hu, y)
X_bp_train, X_bp_test, y_bp_train, y_bp_test = treino_teste_normalizado(X_bp, y)

y_hu_pred = knn(neighbors, X_hu_train, y_hu_train, X_hu_test)
y_bp_pred = knn(neighbors, X_bp_train, y_bp_train, X_bp_test)

print("HU")
print("Accuracy: ", accuracy_score(y_hu_test, y_hu_pred))
print("Confusion Matrix: \n", confusion_matrix(y_hu_test, y_hu_pred))
print("Classification Report: \n", classification_report(y_hu_test, y_hu_pred))
print("\n")
print("BP")
print("Accuracy: ", accuracy_score(y_bp_test, y_bp_pred))
print("Confusion Matrix: \n", confusion_matrix(y_bp_test, y_bp_pred))
print("Classification Report: \n", classification_report(y_bp_test, y_bp_pred))
