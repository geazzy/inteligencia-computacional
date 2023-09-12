import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from PIL import Image #https://github.com/python-pillow/Pillow
import cv2

class CompareClass:
    
    def pearson_correlation(self, image1, image2):
        i1 = Image.open(image1)
        i2 = Image.open(image2)
        
        # Redimensionar as imagens
        # if image_type == "folha":
        if i1.size != i2.size:
            width, height = min(i1.size, i2.size)
            img1 = i1.resize((width, height))
            img2 = i2.resize((width, height))
        else:
            img1 = i1
            img2 = i2

        # Carregar as imagens como arrays NumPy
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        # Transformar as matrizes 2D das imagens em vetores 1D
        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        # Calcular o coeficiente de correlação de Pearson
        correlation = pearsonr(img1_vector, img2_vector)

        return correlation.statistic
    

    def mse(self, image1, image2):
        i1 = Image.open(image1)
        i2 = Image.open(image2)
        
        # Redimensionar as imagens
        # if image_type == "folha":
        if i1.size != i2.size:
            width, height = min(i1.size, i2.size)
            img1 = i1.resize((width, height))
            img2 = i2.resize((width, height))
        else:
            img1 = i1
            img2 = i2
        
        # Carregar as imagens como arrays NumPy
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        # Transformar as matrizes 2D das imagens em vetores 1D
        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        # Calcular o Erro Quadrático Médio
        mse = mean_squared_error(img1_vector, img2_vector)

        return mse
    

    def openImage(self, filename, resize=(60, 60), border_color=(255,255,255)):
        image = cv2.imread(filename)
        h, w = image.shape[0], image.shape[1]
        if h != w:
            margin = int(abs(h-w) / 2)
            mH, mW = (0, margin) if h>w else (margin, 0)
            image = cv2.copyMakeBorder(image, mH, mH, mW, mW, cv2.BORDER_CONSTANT, value=border_color)
        image = cv2.resize(image, resize)
        return image #npy array


    def count_pixels(self, filename, num_parts=3, resize=(60, 60), border_color=(255,255,255)):
        image = self.openImage(filename, resize, border_color)
        width, height = image.shape[0], image.shape[1]

        # Tamanho dos quadrantes
        part_width = width // num_parts
        part_height = height // num_parts

        counts = []

        for i in range(num_parts):
            for j in range(num_parts):
                #coordenadas da parte da imagem
                left = i * part_width
                upper = j * part_height
                right = left + part_width
                lower = upper + part_height

                # Retorna o quadrante da imagem
                # part = image.crop((left, upper, right, lower))
                part = image[left:right, upper:lower]

                # Contar os pixels pretos (assumindo que valores próximos de 0 são pretos)
                black_pixels = np.sum(np.array(part) <= 128)

                # Adicionar a contagem à lista
                counts.append(black_pixels)

        return counts
    
    def compare_arrays(self, array1, array2):
        # Calcular o coeficiente de correlação de Pearson
        correlation = pearsonr(array1, array2)
        mse = mean_squared_error(array1, array2)
        return (correlation.statistic, mse)


# c = CompareClass()
# x = c.count_pixels('fourShapes/circle/1187.png')
# print(x)