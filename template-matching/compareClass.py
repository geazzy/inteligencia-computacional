import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt #apenas para mostrar as imagens no jupyter
import cv2

class CompareClass:
    
    def pearson_correlation(self, image1, image2, resize=(200, 200), border_color=(0, 0, 0)):

        # img1_array = img1 = self.openImage(str(image1), resize, border_color)
        # img2_array = img2 = self.openImage(str(image2), resize, border_color)
        # print("imagens")
        # plt.imshow(img1)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()
       
        # Carregar as imagens como arrays NumPy
        img1_array = self.openImage(str(image1), resize, border_color)
        img2_array = self.openImage(str(image2), resize, border_color)

        # Transformar as matrizes 2D das imagens em vetores 1D
        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        # Calcular o coeficiente de correlação de Pearson
        correlation = pearsonr(img1_vector, img2_vector)

        return correlation.statistic
    

    def mse(self, image1, image2, resize=(200, 200), border_color=(0, 0, 0)):

        # Carregar as imagens como arrays NumPy
        img1_array = np.array(self.openImage(str(image1), resize, border_color))
        img2_array = np.array(self.openImage(str(image2), resize, border_color))

        # Transformar as matrizes 2D das imagens em vetores 1D
        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        # Calcular o Erro Quadrático Médio
        mse = mean_squared_error(img1_vector, img2_vector)

        return mse
    

    def openImage(self, filename, resize=(200, 200), border_color=(255,255,255)):
        image = cv2.imread(filename)
        h, w = image.shape[0], image.shape[1]
        if h != w:
            margin = int(abs(h-w) / 2)
            mH, mW = (0, margin) if h>w else (margin, 0)
            image = cv2.copyMakeBorder(image, mH, mH, mW, mW, cv2.BORDER_CONSTANT, value=border_color)
        image = cv2.resize(image, resize)
        return image #npy array


    def count_pixels(self, filename, num_parts=3, resize=(200, 200), border_color=(255,255,255)):
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
                part = image[left:right, upper:lower]

                # Contar os pixels pretos (assumindo que 0 são pretos)
                black_pixels = np.sum(np.array(part) == 0)

                # Adicionar a contagem à lista
                counts.append(black_pixels)

        return counts
    
    def compare_arrays(self, array1, array2):
        # Calcular o coeficiente de correlação de Pearson
        correlation = pearsonr(array1, array2)
        mse = mean_squared_error(array1, array2)
        return (correlation.statistic, mse)

