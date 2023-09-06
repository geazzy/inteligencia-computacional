import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from PIL import Image #https://github.com/python-pillow/Pillow


class CompareClass:
    
    # def setResultado(self, metodo, template, arquivo_comparado, resultado):
    #     self.metodo = metodo
    #     self.template = template
    #     self.arquivo_comparado = arquivo_comparado
    #     self.resultado = resultado
        
    # def __str__(self) -> str:
    #     return f"Metodo: {self.metodo}, Arquivo comparado: {self.arquivo_comparado}, Resultado: {self.resultado}"
    
    def pearson_correlation(self, image1, image2):
        
        # Carregar as imagens como arrays NumPy
        img1_array = np.array(Image.open(image1))
        img2_array = np.array(Image.open(image2))

        # Transformar as matrizes 2D das imagens em vetores 1D
        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        # Calcular o coeficiente de correlação de Pearson
        correlation = pearsonr(img1_vector, img2_vector)

        return correlation.statistic

    def mse(self, image1, image2):

        img1_array = np.array(Image.open(image1))
        img2_array = np.array(Image.open(image2))

        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        mse = mean_squared_error(img1_vector, img2_vector)

        return mse
    
    def __str__(self) -> str:
        return f"Metodo: {self.metodo}, Arquivo comparado: {self.arquivo_comparado}, Resultado: {self.resultado}"

