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
    
    def pearson_correlation(self, image1, image2, image_type):
        
        i1 = Image.open(image1)
        i2 = Image.open(image2)
        
        if image_type == "folha":
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

    def mse(self, image1, image2, image_type):

        i1 = Image.open(image1)
        i2 = Image.open(image2)
        
        if image_type == "folha":
            width, height = min(i1.size, i2.size)
            img1 = i1.resize((width, height))
            img2 = i2.resize((width, height))
        else:
            img1 = i1
            img2 = i2
        
        # Carregar as imagens como arrays NumPy
        img1_array = np.array(img1)
        img2_array = np.array(img2)

        img1_vector = img1_array.flatten()
        img2_vector = img2_array.flatten()

        mse = mean_squared_error(img1_vector, img2_vector)

        return mse
    
    def count_pixels(self, image, num_parts=3):

        image = Image.open(image)
        width, height = image.size

        part_width = width // num_parts
        part_height = height // num_parts
        print(part_width, part_height)
        counts = []

        for i in range(num_parts):
            for j in range(num_parts):
                #coordenadas da parte da imagem
                left = i * part_width
                upper = j * part_height
                right = left + part_width
                lower = upper + part_height

                
                part = image.crop((left, upper, right, lower))

                # Contar os pixels pretos (assumindo que valores próximos de 0 são pretos)
                black_pixels = np.sum(np.array(part) == 0)

                # Adicionar a contagem à lista
                counts.append(black_pixels)

        return counts

