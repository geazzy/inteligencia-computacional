# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import pearsonr
from PIL import Image #https://github.com/python-pillow/Pillow
from sklearn.metrics import mean_squared_error
from compareClass import CompareClass
from imageClass import ImageClass 

from pathlib import Path

def get_files_recursive(caminho_p):
 
  file_list = []
  for pasta in Path(caminho_p).iterdir():
    if pasta.is_dir():
        file_list.extend(get_files_recursive(pasta))
    elif pasta.is_file():
        if pasta.name.endswith(".png") or pasta.name.endswith(".jpg"):
            file_list.append(ImageClass(pasta.parent.name,pasta))
          
  return file_list

print("Carregando imagens...")
formas = get_files_recursive("fourShapes/")
folhas = get_files_recursive("folhas/")

#formas_classe = sorted(["square", "circle", "triangle", "star"])
#folhas_classe = sorted(["Acer_Mono", "Acer_Capillipes", "Acer_Opalus" ])
forma_template = {}
forma_template["square"] = "fourShapes/square/847.png"
forma_template["circle"] = "fourShapes/circle/5.png"
forma_template["triangle"] = "fourShapes/triangle/956.png"
forma_template["star"] = "fourShapes/star/511.png"
folha_template = {}
folha_template["Acer_Mono"] = "folhas/Acer_Mono/499.jpg"
folha_template["Acer_Capillipes"] = "folhas/Acer_Capillipes/610.jpg"
folha_template["Acer_Opalus"] = "folhas/Acer_Opalus/1.jpg"

print("Formas carregadas:")
print("len(formas): ", len(formas))
for forma in formas:
    print(forma.caminho)
    c = CompareClass()
    cp = c.count_pixels(forma.caminho, 3)
    print(CompareClass().count_pixels(forma.caminho, 3))
    print(forma_template.get(forma.classe))
    
    template_feature = c.count_pixels(forma_template.get(forma.classe))
    image_feature = c.count_pixels(forma.caminho, 3)
    corr_template_image1, _ = pearsonr(template_feature, image_feature)
    print("Coeficiente de Correlação de Pearson (Template vs. Image1):", corr_template_image1)
    mse_template_image1 = np.mean((np.array(template_feature) - np.array(image_feature))**2)
    print("Erro Quadrático Médio (Template vs. Image1):", mse_template_image1)
    # pearson_correlation_forma = CompareClass().pearson_correlation(forma_template.get(forma.classe), forma.caminho)
    # mse_forma = CompareClass().mse(forma_template.get(forma.classe), forma.caminho)
    # forma.set_resultado_pearson(forma_template.get(forma.classe), pearson_correlation_forma)
    # forma.set_resultado_mse(forma_template.get(forma.classe), mse_forma)
    # print(forma)
    
# ## AS FOLHAS NÃO TÊM O MESMO TAMANHO
# print("Folhas carregadas:")
# print("len(folhas): ", len(folhas))
# for folha in folhas:

#     pearson_correlation_folha = CompareClass().pearson_correlation(folha_template.get(folha.classe), folha.caminho)
#     mse_folha = CompareClass().mse(folha_template.get(folha.classe), folha.caminho)
#     folha.set_resultado_pearson(folha_template.get(folha.classe), pearson_correlation_folha)
#     folha.set_resultado_mse(folha_template.get(folha.classe), mse_folha)
#     print(folha)


