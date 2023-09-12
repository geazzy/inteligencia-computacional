from pathlib import Path
import random

class Utils:

  def get_files(self, caminho_p):
    file_dict = {}
    
    for pasta in Path(caminho_p).iterdir():
      if pasta.is_dir():
        file_dict[pasta.name] = []
        
        for file in Path(pasta).iterdir():
            if file.is_file() and (file.name.endswith(".png") or file.name.endswith(".jpg")):
                file_dict[pasta.name].append(file)
    return file_dict
  
  def get_random_images(self, dict_images):
    conjunto = {}
    for item in dict_images:
        template_i = random.randint(0, len(dict_images[item]) - 1)
        template = dict_images[item][template_i]
        dict_images[item].remove(template) # Remove o template da lista de imagens
        
        random3a = random.sample(dict_images[item], 3)
        for i in range(3):
          dict_images[item].remove(random3a[i])
          
        # random3b = random.sample(dict_images[item], 3)
        # for i in range(3):
        #   dict_images[item].remove(random3b[i])

        #conjunto[item] = [template, random3a, random3b, dict_images[item]]
        conjunto[item] = [template, random3a]

    return conjunto
    

print("Carregando imagens...")
u = Utils()
formas = u.get_files("fourShapes/")
folhas = u.get_files("folhas/")

# print("Formas carregadas:")
# print("Classes: ", len(formas))
# print(formas)

# print("Folhas carregadas:")
# print("Classes: ", len(folhas))
# print(folhas)

conjunto_formas = u.get_random_images(formas)
for key in conjunto_formas.keys():
    print("Classe: ", key)
    temppalete = conjunto_formas[key][0]
    print("template: ", temppalete)
    conjunto_a = conjunto_formas[key][1]
    print("conjunto_a: ", conjunto_a)
    # conjunto_b = conjunto_formas[key][2]
    # print("conjunto_b: ", conjunto_b)