class ImageClass:
    classe = str()
    caminho = str()
    template = str()
    resultado_pearson = float()
    resultado_mse = float()
    
    def __init__(self, classe, caminho) -> None:
        self.classe = classe
        self.caminho = caminho    
    
    def set_resultado_pearson(self, template, resultado):
        self.template = template
        self.resultado_pearson = resultado
    
    def set_resultado_mse(self, template, resultado):
        self.template = template
        self.resultado_mse = resultado
        
    def __str__(self) -> str:
        return f"Classe: {self.classe}, Caminho: {self.caminho}, Template: {self.template}, Resultado Pearson: {self.resultado_pearson}, Resultado MSE: {self.resultado_mse}"