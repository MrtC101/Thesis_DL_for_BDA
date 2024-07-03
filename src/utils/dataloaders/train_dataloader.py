
import random
from torch.utils.data import DataLoader


class TrainDataLoader(DataLoader):
    """
    Implementation of a DataLoader class with an extra method for image
    sampling.
    """
    last_num :int = 0
    last_sample : list 
    seed  : int = None
    
    def det_img_sample(self, number : int, normalized : bool) -> list:
        """
            Method that returns a random number of images from the dataloader
            and always are the same.

            Args:
                number : number of patches from TrainDataset to visualize.
                normalized : if patches are normalized or not.
        """      
        if(number != self.last_num):  
            # Establecer la semilla aleatoria
            if not self.seed:
                self.seed = random.randint(0, 2**31)
            
            random.seed(self.seed)
            
            # Seleccionar n índices aleatorios deterministas
            sample_idxs = random.sample(range(len(self)), number)

            # Obtener los elementos de los índices seleccionados
            self.dataset.set_normalize(normalized)
            
            self.last_sample = []
            for i in sample_idxs:
                self.last_sample.append(self.dataset[i])
            
            self.dataset.set_normalize(True)

        return self.last_sample
