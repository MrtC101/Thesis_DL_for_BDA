"""
Clase que se encarga de iterar sobre una carpeta de parches predichos y juntarlos todos en una imagen 
de 1024x1024.
"""

@staticmethod
def mergeCrops(mask_patches : list) -> np.array:
    rows = []
    row = []
    for i, patch in enumerate(mask_patches):
        row.append(patch)
        if(i%4==3):
            line = np.hstack(row)
            rows.append(line)
            row = []
    mask = np.vstack(rows)
    return mask
