from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Acceder a las variables de entorno
mi_variable = os.getenv("PYTHONPATH")
