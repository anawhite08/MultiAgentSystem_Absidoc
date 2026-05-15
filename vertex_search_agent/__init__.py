from . import agent
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_ENGINE_ID = os.getenv("GOOGLE_ENGINE_ID")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_BD_DIRECCION=os.getenv("direccion")
GOOGLE_BD_USER=os.getenv("userbd")
GOOGLE_BD_PASSWORDBD=os.getenv("passwordbd")
GOOGLE_BD_BD=os.getenv("bd")
GESTOR_API_BASE_URL = os.getenv("GESTOR_API_BASE_URL")