# 1. Usar una imagen base de Python ligera
FROM python:3.11-slim

# 2. Evitar que Python genere archivos .pyc y asegurar que los logs salgan en tiempo real
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV GOOGLE_GENAI_USE_VERTEXAI true
ENV GOOGLE_CLOUD_PROJECT absidedoc
ENV GOOGLE_CLOUD_LOCATION us-central1
ENV GOOGLE_ENGINE_ID absidedoc_1771937853551
ENV AGENT_PATH /
ENV SERVICE_NAME vertex_search_agent
ENV APP_NAME vertex_search_agent_app
ENV direccion absidedoc:us-central1:vzabsdocprd
ENV userbd postgres
ENV passwordbd K@*|o9gb*ayOa.XJ
ENV bd postgres
ENV GESTOR_API_BASE_URL https://gestor-api-302797487278.us-central1.run.app
WORKDIR $APP_HOME

# 3. Instalar dependencias del sistema necesarias para conectores de BD (si fuera necesario)
# En este caso, pg8000 es puro Python, así que mantenemos la imagen limpia.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar el archivo de dependencias e instalarlas
# Asegúrate de tener un archivo requirements.txt en tu carpeta
COPY requirements.txt .
# Actualiza pip antes de instalar para evitar conflictos de versiones
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código de la aplicación
COPY . .

# 7. Ejecutar la aplicación
# NOTA: Cloud Run espera un servidor HTTP (como Flask o FastAPI) que envuelva a tus agentes.
# Usa el shell para expandir la variable PORT que da Google
CMD ["sh", "-c", "adk web --host 0.0.0.0 --port ${PORT:-8000}"]