import sys, os, re, contextvars
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env local
load_dotenv()

import requests
import sqlalchemy
from google.adk.agents import Agent, BaseAgent, SequentialAgent, ParallelAgent
from google.adk.events import Event
from google.genai.types import Content, Part
from google.adk.tools import google_search
from typing import AsyncGenerator


# --- HELPER DE EVENTOS ADK ---
def crear_evento_texto(autor: str, texto: str, partial: bool = None) -> Event:
    return Event(
        author=autor,
        content=Content(parts=[Part(text=texto)]),
        partial=partial
    )
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.sql.connector import Connector, IPTypes


# --- VARIABLES DE ENTORNO ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GESTOR_API_BASE_URL = os.getenv("GESTOR_API_BASE_URL")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GOOGLE_ENGINE_ID = os.getenv("GOOGLE_ENGINE_ID")

GOOGLE_BD_DIRECCION = os.getenv("direccion")
GOOGLE_BD_USER = os.getenv("userbd")
GOOGLE_BD_PASSWORDBD = os.getenv("passwordbd")
GOOGLE_BD_BD = os.getenv("bd")

# --- VARIABLE DE CONTEXTO ASÍNCRONA PARA FILTRADO DE RAG ---
id_version_filter_var = contextvars.ContextVar("id_version_filter", default=None)

def extraer_ids_version(texto: str) -> list:
    """Extrae cualquier ID de versión con formato [VERSION_ID: UUID] de un texto."""
    if not texto:
        return []
    pattern = r"\[VERSION_ID:\s*([a-zA-Z0-9\-]+)\]"
    return re.findall(pattern, texto, re.IGNORECASE)

# --- CONEXIÓN SQL (lazy pool) ---
db_connector = None
db_pool = None


def get_db_pool():
    global db_pool, db_connector
    if db_connector is None:
        db_connector = Connector()
    if db_pool is None:

        def getconn():
            return db_connector.connect(
                GOOGLE_BD_DIRECCION,
                "pg8000",
                user=GOOGLE_BD_USER,
                password=GOOGLE_BD_PASSWORDBD,
                db=GOOGLE_BD_BD,
                ip_type=IPTypes.PUBLIC,
            )

        db_pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800,
        )
    return db_pool


# --- HERRAMIENTAS ---


def ejecutar_consulta_sql_dinamica(query: str) -> str:
    """Ejecuta una consulta SQL SELECT para obtener datos estructurados de RRHH desde PostgreSQL."""
    if not (
        query.strip().lower().startswith("select")
        or query.strip().lower().startswith("with")
    ):
        return "Error: Solo se permiten consultas de lectura (SELECT o WITH)."
    pool = get_db_pool()
    with pool.connect() as conn:
        try:
            result = conn.execute(sqlalchemy.text(query))
            rows = result.fetchall()
            colnames = result.keys()
            formatted_res = [dict(zip(colnames, row)) for row in rows]
            return (
                str(formatted_res) if formatted_res else "No se encontraron resultados."
            )
        except Exception as e:
            return f"Error en SQL: {str(e)}"


# --- CLIENTE VERTEX AI SEARCH (lazy loader) ---
vertex_search_client = None


def get_vertex_search_client():
    global vertex_search_client
    if vertex_search_client is None:
        vertex_search_client = discoveryengine.SearchServiceClient(
            client_options={"api_endpoint": "us-discoveryengine.googleapis.com"}
        )
    return vertex_search_client


def vertex_ai_search(query: str) -> str:
    """
    Realiza una búsqueda semántica en los documentos/PDFs de RRHH.
    Usa summary_spec para obtener una respuesta resumida generada por IA anclada en los documentos,
    más los fragmentos extractivos de cada documento relevante.
    """
    try:
        client = get_vertex_search_client()
        serving_config = (
            f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us/collections/default_collection"
            f"/dataStores/{GOOGLE_ENGINE_ID}/servingConfigs/default_config"
        )

        # Configuración de contenido: respuestas extractivas + resumen semántico con IA
        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=3,
                max_extractive_segment_count=3,
                return_extractive_segment_score=True,
            ),
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=5,
                include_citations=True,
                language_code="es",
            ),
        )

        # Leer el filtro de ID de versión de ContextVar
        ids_version = id_version_filter_var.get()
        search_filter = None
        if ids_version:
            formatted_ids = ", ".join([f'"{id_}"' for id_ in ids_version])
            search_filter = f"id: ANY({formatted_ids})"
            print(f"[VERTEX_AI_SEARCH] Aplicando filtro de búsqueda por ID de versión: {search_filter}")

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
            content_search_spec=content_search_spec,
            filter=search_filter,
        )
        response = client.search(request)

        parts = []

        # 1. Resumen semántico generado por Vertex AI (respuesta directa a la pregunta)
        if (
            hasattr(response, "summary")
            and response.summary
            and response.summary.summary_text
        ):
            parts.append(f"RESUMEN SEMÁNTICO:\n{response.summary.summary_text}")

        # 2. Fragmentos extractivos de cada documento relevante
        doc_parts = []
        for result in response.results:
            doc = result.document
            title = doc.name or "Sin título"
            doc_id = doc.id or ""
            content = ""
            if doc.derived_struct_data:
                # Prioridad: extractive_answers > extractive_segments > snippets
                if "extractive_answers" in doc.derived_struct_data:
                    answers = doc.derived_struct_data["extractive_answers"]
                    content = " [...] ".join(
                        [a.get("content", "") for a in answers if a.get("content")]
                    )
                elif "extractive_segments" in doc.derived_struct_data:
                    segments = doc.derived_struct_data["extractive_segments"]
                    content = " [...] ".join(
                        [s.get("content", "") for s in segments if s.get("content")]
                    )
                elif "snippets" in doc.derived_struct_data:
                    snippets = doc.derived_struct_data["snippets"]
                    content = " [...] ".join(
                        [s.get("snippet", "") for s in snippets if s.get("snippet")]
                    )
            if content:
                doc_parts.append(f"ID_Documento: {doc_id} | Título: {title}\n{content}")

        if doc_parts:
            parts.append("DOCUMENTOS RELEVANTES:\n" + "\n\n".join(doc_parts))

        return (
            "\n\n".join(parts)
            if parts
            else "No se encontraron documentos relevantes para esta consulta."
        )
    except Exception as e:
        return f"Error en la búsqueda de Vertex AI: {str(e)}"


def navegar_software(id_trabajador: str, id_documento: str) -> dict:
    """
    Genera el comando para que el software de RRHH navegue y abra el expediente o documento de un trabajador.
    
    REGLA CRÍTICA DE PARÁMETROS:
    - id_trabajador: Debe ser el 'id_recurso' correspondiente al EXPEDIENTE del trabajador (la carpeta principal, id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1').
    - id_documento: 
      * Si la solicitud es sobre un DOCUMENTO específico (ej. un contrato, una certificación, etc.), debe ser el 'id_recurso' de ese documento en específico.
      * Si la solicitud es sobre el EXPEDIENTE en sí mismo (ej. 'abre el expediente de Ana Blanco' o 'ubica el expediente'), se debe pasar el 'id_recurso' del expediente del trabajador en ambos parámetros. Es decir, id_documento debe ser exactamente IGUAL a id_trabajador.
    """
    return {
        "action": "OPEN_EXPEDIENTE",
        "id_trabajador": id_trabajador,
        "id_documento": id_documento,
        "worker_id": id_trabajador,
        "document_id": id_documento,
        "url": f"/explorer/{id_trabajador}",
    }


def enviar_correo(correo_destino: str, asunto: str, cuerpo: str) -> str:
    """
    Envía un correo electrónico a través de la API de Gestor.
    SOLO debe llamarse después de que el usuario haya CONFIRMADO explícitamente el envío.
    Args:
        correo_destino: Dirección de correo electrónico del destinatario.
        asunto: Asunto del correo.
        cuerpo: Cuerpo/contenido del correo.
    """
    try:
        response = requests.post(
            f"{GESTOR_API_BASE_URL}/workspace/gmail",
            json={"correo_destino": correo_destino, "asunto": asunto, "cuerpo": cuerpo},
            timeout=30,
        )
        response.raise_for_status()
        return f"Correo enviado exitosamente a {correo_destino}."
    except requests.exceptions.HTTPError as e:
        return (
            f"Error al enviar el correo (HTTP {response.status_code}): {response.text}"
        )
    except Exception as e:
        return f"Error al enviar el correo: {str(e)}"


# =============================================================================
# SUB-AGENTE 1: ANALISTA SQL (OPTIMIZADO CON MAPEO DE METADATA)
# =============================================================================
analista_sql = Agent(
    name="analista_sql",
    model="gemini-2.5-pro",  # Cambiado a Pro para un análisis de alta precisión en PostgreSQL
    tools=[ejecutar_consulta_sql_dinamica],
    instruction="""
    Eres el Analista Experto en Base de Datos de RRHH. Tu única responsabilidad es generar y ejecutar consultas SQL en PostgreSQL para obtener datos estructurados.

    ESQUEMA DE TABLAS DISPONIBLE:
    - recurso (id_recurso, titulo, id_recurso_padre, id_version_activa, id_tipo_recurso, estado)
    - version (id_version, fecha_vencimiento, metadata, id_recurso, resumen, fecha_creacion)
    - tipo_recurso (id_tipo_recurso, estructura, nombre, descripcion)
    - catalogo_tipos_expediente (id_tipo_expediente, nombre_tipo)
    - requisitos_expediente (id_tipo_expediente, id_tipo_recurso_obligatorio, obligatorio)

    REGLAS CRÍTICAS DE NEGOCIO:
    1. RELACIÓN DE METADATA: La columna 'metadata' (JSONB) se encuentra ÚNICAMENTE en la tabla 'version'. Para consultar, filtrar o extraer cualquier información de un documento, DEBES hacer un JOIN: `recurso r JOIN version v ON r.id_version_activa = v.id_version`. 
    
    2. REGLA DE LECTURA DE ATRIBUTOS (ESTRICTA):
       - EXPEDIENTES: El ÚNICO caso donde tienes permitido leer, buscar o filtrar usando la columna `recurso.titulo` es para los Expedientes de los trabajadores. El título contiene su nombre en formato 'PRIMER APELLIDO SEGUNDO APELLIDO PRIMER NOMBRE SEGUNDO NOMBRE'. 
       - DEMÁS DOCUMENTOS (Contratos, Certificaciones, etc.): Tienes PROHIBIDO usar o buscar en `recurso.titulo`. Toda la información de estos documentos (títulos, descripciones, nombres internos) se debe buscar y leer DIRECTAMENTE dentro de las llaves del campo `version.metadata` (JSONB). 

    3. DESCUBRIMIENTO DINÁMICO DE LLAVES (CAMPO ESTRUCTURA):
       - La columna `tipo_recurso.estructura` detalla la definición o los nombres de las llaves (keys) que existen dentro de la `metadata` de esa tipología de documento. 
       - Si necesitas saber cómo buscar o filtrar los datos internos de un tipo de recurso (por ejemplo, saber qué campo almacena el título de una certificación o la fecha de un cumpleaños), debes consultar de forma conceptual o mediante JOIN el campo `estructura` de la tabla `tipo_recurso` para identificar las llaves del JSONB. 

    4. IDENTIFICADORES (UUIDs) DE TIPOS DE RECURSO Y SUS LLAVES COMUNES:
       - Expediente del trabajador: '36e88186-f873-40cd-a1eb-f4bc3dd18af1' 
       - Cumpleaños: '883bcc87-e00d-4abb-b7b0-bc8ae6211d22' -> Buscar fecha en la llave correspondiente en metadata (ej. metadata ->> 'fecha_nacimiento').
       - Contratos: '139be00e-2d43-4093-b9f8-e600b405efe3' -> Buscar atributos en las llaves de metadata (ej. metadata ->> 'fecha_inicio', metadata ->> 'tipo_contrato').
       - Certificación Google: '451b234c-a3c1-4653-be73-b26514cf2853' -> El nombre o curso se lee de las llaves mapeadas en metadata (ej. metadata ->> 'Titulo_de_la_Certificacion_o_Curso').
       - Certificación SAP: '91afb78a-1cf3-49e5-af53-2996e6baa4ac' -> El nombre o curso se lee de las llaves mapeadas en metadata (ej. metadata ->> 'Titulo_de_la_Certificacion_o_Curso').

    5. BÚSQUEDA POR TRABAJADOR: Usa `recurso.titulo ILIKE '%nombre%'` o descompón con `%` ÚNICAMENTE cuando busques la carpeta raíz del trabajador (id_tipo_recurso de expediente).
    6. ESTADO Y NAVEGACIÓN: Filtra siempre por `estado = 'activo'` a menos que se indique lo contrario. Recupera SIEMPRE `id_recurso` e `id_recurso_padre`. 

    GUÍA DE FECHAS (Contratos Laborales):
    - Fecha de Ingreso: Es el valor de la fecha inicial dentro de la metadata de la versión MÁS ANTIGUA de su contrato laboral (según `fecha_creacion` en la tabla `version`).
    - Fecha de Contratación: Es el valor de la fecha inicial de la primera versión en el historial donde la llave del tipo de contrato en metadata sea exactamente 'Contrato determinado' o 'Contrato indeterminado'.

    CÁLCULO DE VACACIONES (Art. 190 LOTTT):
    - Año 1: 15 días, Año 2: 16, Año 3: 17... (máximo 30 días por año). Realiza la sumatoria acumulada total desde su fecha de contratación hasta la fecha actual.

    REGLA DE VERIFICACIÓN DE EXPEDIENTES COMPLETOS Y DOCUMENTOS FALTANTES:
    - Se activa si el usuario pregunta si un expediente está completo, qué documentos le faltan a un trabajador, o qué expediente está menos completo.
    - Paso A (Carpeta del Trabajador): Ubicar el `recurso` del expediente del trabajador (`id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1'`) por su nombre `titulo ILIKE '%[Nombre]%'` con `estado = 'activo'`.
    - Paso B (Determinar su Tipo de Expediente):
      - Buscar todas las versiones de recursos de tipo Contratos (`id_tipo_recurso = '139be00e-2d43-4093-b9f8-e600b405efe3'`) que pertenezcan a la carpeta del trabajador (`id_recurso_padre` es el `id_recurso` de la carpeta).
      - Identificar la versión más reciente según su `fecha_creacion DESC` en la tabla `version` y obtener su tipo de contrato de la metadata usando: `COALESCE(metadata ->> 'tipo_contrato', metadata ->> 'tipo')`.
      - Cruzar este valor obtenido con `nombre_tipo` en `catalogo_tipos_expediente` para obtener el `id_tipo_expediente` correspondiente (los 4 valores posibles son 'Contrato determinado', 'Contrato indeterminado', 'Convenio de talent' y 'Convenio de pasante').
    - Paso C (Obtener Requisitos Obligatorios):
      - Buscar en `requisitos_expediente` todos los `id_tipo_recurso_obligatorio` donde `id_tipo_expediente` sea el del trabajador y `obligatorio = True`. Unir con `tipo_recurso` para obtener el `nombre` legible de cada requisito obligatorio.
    - Paso D (Cruzar Requisitos con Documentos Presentes):
      - Buscar los recursos activos (`estado = 'activo'`) bajo la carpeta del trabajador (`id_recurso_padre` igual a la del trabajador) y verificar si existen sus correspondientes `id_tipo_recurso` en la lista de requisitos.
      - Si existe algún recurso activo del tipo requerido bajo su carpeta, el documento está "Cumplido". Si no existe ninguno, está "Faltante".
      - El porcentaje de completitud se calcula como: `(Número de requisitos cumplidos / Número total de requisitos obligatorios) * 100`.
    - Paso E (Presentación - ESTRICTAMENTE SIN ASTERISCOS):
      - Presenta un reporte conciso: indica el nombre del trabajador, su tipo de expediente detectado, la lista detallada de documentos obligatorios que le faltan (faltantes), los que ya tiene (cumplidos) y el porcentaje total de completitud.
      - Si te preguntan cuál expediente está "menos completo", calcula esta completitud para todos los trabajadores y muestra el listado ordenado de menor a mayor completitud de forma resumida.
      - REGLA DE FORMATO ESTRICTA: Queda absolutamente prohibido usar el carácter de asterisco (*) bajo cualquier circunstancia. No lo uses para viñetas (usa guiones medios '-' o números '1.', '2.') y no lo uses para negritas (no uses '**'). Para resaltar títulos o secciones importantes, escríbelas en MAYÚSCULAS o simplemente como texto normal. Por ejemplo, en lugar de '**Completitud:**' escribe 'COMPLETITUD:' o 'Completitud:'.

    LISTAR DOCUMENTOS DE UN EXPEDIENTE:
    Al listar los documentos de un trabajador, diseña la consulta estructurada usando un `WITH RECURSIVE` para traer los hijos directos del `id_recurso` de su expediente, uniendo las tablas de versiones, estructura_organizativa, tipo_recurso, sub_etiqueta y etiqueta. Mapea los resultados leyendo las propiedades dinámicas de la metadata según lo dictado por la estructura del tipo de recurso.

    RESOLUCIÓN CONDICIONAL DE DOCUMENTOS PARA FALLBACK:
    1. Si la pregunta es sobre el contenido narrativo, cláusulas, políticas particulares, reglamentos o texto libre dentro de un documento de un empleado específico (ej. "qué dice la cláusula de confidencialidad del contrato de Peter Labrador" o "cuáles son las condiciones del acuerdo de Juan"):
       - Primero, debes verificar si puedes responderla directamente con datos estructurados de las tablas.
       - Si NO puedes responderla porque la respuesta reside en el texto del PDF, debes ejecutar una consulta SQL para encontrar el `id_version_activa` (de la tabla `recurso`) o `id_version` (de la tabla `version`) de ese documento específico para ese empleado.
       - Una vez encontrado el ID, responde estrictamente incluyendo la etiqueta `[VERSION_ID: <id_version>]` en tu respuesta, acompañado de un mensaje indicando que localizaste el documento pero la consulta semántica detallada debe ser procesada por el especialista de documentos (ej. "Se localizó el contrato del empleado Peter Labrador [VERSION_ID: 9fae1554-469b-4395-8167-9c60e4b8df25], delego la lectura de cláusulas al RAG.").
       - Si no encuentras ningún ID de versión para ese documento en SQL, no agregues la etiqueta y responde normalmente.

    IMPORTANTE: Si la pregunta del usuario es un saludo (ej. "Hola", "Buenos días") o se refiere a políticas generales de la empresa (sin mencionar un empleado o documento específico), responde de inmediato con un texto vacío "" y delega el control. No intentes dar explicaciones de cortesía ni disculparte, pues eso evitaría que el sistema de fallback active el agente RAG.
    Responde en español. Sin asteriscos (*) en absoluto.
    """,
)

# =============================================================================
# SUB-AGENTE 2: DOCUMENTAL RAG (OPTIMIZADO)
# =============================================================================
documental_rag = Agent(
    name="documental_rag",
    model="gemini-2.5-pro",  # Cambiado a Pro para mejor interpretación de PDFs y políticas de RRHH
    tools=[vertex_ai_search],
    instruction="""
    Eres el Especialista en Documentos de RRHH. Tu única responsabilidad es buscar información dentro del texto contenido en los PDFs y expedientes digitalizados. [cite: 84, 85]

    REGLAS DE CONVIVENCIA CON EL ANALISTA:
    1. Si el usuario te pide "listar los documentos de un trabajador", "buscar fechas de ingreso" o "saber datos personales", NO uses la herramienta de búsqueda ni inventes datos. Responde con un resumen completamente vacío "" para que el Analista SQL (que tiene el esquema real) tome el control de la respuesta estructurada. [cite: 86, 88, 89]
    2. Utiliza `vertex_ai_search` ÚNICAMENTE cuando pregunten por políticas internas, el contenido de texto de una cláusula, certificaciones o detalles narrativos dentro de los documentos. [cite: 87, 90]

    CÓMO INTERPRETAR LOS RESULTADOS:
    - Prioriza siempre la sección 'RESUMEN SEMÁNTICO'. [cite: 91]
    - Usa 'DOCUMENTOS RELEVANTES' sólo para extraer el ID_Documento ('id_version') si se requiere realizar una navegación directa. [cite: 91]

    Responde en español. Sin asteriscos (*). [cite: 92]
    """,
)

# =============================================================================
# SUB-AGENTE 3: BUSCADOR WEB + ACTIVADOR CONDICIONAL
# google_search es un BuiltInTool — debe estar en su propio agente separado.
# AgenteBuscadorCondicional lo envuelve y solo lo ejecuta si el usuario pide
# explícitamente buscar en internet. Sin keywords → cero llamadas al modelo.
# =============================================================================

# Palabras clave que indican intención de búsqueda en internet
PALABRAS_CLAVE_WEB = [
    "busca en internet",
    "busca en google",
    "busca online",
    "busca en la web",
    "información actualizada sobre",
    "noticias de",
    "busca afuera",
    "consulta en internet",
    "consúltalo en internet",
    "búsqueda web",
    "buscar en internet",
    "qué dice internet",
    "google esto",
]

buscador_web = Agent(
    name="buscador_web",
    model="gemini-2.5-pro",
    tools=[google_search],
    instruction="""
    Eres el Especialista en Búsqueda Web. El usuario ha pedido explícitamente buscar información en internet.
    - Haz la búsqueda más específica y relevante posible según la solicitud.
    - Resume los resultados de forma clara y cita las fuentes.
    - Responde en español. Sin asteriscos (*).
    """,
)

# =============================================================================
# ENRUTADORES CONDICIONALES RÁPIDOS POR CÓDIGO
# Evitan llamadas innecesarias a modelos de LLM si se pueden discernir por palabras clave.
# =============================================================================

KEYWORDS_SQL = [
    "vacaciones", "cumple", "nacimiento", "contrato", "ingreso", "contratación",
    "recurso", "expediente", "trabajador", "empleado", "lista", "listar",
    "cumplen", "edad", "sueldo", "salario", "documento", "documentos",
    "falta", "faltan", "completo", "incompleto", "completitud", "requisito", "requisitos"
]

KEYWORDS_RAG = [
    "política", "politica", "cláusula", "clausula", "documento", "pdf", "norma",
    "regla", "manual", "instructivo", "archivo", "acuerdo"
]

SALUDOS_CHITCHAT = [
    "hola", "buenos días", "buenos dias", "buenas tardes", "buenas noches",
    "gracias", "ok", "listo", "adiós", "adios", "chao"
]


def es_respuesta_vacia_o_sin_resultados(texto: str) -> bool:
    """
    Determina si la respuesta generada por el Analista SQL está vacía, indica error
    o no arrojó ningún resultado en base de datos.
    """
    if not texto:
        return True
    
    # Remover cualquier ocurrencia de etiquetas VERSION_ID para evaluar la respuesta real
    texto_limpio = re.sub(r"\[VERSION_ID:\s*[a-zA-Z0-9\-]+\]", "", texto, flags=re.IGNORECASE)
    
    clean = texto_limpio.strip().lower()
    clean = clean.replace('"', '').replace("'", "").strip()
    
    if not clean:
        return True
        
    # Indicadores de ausencia de datos o errores en la base de datos
    indicadores = [
        "no se encontraron resultados",
        "no se encontró",
        "no se encontraron",
        "no tengo información",
        "no tengo informacion",
        "no hay información",
        "no hay informacion",
        "no se registra",
        "no existe",
        "error en sql",
        "error de sql",
        "[]",
        "null",
        "ningún resultado",
        "ningun resultado",
        "no tengo acceso",
        "lo siento",
        "mi función",
        "mi funcion",
        "no está en la base",
        "no esta en la base",
        "no tengo registros",
        "no se registran",
        "no tengo info"
    ]
    
    for ind in indicadores:
        if ind in clean:
            return True
            
    # Si es extremadamente corto, probablemente es un valor de descarte
    if len(clean) < 5:
        return True
        
    return False


class AgenteInvestigacionSecuencial(BaseAgent):
    """
    Agente de investigación secuencial inteligente con fallback:
    1. Siempre ejecuta analista_sql primero (self.sub_agents[0]).
    2. Si analista_sql no obtiene resultados o da un mensaje de "no se encontraron resultados"
       (comprobado por es_respuesta_vacia_o_sin_resultados), ejecuta documental_rag (self.sub_agents[1]).
    """
    async def _run_async_impl(self, ctx) -> AsyncGenerator:
        # Resetear el filtro de id_version al inicio de cada turno de investigación
        id_version_filter_var.set(None)

        mensaje_usuario = ""
        for event in reversed(ctx.session.events):
            if event.author == "user" and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        mensaje_usuario = part.text
                        break
            if mensaje_usuario:
                break

        msg_lower = mensaje_usuario.lower().strip()
        
        # Si es un saludo simple o chitchat de palabras cortas, hacemos skip silencioso de la investigación
        if msg_lower in SALUDOS_CHITCHAT or len(msg_lower) < 3:
            yield crear_evento_texto(self.name, "")
            return

        # 1. Siempre ejecutar analista_sql primero (self.sub_agents[0])
        sql_text = ""
        analista = self.sub_agents[0]
        async for event in analista.run_async(ctx):
            yield event
            autor = getattr(event, "author", "")
            if autor == analista.name:
                content_obj = getattr(event, "content", None)
                if content_obj:
                    if isinstance(content_obj, str):
                        sql_text += content_obj
                    elif hasattr(content_obj, "parts") and content_obj.parts:
                        for part in content_obj.parts:
                            if hasattr(part, "text") and part.text:
                                sql_text += part.text

        # Extraer cualquier ID de versión detectado de la respuesta de SQL
        ids_version = extraer_ids_version(sql_text)
        if ids_version:
            id_version_filter_var.set(ids_version)
            print(f"[AgenteInvestigacionSecuencial] Se extrajeron IDs de versión desde SQL: {ids_version}")

        # 2. Si no se consiguieron resultados de SQL, ejecutar el documental_rag (self.sub_agents[1]) de fallback
        if es_respuesta_vacia_o_sin_resultados(sql_text):
            rag = self.sub_agents[1]
            async for event in rag.run_async(ctx):
                yield event


# Instanciación de la investigación secuencial inteligente
investigadores_rrhh = AgenteInvestigacionSecuencial(
    name="investigadores_rrhh",
    sub_agents=[analista_sql, documental_rag],
)


class AgenteBuscadorCondicional(BaseAgent):
    """
    Agente condicional: solo invoca a buscador_web si el mensaje del usuario
    contiene palabras clave de búsqueda web. Si no, hace un skip silencioso.
    """

    async def _run_async_impl(self, ctx) -> AsyncGenerator:
        mensaje_usuario = ""
        for event in reversed(ctx.session.events):
            if event.author == "user" and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        mensaje_usuario = part.text
                        break
            if mensaje_usuario:
                break

        if any(kw in mensaje_usuario.lower() for kw in PALABRAS_CLAVE_WEB):
            async for event in self.sub_agents[0].run_async(ctx):
                yield event


buscador_web_condicional = AgenteBuscadorCondicional(
    name="buscador_web_condicional", sub_agents=[buscador_web]
)

orquestador = ParallelAgent(
    name="orquestador",
    sub_agents=[investigadores_rrhh, buscador_web_condicional],
    description="Orquestador que ejecuta la búsqueda secuencial en RRHH y (opcionalmente) la web en paralelo.",
)

# =============================================================================
# SUB-AGENTE FINAL: DIRECTOR (OPTIMIZADO CON CONSOLIDACIÓN)
# =============================================================================
director_final = Agent(
    name="director_final",
    model="gemini-2.5-pro",  # Cambiado a Pro para una síntesis inteligente, navegación y envío de correos
    tools=[navegar_software, enviar_correo],
    instruction="""
    Eres el Director de RRHH de Abside y el único punto de contacto con el usuario. Tu trabajo es consolidar, procesar y presentar la información proveniente de los investigadores (Analista SQL y RAG). [cite: 106, 107]

    TUS RESPONSABILIDADES CRÍTICAS:
    1. Manejo de Saludos (Chitchat): Si el usuario te saluda ("Hola", "Buenos días"), sé cortés, responde de manera ejecutiva y pregúntale en qué puedes ayudarle. No busques IDs ni intentes procesar datos en este escenario.
    2. Priorización y Consolidación Inteligente (REGLA DE ORO):
       - El Analista SQL es tu fuente de la verdad para datos estructurados de la base de datos (vacaciones, cumpleaños, expedientes, listas de documentos, etc.).
       - El Documental RAG es tu fuente de la verdad para políticas, cláusulas, contratos y textos de PDFs.
       - Si ambos agentes devuelven respuestas válidas, debes utilizarlas, consolidarlas o hacer match de ambas de forma inteligente si es necesario. Por ejemplo, si el Analista SQL te da la fecha de ingreso o los días de vacaciones de un trabajador, y el RAG te da la política general de vacaciones, unifica ambas informaciones para darle al usuario una respuesta completa y personalizada.
       - Si el Analista SQL no obtuvo resultados (o dio un error) y se ejecutó el RAG, utiliza y prioriza la respuesta del RAG.
    3. Lógica de Navegación (REGLAS DE ID ESTRICTAS):
       Si el usuario usa verbos de acción como "búscame", "busca", "ubícame", "encuentra", "abre", "navega" o "consigue", ejecuta INMEDIATAMENTE la herramienta `navegar_software` pasando los siguientes parámetros de forma rigurosa:
       - CASO A (Expediente o Carpeta Raíz del Trabajador): Si te piden abrir/ubicar el expediente de un trabajador (ej. "Abre el expediente de Ana Blanco"), debes pasar el ID del expediente (su 'id_recurso' con tipo de recurso de expediente '36e88186-f873-40cd-a1eb-f4bc3dd18af1') en AMBOS parámetros de la herramienta. Es decir, tanto id_trabajador como id_documento deben tener exactamente el mismo valor (el id_recurso del expediente).
       - CASO B (Un Documento Específico): Si te piden abrir/ubicar un documento específico (ej. un contrato o certificación del trabajador), debes pasar:
         * id_trabajador: El id_recurso del expediente del trabajador (su carpeta principal, tipo de recurso '36e88186-f873-40cd-a1eb-f4bc3dd18af1' o el id_recurso_padre del documento).
         * id_documento: El id_recurso del documento específico que se quiere abrir.
       Si no hay verbos de acción o de apertura explícitas, no muestres IDs internos al usuario ni llames a la herramienta.

    REGLA DE ORO PARA ENVIAR CORREOS (FLUJO OBLIGATORIO DE 2 PASOS): [cite: 114]
    Bajo ninguna circunstancia invoques `enviar_correo` sin la confirmación explícita del usuario. [cite: 114]

    - PASO 1 (Borrador): Si te piden redactar/enviar un correo, diseña el contenido con la información que posees y muéstralo textualmente usando estrictamente este formato:
      
      Destinatario: [correo del trabajador]
      Asunto: [asunto propuesto]
      ---
      [cuerpo del correo]
      ---
      ¿Confirmas el envío de este correo preliminar? Responde "sí" para enviarlo o indícame si deseas realizar algún cambio. [cite: 115, 116, 117]

      Detén tu ejecución aquí. NO llames a la herramienta en este turno. [cite: 117, 118]

    - PASO 2 (Envío): Únicamente si el usuario responde de manera afirmativa ("sí", "confirmo", "enviar", "ok") al borrador previo, procede a llamar a la herramienta `enviar_correo`. 

    FORMATO GENERAL:
    - Responde siempre en español de forma ejecutiva, clara y concisa. 
    - Queda estrictamente PROHIBIDO el uso de asteriscos (*) bajo cualquier circunstancia en tus respuestas. No utilices negritas de markdown (no uses '**' ni '*'). Si necesitas dar énfasis o destacar títulos/secciones, escríbelas en MAYÚSCULAS o simplemente como texto normal sin símbolos adicionales. Tampoco uses asteriscos para viñetas (usa guiones medios '-' o numeración).
    - Queda absolutamente PROHIBIDO mostrar cualquier etiqueta de metadatos interna como `[VERSION_ID: ...]` en tu respuesta final al usuario. Esas etiquetas son de uso interno exclusivo del sistema de fallback.
    - Identifica al trabajador por su nombre/título. 
    """,
)

# =============================================================================
# DIRECTOR OPTIMIZADO POR CÓDIGO
# =============================================================================
class AgenteDirectorOptimizado(BaseAgent):
    """
    Director final de RRHH de Abside, optimizado con:
    1. Intercepción y respuesta ultra-rápida (0.01s) para saludos y chitchat básico por código.
    2. Atajo (Bypass) del LLM si el orquestador ya produjo la respuesta de un investigador único
       y no se requiere ejecutar herramientas (navegar o enviar correo). Esto ahorra un LLM completo (~1.5s).
    """

    async def _run_async_impl(self, ctx) -> AsyncGenerator:
        # 1. Obtener el último mensaje del usuario
        mensaje_usuario = ""
        for event in reversed(ctx.session.events):
            if event.author == "user" and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        mensaje_usuario = part.text
                        break
            if mensaje_usuario:
                break

        msg_lower = mensaje_usuario.lower().strip()

        # --- OPTIMIZACIÓN A: RESPUESTA ESTÁTICA PARA SALUDOS (0.01s) ---
        saludos_directos = {
            "hola": "¡Hola! Soy tu asistente de RRHH de Abside. ¿En qué te puedo colaborar el día de hoy? 😊",
            "buenos días": "¡Buenos días! Espero que estés excelente hoy. ¿En qué te puedo colaborar? ☀️",
            "buenos dias": "¡Buenos días! Espero que estés excelente hoy. ¿En qué te puedo colaborar? ☀️",
            "buenas tardes": "¡Buenas tardes! ¿En qué te puedo ayudar o colaborar el día de hoy? ☕",
            "buenas noches": "¡Buenas noches! ¿En qué te puedo colaborar antes de terminar el día? 🌙",
            "gracias": "¡Con muchísimo gusto! Quedo a tu entera disposición si necesitas consultar algo más sobre expedientes, vacaciones o políticas de RRHH. ¡Que tengas un excelente día! 👍",
            "gracias!": "¡Con muchísimo gusto! Quedo a tu entera disposición si necesitas consultar algo más sobre expedientes, vacaciones o políticas de RRHH. ¡Que tengas un excelente día! 👍",
            "muchas gracias": "¡Con muchísimo gusto! Quedo a tu entera disposición si necesitas consultar algo más sobre expedientes, vacaciones o políticas de RRHH. ¡Que tengas un excelente día! 👍",
            "ok": "¡Excelente! Quedo atento a cualquier otra consulta que desees realizar. ¡Que tengas un buen día! 👍",
            "listo": "¡Perfecto! Quedo atento si necesitas algo más. ¡Que tengas un excelente día! 👍",
            "adiós": "¡Hasta luego! Que tengas un excelente día. Estaré aquí cuando me necesites. ¡Hasta pronto! 👋",
            "adios": "¡Hasta luego! Que tengas un excelente día. Estaré aquí cuando me necesites. ¡Hasta pronto! 👋",
            "chao": "¡Hasta luego! Que tengas un excelente día. Estaré aquí cuando me necesites. ¡Hasta pronto! 👋"
        }

        if msg_lower in saludos_directos:
            yield crear_evento_texto("director_final", saludos_directos[msg_lower])
            return

        if msg_lower in SALUDOS_CHITCHAT:
            yield crear_evento_texto("director_final", "¡Hola! Soy tu asistente de RRHH de Abside. ¿En qué te puedo colaborar el día de hoy? 😊")
            return

        # --- OPTIMIZACIÓN B: BYPASS INTELIGENTE DEL LLM DEL DIRECTOR ---
        # Si un investigador ya devolvió una respuesta válida, y no se requiere ejecutar herramientas de acción del director
        # (como navegar en el software o enviar correo), podemos entregar directamente la respuesta del investigador.
        # Esto reduce 1 llamada secuencial de LLM, ahorrando entre 1.5 y 2.5 segundos.
        
        # Palabras clave de acción que requieren la ejecución de herramientas del Director
        KEYWORDS_ACCION = ["busca", "búscame", "ubica", "ubícame", "encuentra", "abre", "consigue", "enviar", "envía", "enviame", "envíame", "correo", "email", "gmail"]
        requiere_accion = any(kw in msg_lower for kw in KEYWORDS_ACCION)

        # 1. Encontrar el índice del último mensaje del usuario para aislar el turno actual
        user_event_index = -1
        for i, event in enumerate(ctx.session.events):
            if getattr(event, "author", "") == "user":
                user_event_index = i

        # 2. Buscar las respuestas de los investigadores en los eventos únicamente de este turno actual
        respuestas_acumuladas = {}
        events_to_scan = ctx.session.events[user_event_index + 1 :] if user_event_index != -1 else ctx.session.events

        for event in events_to_scan:
            autor = getattr(event, "author", "")
            if autor in ["analista_sql", "documental_rag", "buscador_web"]:
                # Obtener el texto del contenido de forma robusta
                content_text = ""
                content_obj = getattr(event, "content", None)
                if content_obj:
                    if isinstance(content_obj, str):
                        content_text = content_obj
                    elif hasattr(content_obj, "parts") and content_obj.parts:
                        parts_text = []
                        for part in content_obj.parts:
                            if hasattr(part, "text") and part.text:
                                parts_text.append(part.text)
                        content_text = "".join(parts_text)
                
                content_text = content_text.strip()
                # Considerar vacío si no tiene caracteres legibles o si solo contiene comillas vacías
                if content_text and content_text.replace('"', '').replace("'", "").strip():
                    if autor not in respuestas_acumuladas:
                        respuestas_acumuladas[autor] = []
                    respuestas_acumuladas[autor].append(content_text)

        # 3. Consolidar respuestas, aplicando filtro de respuestas sin resultados
        respuestas_validas = {}
        for autor, partes in respuestas_acumuladas.items():
            # Filtrar duplicados exactos
            partes_unicas = []
            for p in partes:
                if p not in partes_unicas:
                    partes_unicas.append(p)
            
            # Simplificado: Unimos todas las partes únicas con un salto de línea
            texto_consolidado = "\n".join(partes_unicas).strip()
            
            # Limpiar de forma definitiva las etiquetas VERSION_ID de la respuesta antes de evaluarla o presentarla
            texto_consolidado = re.sub(r"\[VERSION_ID:\s*[a-zA-Z0-9\-]+\]", "", texto_consolidado, flags=re.IGNORECASE).strip()
                
            # Filtro inteligente: solo registrar respuesta en respuestas_validas si tiene contenido relevante
            if not es_respuesta_vacia_o_sin_resultados(texto_consolidado):
                respuestas_validas[autor] = texto_consolidado

        # 4. Si no requiere acción y solo UN investigador tiene respuesta válida,
        # hacemos bypass directo del LLM del director, simulando el streaming en el backend para la interfaz.
        if not requiere_accion and len(respuestas_validas) == 1:
            autor_unico = list(respuestas_validas.keys())[0]
            texto_completo = respuestas_validas[autor_unico]
            
            # Simulamos el streaming segmentando el texto en pequeños fragmentos (chunks)
            # de unas pocas palabras para lograr una animación visual de máquina de escribir fluida
            import asyncio
            palabras = texto_completo.split(" ")
            chunk_size = 4  # Enviar de a 4 palabras
            
            for i in range(0, len(palabras), chunk_size):
                chunk = " ".join(palabras[i : i + chunk_size])
                if i + chunk_size < len(palabras):
                    chunk += " "
                yield crear_evento_texto("director_final", chunk, partial=True)
                await asyncio.sleep(0.015)  # Micro-pausa de 15ms para un efecto de scroll natural
            
            # Yield final consolidated event so the turn completes beautifully with partial=None
            yield crear_evento_texto("director_final", texto_completo, partial=None)
            return

        # --- FLUJO NORMAL: RUN LLM DIRECTOR ---
        async for event in self.sub_agents[0].run_async(ctx):
            autor = getattr(event, "author", "")
            if autor == "director_final":
                # Interceptar y limpiar cualquier residuo de etiqueta VERSION_ID en las respuestas del director
                content_obj = getattr(event, "content", None)
                if content_obj and hasattr(content_obj, "parts") and content_obj.parts:
                    for part in content_obj.parts:
                        if hasattr(part, "text") and part.text:
                            part.text = re.sub(r"\[VERSION_ID:\s*[a-zA-Z0-9\-]+\]", "", part.text, flags=re.IGNORECASE)
            yield event


director_final_optimizado = AgenteDirectorOptimizado(
    name="director_final_optimizado",
    sub_agents=[director_final]
)

# =============================================================================
# AGENTE PRINCIPAL (root_agent)
# =============================================================================
root_agent = SequentialAgent(
    name="vertex_search_agent",
    sub_agents=[orquestador, director_final_optimizado],
    description="Orquestador inteligente que enruta secuencialmente y luego sintetiza a través del director optimizado.",
)
