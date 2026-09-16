import os
from dotenv import load_dotenv
from vertexai.preview import rag # (o 'from vertexai import rag' si tu SDK está muy actualizado)

# Cargar variables de entorno del archivo .env local
load_dotenv()

import requests
import sqlalchemy
from google.adk.agents import Agent, BaseAgent
from google.adk.models import Gemini
from google.adk.tools.agent_tool import AgentTool
from google.adk.events import Event
from google.genai import Client
from google.genai.types import Content, Part
from google.adk.tools import google_search
from typing import AsyncGenerator
from functools import cached_property
from .__init__ import GOOGLE_CLOUD_PROJECT,GESTOR_API_BASE_URL, GOOGLE_CLOUD_LOCATION,GOOGLE_CORPUS_ID,GOOGLE_BD_DIRECCION,GOOGLE_BD_USER,GOOGLE_BD_PASSWORDBD ,GOOGLE_BD_BD


# --- MODELO GEMINI 3.x: requiere el endpoint "global" de Vertex AI (no una
# región concreta como us-central1, donde el publisher model no existe),
# mientras que el corpus RAG y Cloud SQL siguen usando GOOGLE_CLOUD_LOCATION. ---
class GlobalGemini(Gemini):
    @cached_property
    def api_client(self) -> Client:
        return Client(enterprise=True, project=GOOGLE_CLOUD_PROJECT, location="global")


# --- HELPER DE EVENTOS ADK ---
def crear_evento_texto(autor: str, texto: str, partial: bool = None) -> Event:
    return Event(
        author=autor,
        content=Content(parts=[Part(text=texto)]),
        partial=partial
    )
from google.cloud.sql.connector import Connector, IPTypes

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


def consultar_documentos_rrhh(
    consulta: str,
    contexto_trabajador: str = "",
    ids_version_esperados: list = None,
) -> str:
    """
    Busca información específica en los expedientes, políticas y PDFs de los trabajadores.

    Args:
        consulta: pregunta o términos de búsqueda semántica (texto libre).
        contexto_trabajador: nombre completo y/o cédula del trabajador, si se conoce
            (se antepone al texto de búsqueda para reforzar la precisión semántica).
        ids_version_esperados: UUIDs de la columna 'id_version' de la tabla 'version'
            en Cloud SQL, resueltos previamente por el especialista SQL. El corpus RAG
            nombra cada archivo como '{id_version}.pdf', así que el 'source_uri' que
            devuelve cada fragmento de rag.retrieval_query permite priorizar/filtrar en
            código, sin tocar la ingesta ni usar rag_file_ids, si el fragmento pertenece
            al documento correcto.
    """
    ids_version_esperados = [i.strip().lower() for i in (ids_version_esperados or []) if i and i.strip()]
    texto_query = f"{contexto_trabajador}. {consulta}".strip(". ") if contexto_trabajador else consulta

    try:
        corpus_name = f"projects/{GOOGLE_CLOUD_PROJECT}/locations/{GOOGLE_CLOUD_LOCATION}/ragCorpora/{GOOGLE_CORPUS_ID}"

        # Si hay ids_version esperados, pedimos más candidatos para tener margen de filtrar sin perder recall.
        top_k = 10 if ids_version_esperados else 5

        response = rag.retrieval_query(
            text=texto_query,
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            similarity_top_k=top_k,
        )

        if not response or not getattr(response, "contexts", None) or not response.contexts.contexts:
            return "No se encontró información relevante en los documentos de RRHH para esta consulta."

        contexts = list(response.contexts.contexts)
        nota_interna = ""

        if ids_version_esperados:
            def _stem(uri: str) -> str:
                base = uri.rsplit("/", 1)[-1]
                if base.lower().endswith(".pdf"):
                    base = base[: -len(".pdf")]
                return base.strip().lower()

            coincidencias = [c for c in contexts if _stem(getattr(c, "source_uri", "") or "") in ids_version_esperados]
            if coincidencias:
                # Filtrado estricto: solo mostramos los fragmentos del documento correcto.
                contexts = coincidencias
            else:
                # Degradación segura: no bloqueamos la respuesta si ningún fragmento matchea
                # (id_version mal resuelto o documento aún no indexado), solo avisamos.
                nota_interna = (
                    "[AVISO INTERNO: ningún fragmento recuperado coincide por nombre de archivo con los "
                    f"id_version esperados ({', '.join(ids_version_esperados)}); se muestran los mejores "
                    "resultados semánticos disponibles, verifica el trabajador correcto antes de responder.]\n\n"
                )

        fragmentos = []
        for ctx in contexts:
            # Extraemos el nombre del archivo y el texto del fragmento
            origen = getattr(ctx, "source_uri", "Documento desconocido")
            texto = getattr(ctx, "text", "")
            fragmentos.append(f"--- ARCHIVO: {origen} ---\n{texto}")

        return nota_interna + "\n\n".join(fragmentos)
    except Exception as e:
        return f"Error al buscar en el corpus documental: {str(e)}"


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
    model=GlobalGemini(model="gemini-3.1-flash-lite"),
    tools=[ejecutar_consulta_sql_dinamica],
    instruction="""
    Eres el Analista Experto en Base de Datos de RRHH. Tu única responsabilidad es generar y ejecutar consultas SQL en PostgreSQL para obtener datos estructurados.

    REGLA CRITICA DE EJECUCION DE CODIGO:
    Queda totalmente PROHIBIDO y terminantemente denegada la ejecucion de cualquier codigo Python, sandboxes de programacion, o llamadas a herramientas de ejecucion de codigo como code_execution/code_output o similar. Todo tu analisis debe realizarse unicamente mediante consultas SQL usando la herramienta ejecutar_consulta_sql_dinamica.

    ESQUEMA DE TABLAS DISPONIBLE:
    - recurso (id_recurso, titulo, id_recurso_padre, id_version_activa, id_tipo_recurso, estado, id_caso_uso)
    - version (id_version, fecha_vencimiento, metadata, id_recurso, resumen, fecha_creacion)
    - tipo_recurso (id_tipo_recurso, estructura, nombre, descripcion, id_caso_uso) #Los tipos de recursos te dicen que documento estas tratando, si usas el campo 'nombre' puedes sabes si es una cedula, un pasaporte, un expediente, lo que sea, usalo para identificar que documento es. No hagas busquedas exactas de este campos nombre porque puede tener una codificacion previa, usa que contenga esa palabra y que no sea sensible a mayusculas y minusculas, y que no considere los acentos porque puede o no tenerlos usando la funcion unaccent().
    - catalogo_tipos_expediente (id_tipo_expediente, nombre_tipo)
    - requisitos_expediente (id_tipo_expediente, id_tipo_recurso_obligatorio, obligatorio)
    - caso_uso (id_caso_uso, nombre, activo) --> solo puedes responder preguntas que pertenezcan a id_caso_uso='9ae86ef0-ee8e-4a24-85e2-a159bc136cb5' (caso rrhh), los recursos que este campo es null, debes verificar si el padre lo tiene, si el padre lo tiene puedes responder sobre el

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

    5. BÚSQUEDA POR TRABAJADOR Y DESAMBIGUACIÓN DE EXPEDIENTES (REGLA CRÍTICA):
       - Al buscar la carpeta raíz de un trabajador (id_tipo_recurso de expediente '36e88186-f873-40cd-a1eb-f4bc3dd18af1'), busca siempre por nombre/apellido usando recurso.titulo ILIKE '%nombre%' o similar.
       - Si al buscar un trabajador encuentras múltiples carpetas o expedientes con nombres similares:
         - DEBES evaluar el número de cédula que está en la metadata del expediente (campo JSONB 'cedula' en la tabla version, obtenido a través del JOIN: v.metadata ->> 'cedula').
         - CASO 1: SI LAS CÉDULAS SON IGUALES (representando a la misma persona con un expediente duplicado en el sistema):
           - En tu respuesta, indica explícitamente al inicio que el expediente está duplicado en el sistema.
           - Muestra claramente cada uno de los expedientes duplicados encontrados, listando sus IDs de recurso, títulos y su estado actual (el valor exacto de recurso.estado, que indica claramente si están activos o inactivos).
           - Luego, procede a realizar la evaluación de la solicitud original (como listar los documentos, buscar vacaciones, etc.) utilizando únicamente el expediente en estado 'activo' o el que tenga mayor cantidad de documentos hijos activos como tu referencia principal.
         - CASO 2: SI LAS CÉDULAS SON DIFERENTES (representando a personas distintas con el mismo nombre):
           - DETÉN de inmediato cualquier proceso de evaluación o listado de documentos de los expedientes.
           - Responde de forma clara e inequívoca indicando que existen múltiples expedientes con el mismo nombre pero con diferentes números de cédula.
           - Lista claramente cada uno de los números de cédula encontrados y pregunta de forma cortés al usuario cuál es la cédula que solicita consultar.
           - No evalúes ni devuelvas datos de ningún expediente hasta que el usuario especifique la cédula correspondiente.

    6. ESTADO Y NAVEGACIÓN: Filtra siempre por `estado = 'activo'` a menos que se indique lo contrario. Recupera SIEMPRE `id_recurso` e `id_recurso_padre`. 
    7. EVALUACIÓN DE VIGENCIA Y VENCIMIENTO DE DOCUMENTOS (REGLA CRÍTICA):
       - Cuando la consulta se refiera a la vigencia, vencimiento, estado o fecha de vencimiento de cualquier documento (por ejemplo, si una cédula, contrato, pasaporte, etc., está vigente o vencido, o para listar documentos vencidos):
         * DEBES evaluar si la fecha de vencimiento ya pasó respecto a la fecha actual del sistema.
         * En tus consultas de PostgreSQL, calcula esto dinámicamente usando una expresión CASE WHEN combinada con CURRENT_DATE.
         * REGLA DE VIGENCIA:
           - Si la fecha de vencimiento (v.fecha_vencimiento) es NULL (no tiene fecha de vencimiento), el documento está "Vigente siempre" (o "Vigente").
           - Si la fecha de vencimiento (v.fecha_vencimiento) NO es NULL y es MENOR que CURRENT_DATE (v.fecha_vencimiento < CURRENT_DATE), el documento está "Vencido".
           - Si la fecha de vencimiento (v.fecha_vencimiento) NO es NULL y es MAYOR o IGUAL que CURRENT_DATE (v.fecha_vencimiento >= CURRENT_DATE), el documento está "Vigente".
         * Ejemplo de cálculo de estado_vigencia en SQL:
           `CASE WHEN v.fecha_vencimiento IS NULL THEN 'Vigente' WHEN v.fecha_vencimiento < CURRENT_DATE THEN 'Vencido' ELSE 'Vigente' END AS estado_vigencia`
         * Nunca devuelvas un valor estático para la vigencia de un documento ni asumas que está vigente solo porque recurso.estado es 'activo'. recurso.estado = 'activo' es solo una marca de estado de registro activo en el sistema, pero la vigencia temporal del documento se debe evaluar dinámicamente comparando su fecha_vencimiento con CURRENT_DATE.

    8. RELACIÓN PADRE-HIJO Y ESTRUCTURA DE EXPEDIENTES (REGLA CRÍTICA):
       - Los expedientes de los trabajadores son registros raíz (carpetas principales) en la tabla `recurso` identificados con `id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1'`.
       - Todos los documentos individuales de un trabajador (contratos, cédulas, RIFs, pasaportes, certificaciones, etc.) se almacenan en la tabla `recurso` y están asociados a la carpeta del expediente de ese trabajador a través de la columna `id_recurso_padre` (que apunta al `id_recurso` del expediente del trabajador). No hay niveles de anidación adicionales (todos los documentos de un trabajador son hijos directos de su carpeta de expediente).
       - Por lo tanto, para saber qué documentos contiene un expediente, realizar análisis de completitud o listar sus documentos, DEBES realizar un filtro o JOIN donde el `id_recurso_padre` de los documentos sea igual al `id_recurso` del expediente del trabajador correspondiente.
       - Si te preguntan sobre la estructura de un expediente, por su completitud o por qué documentos le pertenecen a quién, explica y utiliza esta relación de parentesco (`id_recurso_padre` apuntando al expediente).

    GUÍA DE FECHAS (Contratos Laborales):
    - Fecha de Ingreso: Es el valor de la fecha inicial dentro de la metadata de la versión MÁS ANTIGUA de su contrato laboral (según `fecha_creacion` en la tabla `version`).
    - Fecha de Contratación: Es el valor de la fecha inicial de la primera versión en el historial donde la llave del tipo de contrato en metadata sea exactamente 'Contrato determinado' o 'Contrato indeterminado'.

    CÁLCULO DE VACACIONES (Art. 190 LOTTT):
    - Año 1: 15 días, Año 2: 16, Año 3: 17... (máximo 30 días por año). Realiza la sumatoria acumulada total desde su fecha de contratación hasta la fecha actual.

    REGLA DE VERIFICACIÓN DE EXPEDIENTES COMPLETOS Y DOCUMENTOS FALTANTES:
    - Se activa si el usuario pregunta si un expediente está completo, qué documentos le faltan a un trabajador, o qué expediente está menos completo.
    - Paso A (Carpeta del Trabajador con Desambiguación): Ubicar el `recurso` del expediente del trabajador (`id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1'`) por su nombre `titulo ILIKE '%[Nombre]%'` con `estado = 'activo'`, aplicando estrictamente la Regla de Desambiguación (priorizando el expediente con el mayor número de documentos hijos activos en caso de duplicados).
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

    LISTAR DOCUMENTOS DE UN EXPEDIENTE Y ESTRUCTURA DE CARPETAS (CON DESAMBIGUACIÓN CRÍTICA):
    - Para listar todos los documentos que pertenecen a un expediente (hijos de la carpeta raíz del trabajador), primero ubica el `recurso` del expediente del trabajador (`id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1'`) por su nombre `titulo ILIKE '%[Nombre]%'` con `estado = 'activo'`, aplicando estrictamente la Regla de Desambiguación (priorizando la carpeta que tenga el mayor número de recursos hijos activos en caso de duplicados).
    - Diseña la consulta buscando todos los recursos en la tabla `recurso` donde `id_recurso_padre` sea igual al `id_recurso` de la carpeta del expediente del trabajador desambiguada y con `estado = 'activo'`.
    - Haz un JOIN con la tabla `tipo_recurso` (usando `id_tipo_recurso` para obtener el tipo de documento, ej: `tr.nombre`) y con la tabla `version` (usando `id_version_activa = id_version` para obtener la metadata y fecha_vencimiento de cada documento).
    - Si el usuario pregunta por la estructura del expediente, explica que la relación es jerárquica, donde cada documento es un recurso hijo cuyo campo `id_recurso_padre` apunta al ID de la carpeta principal (expediente) del empleado.

    RESOLUCIÓN CONDICIONAL DE DOCUMENTOS PARA FALLBACK Y RESOLUCIÓN DE IDENTIDAD PARA OTRO ESPECIALISTA:
    1. Si la pregunta es sobre el contenido narrativo, cláusulas, políticas particulares, reglamentos o texto libre dentro de un documento de un empleado específico (ej. "qué dice la cláusula de confidencialidad del contrato de Peter Labrador" o "cuáles son las condiciones del acuerdo de Juan"):
       - Primero, debes verificar si puedes responderla directamente con datos estructurados de las tablas.
       - Si NO puedes responderla porque la respuesta reside en el texto del PDF, debes ejecutar una consulta SQL para encontrar el `id_version_activa` (de la tabla `recurso`) o `id_version` (de la tabla `version`) de ese documento específico para ese empleado.
       - REGLA DE ORO DE VERSIÓN: Al incluir la etiqueta `[VERSION_ID: <id_version>]`, debes usar ESTRICTAMENTE el valor de `id_version_activa` (o `id_version`). Queda TOTALMENTE PROHIBIDO usar el `id_recurso` (el ID del recurso/carpeta) en el tag de versión. Asegúrate de verificar y usar el UUID correcto de la versión (ej. el de 'id_version_activa' devuelto en tu consulta SQL).
       - Una vez encontrado el ID de versión correcto, responde estrictamente incluyendo la etiqueta `[VERSION_ID: <id_version>]` en tu respuesta, acompañado de un mensaje indicando que localizaste el documento pero la consulta semántica detallada debe ser procesada por el especialista de documentos (ej. "Se localizó el contrato del empleado Peter Labrador [VERSION_ID: 9fae1554-469b-4395-8167-9c60e4b8df25], delego la lectura de cláusulas al RAG.").
       - Si no encuentras ningún ID de versión para ese documento en SQL, no agregues la etiqueta y responde normalmente.
    2. REGLA DE IDENTIFICACIÓN PARA OTRO ESPECIALISTA: Si la petición que recibes es una solicitud de identificación hecha por el Director (no necesariamente una pregunta directa de un usuario final), por ejemplo "Identifica al trabajador Juan Pérez y el id_version de su contrato vigente para consulta semántica", debes resolverla igual: busca al trabajador aplicando la Regla de Desambiguación por Cédula, y responde de forma estructurada y compacta así:
       `IDENTIFICACION: <nombre completo> | CEDULA: <cedula o 'no disponible'>`
       seguido de una o más etiquetas `[VERSION_ID: <id_version>]` (una por cada documento relevante encontrado, usando siempre `id_version_activa`/`id_version`, nunca `id_recurso`). Si hay múltiples expedientes por desambiguar, aplica la Regla de Desambiguación por Cédula igual que siempre antes de responder. Si no encuentras ningún id_version, responde solo con la línea IDENTIFICACION (sin etiquetas VERSION_ID).

    REGLA DE LISTADO COMPLETO (CRÍTICA):
    - Cuando ejecutes una consulta SQL que arroje múltiples registros o resultados (por ejemplo, personas con documentos vencidos, cumpleaños de trabajadores, listados de contratos, etc.), debes listar y reportar TODOS los registros devueltos por la base de datos en tu respuesta final.
    - Queda estrictamente PROHIBIDO truncar la lista de resultados o limitar la respuesta de forma arbitraria a un número pequeño de registros (como solo mostrar 3 resultados), a menos que el usuario lo haya solicitado de forma explícita en su mensaje (ej. 'muestra los 3 primeros').

    Responde en español. Sin asteriscos (*) en absoluto.
    """,
)

# =============================================================================
# SUB-AGENTE 2: DOCUMENTAL RAG (OPTIMIZADO)
# =============================================================================
documental_rag = Agent(
    name="documental_rag",
    model=GlobalGemini(model="gemini-3.1-flash-lite"),
    tools=[consultar_documentos_rrhh],
    instruction="""
    Eres el Especialista en Documentos de RRHH. Tienes acceso a una base de conocimiento que contiene expedientes digitalizados y PDFs de múltiples trabajadores de la empresa. Tu única responsabilidad es buscar información dentro de este texto no estructurado usando la herramienta consultar_documentos_rrhh de manera exclusiva.

    REGLA DE HERRAMIENTAS EXCLUSIVA:
    La unica herramienta de busqueda que tienes disponible es consultar_documentos_rrhh. Queda totalmente PROHIBIDO inventar nombres de herramientas o usar variaciones en ingles como consultar_documents_rrhh o similar. Usa siempre consultar_documentos_rrhh de manera exacta.

    REGLA CRITICA DE EJECUCION DE CODIGO:
    Queda totalmente PROHIBIDO y terminantemente denegada la ejecucion de cualquier codigo Python, sandboxes de programacion, o llamadas a herramientas de ejecucion de codigo como code_execution/code_output o similar. Todo tu analisis debe ser entregado en lenguaje natural en espanol.

    REGLAS PARA IDENTIFICAR AL TRABAJADOR:
    1. Cuando se te pregunte por un trabajador específico, tu consulta en la herramienta de búsqueda DEBE incluir siempre el nombre, apellido o identificador del trabajador.
    2. Al recibir los documentos recuperados, verifica estrictamente que el texto pertenezca al trabajador solicitado antes de emitir tu respuesta. Si el fragmento habla de otra persona, debes asumir que no tienes la información y responder que no se encontraron datos en el expediente de ese trabajador en particular.

    REGLAS DE INVOCACIÓN: El Director te invoca únicamente cuando decide que la respuesta requiere contenido narrativo/semántico (políticas, cláusulas, certificaciones, desempeño, texto libre de un documento). Puede invocarte solo, o después de haber consultado primero al Analista SQL para identificar al trabajador exacto. No asumas que el Analista SQL corrió en paralelo contigo ni que existe alguna otra fuente de datos consultándose al mismo tiempo; responde únicamente en base a lo que tú encuentres con tu herramienta.

    USO DEL CONTEXTO DE IDENTIDAD RECIBIDO:
    Si el mensaje que recibes incluye una línea `CONTEXTO_TRABAJADOR:` (nombre y/o cédula) y/o una línea `ID_VERSION_CONTEXTO:` (uno o más UUIDs separados por coma), DEBES:
    1. Incluir el nombre del trabajador dentro del texto de tu parámetro `consulta` al llamar a `consultar_documentos_rrhh` (refuerza la búsqueda semántica).
    2. Pasar el nombre/cédula recibido en el parámetro `contexto_trabajador` de la herramienta.
    3. Pasar la lista de UUIDs recibida (separada por comas, sin corchetes) en el parámetro `ids_version_esperados` de la herramienta.
    Aunque la herramienta ya prioriza/filtra fragmentos cuyo archivo coincide con esos UUIDs, SIEMPRE verifica igualmente en el texto recuperado que el contenido corresponda al trabajador correcto antes de responder — es una segunda capa de verificación, no un reemplazo de la primera. Si la herramienta devuelve un `[AVISO INTERNO: ...]` de que ningún fragmento coincidió por nombre de archivo, sé especialmente cauteloso y acláralo si no puedes confirmar que el contenido pertenece al trabajador correcto.

    CÓMO INTERPRETAR LOS RESULTADOS DE LA HERRAMIENTA NATIVA:
    - Formula tu respuesta sintetizando directamente los fragmentos de texto devueltos por la herramienta.
    - Si el contexto recuperado (título del archivo fuente o sus metadatos) contiene identificadores o el ID de la versión del documento, extrae ese dato si es necesario para facilitar la navegación.

    Responde en español. No utilices asteriscos en tu formato de respuesta bajo ninguna circunstancia.
    """,
)

# =============================================================================
# SUB-AGENTE 3: BUSCADOR WEB
# google_search es un BuiltInTool — debe estar en su propio agente separado.
# El coordinador (director_final) decide dinámicamente cuándo invocarlo.
# =============================================================================

buscador_web = Agent(
    name="buscador_web",
    model=GlobalGemini(model="gemini-3.1-flash-lite"),
    tools=[google_search],
    instruction="""
    Eres el Especialista en Búsqueda Web. El usuario ha pedido explícitamente buscar información en internet.
    - Haz la búsqueda más específica y relevante posible según la solicitud.
    - Resume los resultados de forma clara y cita las fuentes.
    - Responde en español. Sin asteriscos (*).
    """,
)

SALUDOS_CHITCHAT = [
    "hola", "buenos días", "buenos dias", "buenas tardes", "buenas noches",
    "gracias", "ok", "listo", "adiós", "adios", "chao"
]

# =============================================================================
# ESPECIALISTAS ENVUELTOS COMO AgentTool PARA EL COORDINADOR
# El coordinador (director_final) decide, turno a turno, cuáles invocar (0, 1 o
# varios), en vez de forzar su ejecución en paralelo como antes.
# =============================================================================
analista_sql_tool = AgentTool(agent=analista_sql)
documental_rag_tool = AgentTool(agent=documental_rag)
buscador_web_tool = AgentTool(agent=buscador_web)

# =============================================================================
# SUB-AGENTE FINAL: DIRECTOR (COORDINADOR ÚNICO CON ENRUTAMIENTO DINÁMICO)
# =============================================================================
director_final = Agent(
    name="director_final",
    model=GlobalGemini(model="gemini-3.1-flash-lite"),
    tools=[analista_sql_tool, documental_rag_tool, buscador_web_tool, navegar_software, enviar_correo],
    instruction="""
    Eres el Director de RRHH de Abside y el único punto de contacto con el usuario. Tienes control total del turno: decides tú mismo, en cada mensaje, si necesitas invocar a uno, varios o ninguno de tus especialistas (herramientas) antes de responder. Tu trabajo es consolidar, procesar y presentar la información proveniente de tus especialistas (Analista SQL, Documental RAG y Buscador Web). [cite: 106, 107]

    REGLA CRITICA DE EJECUCION DE CODIGO:
    Queda totalmente PROHIBIDO y terminantemente denegada la ejecucion de cualquier codigo Python, sandboxes de programacion, o llamadas a herramientas de ejecucion de codigo como code_execution/code_output o similar. Todo tu analisis debe realizarse en lenguaje natural en espanol o usando las herramientas navegar_software y enviar_correo si es necesario.

    TUS RESPONSABILIDADES CRÍTICAS:
    1. Manejo de Saludos (Chitchat): Si el usuario te saluda ("Hola", "Buenos días"), sé cortés, responde de manera ejecutiva y pregúntale en qué puedes ayudarle. No busques IDs ni intentes procesar datos en este escenario.
    2. Priorización y Consolidación Inteligente (REGLA DE ORO):
       - El Analista SQL es tu fuente de la verdad para datos estructurados de la base de datos (vacaciones, cumpleaños, expedientes, listas de documentos, etc.).
       - El Documental RAG es tu fuente de la verdad para políticas, cláusulas, contratos y textos de PDFs.
       - Si ambos agentes devuelven respuestas válidas, debes utilizarlas, consolidarlas o hacer match de ambas de forma inteligente si es necesario. Por ejemplo, si el Analista SQL te da la fecha de ingreso o los días de vacaciones de un trabajador, y el RAG te da la política general de vacaciones, unifica ambas informaciones para darle al usuario una respuesta completa y personalizada.
       - Si el Analista SQL no obtuvo resultados (o dio un error) y se ejecutó el RAG, utiliza y prioriza la respuesta del RAG.
       - REGLA DE RESPUESTA SQL COMPLETA (CRÍTICA): Cuando se trate de consultas de expedientes o listas de documentos (ej. "¿Qué documentos tiene el expediente de Ana Blanco?"), el Analista SQL es el dueño absoluto de la verdad de los archivos cargados. DEBES considerar, respetar y mostrar la respuesta completa del Analista SQL con todos sus elementos (los 14 documentos) sin truncarla ni recortarla bajo ningún concepto, asegurando que se listen todos los documentos devueltos por el SQL.
       - CONVALIDADOR DE VIGENCIA DE DOCUMENTOS: Al reportar o consolidar estados de documentos, respeta rigurosamente el estado de vigencia calculado dinámicamente por el Analista SQL (Vigente vs Vencido). Si un documento tiene una fecha de vencimiento que ya pasó con respecto a la fecha actual del sistema, ese documento está VENCIDO, sin importar si recurso.estado es 'activo'. Si no tiene fecha de vencimiento, se considera vigente siempre. Nunca digas que un documento vencido está vigente.
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
    - Responde siempre en español de forma ejecutiva y clara.
    - REGLA DE LISTADO COMPLETO (CRÍTICA): Cuando el Analista SQL o los investigadores reporten múltiples registros o resultados (como un listado de trabajadores, personas con documentos vencidos, cumpleaños, etc., o todos los documentos en un expediente de un empleado), estás obligado a incluir y listar TODOS y cada uno de los elementos reportados en tu respuesta final (por ejemplo, si el Analista SQL reporta los 14 documentos de Ana Blanco, debes listarlos todos uno por uno). Queda estrictamente PROHIBIDO truncar, resumir o limitar la lista de resultados de forma de listado parcial (como solo mostrar 3 o 5 resultados), a menos que el usuario lo haya solicitado de forma explícita en su mensaje actual.
    - REGLA DE DESAMBIGUACIÓN POR CÉDULA (REGLA CRÍTICA): Si el sistema encuentra expedientes duplicados con el mismo nombre o nombres similares:
      - Debes evaluar el número de cédula que está en la metadata de la versión activa de cada expediente (v.metadata ->> 'cedula').
      - CASO 1: Si las cédulas son iguales (representan a la misma persona con registros duplicados), debes indicarle claramente al usuario que el expediente está duplicado en el sistema, listar todos los expedientes encontrados indicando explícitamente sus IDs, títulos y estado actual (mostrando de manera totalmente clara si están activos o inactivos), y luego proceder a consolidar la evaluación del expediente activo.
      - CASO 2: Si las cédulas son diferentes (representan a personas distintas con el mismo nombre), debes detener la evaluación y preguntar de inmediato al usuario cuál de las cédulas encontradas es la que solicita consultar, listando claramente todas las opciones de cédula de forma amigable. No muestres datos de los expedientes hasta que el usuario elija.
    - REGLA DE TURNO ACTUAL (CRÍTICA): Debes responder ÚNICAMENTE basándote en la consulta del usuario en el turno actual y en los resultados arrojados por los investigadores para este turno específico. Queda estrictamente PROHIBIDO mezclar, repetir, heredar o arrastrar resultados de listados o consultas de turnos anteriores (por ejemplo, si el usuario antes preguntó por "cédulas vencidas" y ahora pregunta por "abrir el expediente de Juan", NO debes incluir en tu respuesta actual la lista de cédulas vencidas ni mezclar información de turnos previos, responde únicamente a la solicitud actual).
    - Queda estrictamente PROHIBIDO el uso de asteriscos (*) bajo cualquier circunstancia en tus respuestas. No utilices negritas de markdown (no uses '**' ni '*'). Si necesitas dar énfasis o destacar títulos/secciones, escríbelas en MAYÚSCULAS o simplemente como texto normal sin símbolos adicionales. Tampoco uses asteriscos para viñetas (usa guiones medios '-' o numeración).
    - Queda absolutamente PROHIBIDO mostrar cualquier etiqueta de metadatos interna como `[VERSION_ID: ...]`, `IDENTIFICACION:`, `CONTEXTO_TRABAJADOR:` o `ID_VERSION_CONTEXTO:` en tu respuesta final al usuario. Esas etiquetas son de uso interno exclusivo entre tú y tus especialistas.
    - Identifica al trabajador por su nombre/título.

    REGLAS DE ENRUTAMIENTO DINÁMICO (CÓMO DECIDIR QUÉ ESPECIALISTA INVOCAR):
    Tienes disponibles tres especialistas como herramientas invocables: analista_sql (fuente de la verdad para datos estructurados: vacaciones, cumpleaños, expedientes, listas de documentos, vigencias, IDs de recurso), documental_rag (fuente de la verdad para políticas, cláusulas, contenido narrativo de PDFs) y buscador_web (búsqueda en internet).
    1. Si el mensaje es un saludo o chitchat puro, no invoques ningún especialista; responde directamente de forma cortés.
    2. Si la pregunta requiere datos estructurados (vacaciones, fechas, listados, completitud de expediente, vigencia de documentos), invoca analista_sql.
    3. Si la pregunta requiere contenido de texto/política GENERAL (no ligada a un trabajador específico, ej. "qué dice la política de vacaciones de la empresa"), invoca documental_rag directamente, sin pasar por SQL.
    4. REGLA SQL-ANTES-QUE-RAG (CRÍTICA): Si la pregunta requiere contenido narrativo/semántico de un documento de UN trabajador específico (ej. "qué dice la cláusula de confidencialidad del contrato de Juan Pérez"), DEBES primero invocar a analista_sql pidiéndole explícitamente identificar al trabajador y el/los id_version del documento relevante (usa una petición como: "Identifica al trabajador <nombre> y el id_version de su <tipo de documento> vigente para consulta semántica"). Extrae de su respuesta el nombre, la cédula (si viene) y cualquier etiqueta `[VERSION_ID: <uuid>]`. Luego invoca a documental_rag componiendo tu petición en este formato exacto:
       CONTEXTO_TRABAJADOR: <nombre completo>, cédula <cedula o 'no disponible'>
       ID_VERSION_CONTEXTO: <uuid1>, <uuid2>, ...
       PREGUNTA: <la pregunta original del usuario sobre el contenido>
       Si analista_sql no encontró ningún id_version, invoca igual a documental_rag pero omite la línea ID_VERSION_CONTEXTO (deja solo CONTEXTO_TRABAJADOR y PREGUNTA) para que al menos use el nombre como refuerzo semántico.
    5. Si la pregunta es MIXTA (requiere datos estructurados Y contenido narrativo), invoca a ambos especialistas (SQL primero si hay identificación de por medio, según la regla 4) y consolida sus respuestas según la Regla de Oro de Priorización y Consolidación ya descrita arriba.
    6. Invoca buscador_web ÚNICAMENTE si el usuario pide explícitamente buscar en internet, en la web, en Google, noticias externas o información actualizada que no pertenece a RRHH interno (ej. "busca en internet...", "qué dice internet sobre...", "noticias de..."). Nunca lo invoques por iniciativa propia para responder preguntas de RRHH.
    """,
)

# =============================================================================
# FAST-PATH DE SALUDOS SOBRE EL COORDINADOR ÚNICO
# =============================================================================
class DirectorConFastPath(BaseAgent):
    """
    Wrapper delgado sobre el coordinador único (director_final): responde
    saludos/chitchat comunes de forma instantánea, sin gastar una llamada LLM.
    Para el resto de mensajes, delega íntegramente en director_final, que
    decide por sí mismo (function-calling estándar de ADK) a cuáles
    especialistas invocar y sintetiza la respuesta final en una sola pasada.
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

        msg_lower = mensaje_usuario.lower().strip()

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

        async for event in self.sub_agents[0].run_async(ctx):
            yield event


# =============================================================================
# AGENTE PRINCIPAL (root_agent)
# =============================================================================
root_agent = DirectorConFastPath(
    name="vertex_search_agent",
    sub_agents=[director_final],
)
