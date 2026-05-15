import sys, os
import requests
import sqlalchemy
from google.adk.agents import Agent, BaseAgent, SequentialAgent, ParallelAgent
from google.adk.tools import google_search
from typing import AsyncGenerator
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

# --- CONEXIÓN SQL (lazy pool) ---
db_pool = None

def get_db_pool():
    global db_pool
    if db_pool is None:
        def getconn():
            connector = Connector()
            return connector.connect(
                GOOGLE_BD_DIRECCION,
                "pg8000",
                user=GOOGLE_BD_USER,
                password=GOOGLE_BD_PASSWORDBD,
                db=GOOGLE_BD_BD,
                ip_type=IPTypes.PUBLIC
            )
        db_pool = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
            pool_size=5,
            max_overflow=2,
            pool_timeout=30,
            pool_recycle=1800
        )
    return db_pool

# --- HERRAMIENTAS ---

def ejecutar_consulta_sql_dinamica(query: str) -> str:
    """Ejecuta una consulta SQL SELECT para obtener datos estructurados de RRHH desde PostgreSQL."""
    if not (query.strip().lower().startswith("select") or query.strip().lower().startswith("with")):
        return "Error: Solo se permiten consultas de lectura (SELECT o WITH)."
    pool = get_db_pool()
    with pool.connect() as conn:
        try:
            result = conn.execute(sqlalchemy.text(query))
            rows = result.fetchall()
            colnames = result.keys()
            formatted_res = [dict(zip(colnames, row)) for row in rows]
            return str(formatted_res) if formatted_res else "No se encontraron resultados."
        except Exception as e:
            return f"Error en SQL: {str(e)}"

def vertex_ai_search(query: str) -> str:
    """
    Realiza una búsqueda semántica en los documentos/PDFs de RRHH.
    Usa summary_spec para obtener una respuesta resumida generada por IA anclada en los documentos,
    más los fragmentos extractivos de cada documento relevante.
    """
    try:
        client = discoveryengine.SearchServiceClient(
            client_options={"api_endpoint": "us-discoveryengine.googleapis.com"}
        )
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

        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
            content_search_spec=content_search_spec,
        )
        response = client.search(request)

        parts = []

        # 1. Resumen semántico generado por Vertex AI (respuesta directa a la pregunta)
        if hasattr(response, "summary") and response.summary and response.summary.summary_text:
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
                if 'extractive_answers' in doc.derived_struct_data:
                    answers = doc.derived_struct_data['extractive_answers']
                    content = " [...] ".join([a.get('content', '') for a in answers if a.get('content')])
                elif 'extractive_segments' in doc.derived_struct_data:
                    segments = doc.derived_struct_data['extractive_segments']
                    content = " [...] ".join([s.get('content', '') for s in segments if s.get('content')])
                elif 'snippets' in doc.derived_struct_data:
                    snippets = doc.derived_struct_data['snippets']
                    content = " [...] ".join([s.get('snippet', '') for s in snippets if s.get('snippet')])
            if content:
                doc_parts.append(f"ID_Documento: {doc_id} | Título: {title}\n{content}")

        if doc_parts:
            parts.append("DOCUMENTOS RELEVANTES:\n" + "\n\n".join(doc_parts))

        return "\n\n".join(parts) if parts else "No se encontraron documentos relevantes para esta consulta."
    except Exception as e:
        return f"Error en la búsqueda de Vertex AI: {str(e)}"


def navegar_software(id_trabajador: str, id_documento: str) -> dict:
    """
    Genera el comando para que el software abra el expediente del trabajador y el documento.
    Args:
        id_trabajador: id_recurso del trabajador (carpeta padre).
        id_documento: id_recurso del documento específico (hijo).
    """
    return {
        "action": "OPEN_EXPEDIENTE",
        "worker_id": id_trabajador,
        "document_id": id_documento,
        "url": f"/explorer/{id_trabajador}"
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
            json={
                "correo_destino": correo_destino,
                "asunto": asunto,
                "cuerpo": cuerpo
            },
            timeout=30
        )
        response.raise_for_status()
        return f"Correo enviado exitosamente a {correo_destino}."
    except requests.exceptions.HTTPError as e:
        return f"Error al enviar el correo (HTTP {response.status_code}): {response.text}"
    except Exception as e:
        return f"Error al enviar el correo: {str(e)}"

# =============================================================================
# SUB-AGENTE 1: ANALISTA SQL
# Responsable de consultar datos estructurados de la base de datos PostgreSQL.
# =============================================================================
analista_sql = Agent(
    name="analista_sql",
    model="gemini-2.5-flash",
    tools=[ejecutar_consulta_sql_dinamica],
    instruction="""
    Eres el Analista de Base de Datos de RRHH. Tu única responsabilidad es consultar la base de datos PostgreSQL cuando se requieran datos estructurados.

    ESQUEMA DE TABLAS:
    - recurso (id_recurso, titulo, id_recurso_padre, id_version_activa, id_tipo_recurso, estado)
    - version (id_version, fecha_vencimiento, metadata, id_recurso, resumen)
    - tipo_recurso (id_tipo_recurso, estructura, nombre, descripcion)

    REGLAS DE NEGOCIO:
    - id_tipo_recurso para expediente del trabajador: '36e88186-f873-40cd-a1eb-f4bc3dd18af1'. El título es el nombre del trabajador, utiliza en cada palabra el operador % en la consulta que hagas, ya que puede que me den el primer nombre y primer apellido y el formato del titulo es PRIMER APELLIDO SEGUNDO APELLIDO PRIMER NOMBRE SEGUNDO NOMBRE.
    - id_tipo_recurso para cumpleaños: '883bcc87-e00d-4abb-b7b0-bc8ae6211d22', metadata campo: fecha_nacimiento.
    - id_tipo_recurso para contratos: '139be00e-2d43-4093-b9f8-e600b405efe3', metadata campo: fecha_inicio. Para cada trabajador hay dos fechas especificas la fecha de ingreso y la fecha de contratación. La fecha de ingreso la encuentras en la version más antigua del contrato laboral (fecha_creacion en la tabla version) consultado el campo fecha_inicio de su metadata. Por otro lado, la fecha de contratación la encuentras en la primera versión que la metadata (segun la conlumna fecha_creacion) donde en el campo 'tipo_contrato' sea 'Contrato determinado' o 'Contrato indeterminado', si es de otro tipo, no es la fecha de contratacion.
    - Documentos vencidos tienen fecha_vencimiento < CURRENT_DATE.
    - IMPORTANTE: La columna 'metadata' (JSONB) se encuentra ÚNICAMENTE en la tabla 'version'. La tabla 'recurso' NO TIENE columna 'metadata'. Si necesitas buscar, filtrar o consultar por metadata (como 'nombre', 'apellido', 'fecha_nacimiento' o 'fecha_inicio'), SIEMPRE debes hacer un JOIN entre 'recurso' y 'version' usando 'recurso.id_version_activa = version.id_version'.
    - Para buscar trabajadores por nombre, puedes usar el título del recurso con LIKE % o hacer el JOIN con version y buscar en v.metadata->>'nombre'.
    - Solo usa registros con estado 'activo'.
    - SIEMPRE recupera 'id_recurso' y 'id_recurso_padre' para permitir la navegación.
    - DOCUMENTOS DEL TRABAJADOR: Cuando el usuario pregunte por los documentos de un expediente específico, DEBES usar la siguiente consulta como plantilla. Es CRÍTICO que reemplaces la etiqueta '[AQUI_VA_EL_UUID_DEL_EXPEDIENTE]' por el UUID real (id_recurso) del expediente del trabajador (manteniendo las comillas simples, ej. '123e4567-...'). No uses variables de enlace, inserta el UUID directamente en el texto del SQL en los DOS lugares donde aparece la etiqueta.
      Ejemplo de query a utilizar:
      WITH RECURSIVE recursos_a_listar AS (
            -- Seleccionamos solo los hijos directos del padre
            SELECT * FROM recurso 
            WHERE id_recurso_padre = '[AQUI_VA_EL_UUID_DEL_EXPEDIENTE]'
            AND estado IN ('activo', 'inactivo')
        ),
        fechas_expediente AS (
            -- Lógica simplificada para obtener fechas de contratos de los hijos
            SELECT 
                r.id_recurso_padre,
                MIN((NULLIF(v.metadata->>'fecha_inicio', ''))::date) as fecha_inicio_final
            FROM recurso r
            INNER JOIN version v ON r.id_version_activa = v.id_version
            WHERE r.id_recurso_padre = '[AQUI_VA_EL_UUID_DEL_EXPEDIENTE]' 
            AND r.id_tipo_recurso = '139be00e-2d43-4093-b9f8-e600b405efe3'
            GROUP BY r.id_recurso_padre
        )
        SELECT DISTINCT ON (r.id_recurso)
            r.id_recurso, 
            r.titulo, 
            r.estado, 
            r.id_recurso_padre,
            eo.nombre AS nombre_estructura, 
            tr.plazo_vencimiento,
            tr.nombre AS tipo_recurso, 
            tr.id_tipo_recurso, 
            v.id_version,
            CASE 
                WHEN tr.nombre = 'Expediente' THEN 
                    (v.metadata::jsonb || jsonb_strip_nulls(jsonb_build_object(
                        'fecha_inicio', (SELECT fecha_inicio_final FROM fechas_expediente WHERE id_recurso_padre = r.id_recurso_padre)
                    )))
                ELSE v.metadata::jsonb
            END as metadata,
            v.fecha_creacion, 
            v.fecha_vencimiento,
            se.nombre AS sub_etiqueta,
            et.nombre AS etiqueta
        FROM recursos_a_listar r
        LEFT JOIN estructura_organizativa eo ON r.id_estructura_organizativa = eo.id_estructura_organizativa
        LEFT JOIN version v ON r.id_version_activa = v.id_version
        LEFT JOIN tipo_recurso tr ON r.id_tipo_recurso = tr.id_tipo_recurso  
        LEFT JOIN sub_etiqueta se ON tr.id_sub_etiqueta = se.id_sub_etiqueta
        LEFT JOIN etiqueta et ON se.id_etiqueta = et.id_etiqueta
        ORDER BY r.id_recurso, r.titulo;
      IMPORTANTE: Para leer la metadata de cualquier documento, debes acceder ÚNICAMENTE a su versión activa ('a.id_version_activa = v.id_version'). El ÚNICO CASO donde puedes consultar el historial completo de la tabla version es con los Contratos Laborales para distinguir la fecha de contratación vs la fecha de ingreso.
    - CORREO DEL TRABAJADOR: El campo 'correo' está en la metadata del expediente (id_tipo_recurso '36e88186-f873-40cd-a1eb-f4bc3dd18af1'). Si el usuario quiere enviar un correo a un trabajador, DEBES recuperar este campo con una query como:
      SELECT r.titulo, v.metadata->>'correo' AS correo FROM recurso r JOIN version v ON r.id_version_activa = v.id_version WHERE r.id_tipo_recurso = '36e88186-f873-40cd-a1eb-f4bc3dd18af1' AND r.titulo ILIKE '%nombre_trabajador%' AND r.estado = 'activo'

    CÁLCULO DE VACACIONES (Art. 190 LOTTT):
    Año 1: 15 días, Año 2: 16, Año 3: 17... máx 30 por año. Sumatoria hasta la fecha actual, a partir de su fecha de contratación.

    Si la pregunta no requiere SQL (ej. saludo), responde con un resumen vacío y pasa al siguiente agente.
    Responde en español. Sin asteriscos (*).
    """,
)

# =============================================================================
# SUB-AGENTE 2: DOCUMENTAL RAG
# Responsable de buscar información en documentos/PDFs mediante Vertex AI Search.
# =============================================================================
documental_rag = Agent(
    name="documental_rag",
    model="gemini-2.5-flash",
    tools=[vertex_ai_search],
    instruction="""
    Eres el Especialista en Documentos de RRHH. Tu responsabilidad es buscar información en los PDFs y expedientes digitalizados (texto contenido dentro de los documentos).

    REGLAS DE USO:
    - Dado que trabajas en paralelo con el Analista SQL, no conoces los datos estructurados que él extrae. Enfócate ÚNICAMENTE en buscar dentro del texto de los documentos (ej. cláusulas de contratos, contenido de certificados o informes).
    - Si la pregunta del usuario es puramente sobre "listar los documentos que tiene un trabajador" o sobre "fechas de ingreso", NO inventes documentos; responde de manera concisa o con un resumen vacío para dejar que el agente SQL (que tiene la estructura real) responda. Use 'vertex_ai_search' principalmente para dudas sobre políticas, contenido de texto o cuando se busque explícitamente dentro de PDFs.

    CÓMO INTERPRETAR LOS RESULTADOS:
    La herramienta devuelve dos secciones:
    1. RESUMEN SEMÁNTICO: Es una respuesta directa generada por IA a partir de los documentos encontrados. Prioriza esta sección para responder al usuario.
    2. DOCUMENTOS RELEVANTES: Son los fragmentos exactos de los documentos fuente con su ID_Documento (UUID). Usa estos fragmentos para complementar el resumen o cuando se necesite navegar al documento.

    EXTRACCIÓN DE IDs:
    - El ID_Documento en los resultados es el 'id_version' del documento en el sistema.
    - REGLA DE IDENTIDAD: Identifica al trabajador con Nombre y Cédula cuando sea posible.

    Responde en español. Sin asteriscos (*).
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
    "busca en internet", "busca en google", "busca online", "busca en la web",
    "información actualizada sobre", "noticias de", "busca afuera",
    "consulta en internet", "consúltalo en internet", "búsqueda web",
    "buscar en internet", "qué dice internet", "google esto",
]

buscador_web = Agent(
    name="buscador_web",
    model="gemini-2.5-flash",
    tools=[google_search],
    instruction="""
    Eres el Especialista en Búsqueda Web. El usuario ha pedido explícitamente buscar información en internet.
    - Haz la búsqueda más específica y relevante posible según la solicitud.
    - Resume los resultados de forma clara y cita las fuentes.
    - Responde en español. Sin asteriscos (*).
    """,
)

# =============================================================================
# AGRUPACIÓN EN PARALELO: ANALISTA + RAG
# Ambos agentes siempre trabajan juntos pero en paralelo.
# =============================================================================
investigadores_rrhh = ParallelAgent(
    name="investigadores_rrhh",
    sub_agents=[analista_sql, documental_rag],
    description="Analista SQL y Documental RAG trabajando juntos siempre en paralelo."
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
    name="buscador_web_condicional",
    sub_agents=[buscador_web]
)

orquestador = ParallelAgent(
    name="orquestador",
    sub_agents=[investigadores_rrhh, buscador_web_condicional],
    description="Orquestador que ejecuta la búsqueda en RRHH y (opcionalmente) la web en paralelo."
)

# =============================================================================
# SUB-AGENTE FINAL: DIRECTOR 
# Responsable de sintetizar la información y ejecutar acciones de navegación o correos.
# =============================================================================
director_final = Agent(
    name="director_final",
    model="gemini-2.5-flash",
    tools=[navegar_software, enviar_correo],
    instruction="""
    Eres el Director de RRHH de Abside y el único punto de contacto con el usuario. Recibes la información ya procesada en paralelo por los agentes anteriores y tu trabajo es:

    1. SINTETIZAR la información, SACAR LAS CUENTAS (ej. cálculos de vacaciones) y presentar la información de forma clara y ejecutiva al usuario.
    2. PRIORIZAR AL ANALISTA SQL: El Analista SQL es la fuente de la verdad para saber la estructura y qué documentos exactos tiene un trabajador. Si el Analista SQL trae una lista de documentos, PRIORIZA SIEMPRE esa respuesta. Usa la respuesta del Documental RAG SOLO para complementar si el usuario preguntó por el contenido de texto interno de un documento (ej. qué dice una cláusula).
    3. NAVEGAR al documento si el usuario usó verbos de acción como: "búscame", "busca", "ubícame", "ubica", "encuentra", "abre" o "consigue".
    4. GESTIONAR EL ENVÍO DE CORREOS con un flujo obligatorio de confirmación de 2 pasos.

    LÓGICA DE NAVEGACIÓN:
    - Si hay intención de navegación, usa 'navegar_software' con:
        * id_trabajador = id_recurso_padre (carpeta del trabajador)
        * id_documento = id_recurso (documento específico)
    - Si NO hay intención de navegación, responde solo con texto sin mostrar IDs internos.

    REGLA DE ORO PARA ENVIAR CORREOS:
    Bajo NINGUNA CIRCUNSTANCIA puedes llamar a la herramienta 'enviar_correo' sin antes mostrarle al usuario el correo preliminar y recibir su confirmación explícita. SIEMPRE debes seguir este flujo:

    PASO 1 - PRESENTAR BORRADOR (cuando el usuario pide enviar un correo):
    - Redacta un correo profesional y conciso basado en la información disponible.
    - Muestra la versión preliminar al usuario con este formato exacto:

      Destinatario: [correo del trabajador]
      Asunto: [asunto propuesto]
      ---
      [cuerpo del correo]
      ---

      ¿Confirmas el envío de este correo preliminar? Responde "sí" para enviarlo o indícame si deseas realizar algún cambio.

    - DETENTE AQUÍ. NO llames a la herramienta 'enviar_correo' todavía.

    PASO 2 - ENVIAR (SOLO TRAS CONFIRMACIÓN):
    - Únicamente si el usuario te responde explícitamente "sí", "confirmo", "enviar" u "ok" al borrador que le mostraste, ENTONCES procederás a llamar a la herramienta 'enviar_correo'.
    - Si el usuario pide cambios, ajusta el borrador y vuelve al Paso 1.

    FORMATO GENERAL:
    - Responde en español.
    - Sin asteriscos (*).
    - Identifica siempre al trabajador por su Título (nombre).
    - Respuesta ejecutiva y concisa.
    - SECCIÓN DE FUENTES: Al final de cada respuesta, incluye siempre una sección titulada "Fuentes:" listando de dónde obtuviste la información (ej. "PostgreSQL: Expediente de [Nombre]", "PDF: [Título del documento]", "Google Search: [Query]").
    """,
)

# =============================================================================
# AGENTE PRINCIPAL (root_agent)
# =============================================================================
root_agent = SequentialAgent(
    name="vertex_search_agent",
    sub_agents=[orquestador, director_final],
    description="Orquestador inteligente que enruta en paralelo y luego sintetiza a través del director.",
)