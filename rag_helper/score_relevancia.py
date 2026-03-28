import logging
import time
from typing import Iterator, List, Literal, Tuple

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
#                     filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")
logger = logging.getLogger()

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from rag_helper.qwen_reranker import QwenReranker


def generar_scores_relevancia_reranker(datos: List[Tuple[str, str]], no_log: bool = False) -> List[Tuple[str, str, float]]:
    """
    Evalúa el score de relevancia de un documento para responder una pregunta
    utilizando un modelo de Reranking y devuelve un iterador con los resultados.

    Args:
        datos: Una lista de tuplas, donde cada tupla contiene dos cadenas:
               - La primera cadena es la pregunta.
               - La segunda cadena es el contenido del documento.

    Yields:
        Un iterador de tuplas. Cada tupla contiene la pregunta original,
        el contenido del documento y un score de relevancia (float entre 0.0 y 1.0).
    """

    reranker = QwenReranker()
    resultados = reranker.rerank(datos)
    if not no_log:
        for i, (pregunta, documento, score) in enumerate(resultados):
            logger.info(f"Relevancia Local: {score}|  Búsqueda:{pregunta[:30]}|   documento:{documento}")
    return resultados

def generar_scores_relevancia_llm(datos: List[Tuple[str, str]], 
                                  no_log: bool = False,
                                  llm = None
                                  ) -> Iterator[Tuple[str, str, float]]:
    """
    Evalúa el score de relevancia de un documento para responder una pregunta
    utilizando el modelo de lenguaje y devuelve un iterador con los resultados.

    Args:
        datos: Una lista de tuplas, donde cada tupla contiene dos cadenas:
               - La primera cadena es la pregunta.
               - La segunda cadena es el contenido del documento.
        no_log: Si es True, no se registrarán los scores de relevancia en el logger.
                Esto puede ser útil para reducir la cantidad de logs generados durante pruebas o ejecuciones masivas.
                Por defecto es False, lo que significa que los scores se registrarán normalmente.
        llm: El modelo de lenguaje a utilizar para la evaluación de relevancia.

    Yields:
        Un iterador de tuplas. Cada tupla contiene la pregunta original,
        el contenido del documento y un score de relevancia (float entre 0.0 y 1.0).
    """
    # Inicializa el modelo LLM
    # La temperatura se establece en 0 para obtener resultados menos aleatorios.
    if not llm:
        llm = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    # Define la plantilla del prompt para la tarea de puntuación de relevancia.
    # Se le instruye al modelo que devuelva únicamente un número flotante.
    prompt_template = """
    Evalúa la relevancia del siguiente contexto de un documento para responder la pregunta.
    Devuelve un único número de punto flotante entre 0.0 y 1.0, donde 1.0 significa que el contexto es 
    perfectamente relevante y suficiente para responder toda la pregunta, 
    mientras que 0.0 significa que no tiene ninguna relevancia con alguna parte de la pregunta.

    Responde únicamente con el número, sin texto adicional.

    Pregunta: {pregunta}

    Contexto: {contexto}
    """

    # Crea un objeto PromptTemplate a partir de la cadena de la plantilla.
    prompt = PromptTemplate.from_template(prompt_template)

    # Crea la cadena (chain) que combina el prompt y el modelo LLM.
    chain = prompt | llm

    # Itera sobre cada tupla de pregunta y documento en la lista de entrada.
    for pregunta, documento in datos:
        # Invoca la cadena con la pregunta y el documento actuales.
        respuesta_llm = chain.invoke({"pregunta": pregunta, "contexto": documento}).content

        # Intenta convertir la respuesta del LLM a un valor flotante.
        # Se incluye un manejo de errores por si el modelo devuelve algo inesperado.
        score = 0.0
        try:
            # Limpia la respuesta de posibles espacios en blanco y la convierte a float.
            score = float(respuesta_llm.strip())
            if not no_log:
                logger.info(f"Relevancia LLM: {score}|  Búsqueda:{pregunta[:30]}|   documento:{documento}")

        except (ValueError, TypeError):
            # Si la conversión falla, se imprime una advertencia y se mantiene el score en 0.0.
            print(f"Advertencia: No se pudo convertir la respuesta del LLM '{respuesta_llm}' a un float. Usando score de 0.0.")

        yield (pregunta, documento, score)

def convertir_documentos_a_tuplas(pregunta: str, documentos: List[Document]) -> List[Tuple[str, str]]:
    """
    Convierte una lista de objetos Document a una lista de tuplas (pregunta, documento).

    Args:
        documentos: Una lista de objetos Document, cada uno con un atributo 'page_content'.

    Returns:
        Una lista de tuplas, donde cada tupla contiene dos cadenas:
        - La primera cadena es la pregunta.
        - La segunda cadena es el contenido del documento.
    """
    return [(pregunta, doc.page_content) for doc in documentos]

def filtrar_documentos_por_relevancia(
    pregunta: str,
    documentos: List[Document],
    umbral_relevancia: float,
    reranker: Literal["local", "llm"] = "local",
    busqueda_permisiva: bool = False,
    llm_reranker = None,
    k = 0
) -> List[Document]:
    """
    Filtra una lista de documentos, manteniendo solo aquellos cuyo contenido
    es relevante para una pregunta dada, basándose en un umbral de relevancia.

    Args:
        pregunta: La pregunta para la cual se evaluará la relevancia de los documentos.
        documentos: Una lista de objetos Document de LangChain, cada uno con un atributo 'page_content'.
        umbral_relevancia: El score mínimo de relevancia (entre 0.0 y 1.0)
                           para que un documento sea considerado relevante.

    Returns:
        Una lista de objetos Document que tienen un score de relevancia
        igual o superior al umbral_relevancia y ordenada por relevancia decreciente.
    """
    umbral_relevancia_permisivo = 0.1
    if not (0.0 <= umbral_relevancia <= 1.0):
        raise ValueError("El umbral_relevancia debe estar entre 0.0 y 1.0.")

    start_time = time.time()

    # Prepara los datos para la función generar_scores_relevancia
    # Cada tupla contiene (pregunta, contenido_del_documento)
    #datos_para_evaluar = [(pregunta, doc.page_content) for doc in documentos]
    datos_para_evaluar = convertir_documentos_a_tuplas(pregunta, documentos)
    logger.info(f"Documentos a filtrar: {len(documentos)}, Búsqueda:{pregunta[:30]}, Umbral relevancia:{umbral_relevancia}, permisiva:{busqueda_permisiva}")

    documentos_relevantes = []
    if reranker == "local":
        score_documentos = generar_scores_relevancia_reranker(datos_para_evaluar)
    elif reranker == "llm":
        score_documentos = generar_scores_relevancia_llm(datos_para_evaluar, llm=llm_reranker)
    # Usamos enumerate para poder acceder al documento original por su índice
    # junto con su score de relevancia.
    for i, (q, content, score) in enumerate(score_documentos):
        if score >= umbral_relevancia:
            documentos_relevantes.append((documentos[i], score)) # Añade el documento original si es relevante
    
    if busqueda_permisiva and not documentos_relevantes and umbral_relevancia > umbral_relevancia_permisivo:
        logger.info(f"Busqueda no devolvió documentos relevantes, se cambia umbral permisivo '{umbral_relevancia_permisivo}'")
        for i, (q, content, score) in enumerate(score_documentos):
            if score >= umbral_relevancia_permisivo:
                documentos_relevantes.append((documentos[i], score)) # Añade el documento original si es relevante
            
    if len(documentos_relevantes) == 0:
        logger.warning(f"No se encontraron documentos relevantes para la búsqueda: '{pregunta}'")
        return []

    documentos_relevantes = sorted(documentos_relevantes, key=lambda doc: doc[1], reverse=True)

    logger.info(f"Documentos devueltos: {len(documentos_relevantes)}")
    logger.info(f"Tiempo de filtrado: {time.time() - start_time:.2f} segundos")

    for doc, score in documentos_relevantes:
        doc.metadata["score_relevancia"] = score
    
    if k > 0:
        documentos_relevantes = documentos_relevantes[:k]
    
    return [doc for doc, score in documentos_relevantes]

def imprimir_scores_relevancia(scores_relevancia, titulo, longitud_documento=100):
    """
    Prints relevance scores for question-document pairs with formatted output.
    Args:
        scores_relevancia (list): A list of tuples containing (pregunta, doc, score) where:
            - pregunta (str): The question text
            - doc (str): The retrieved document excerpt
            - score (float): The relevance score between 0 and 1
        titulo (str): The title to display at the top of the output
        longitud_documento (int, optional): Maximum length of the document excerpt to display. Defaults to 100.
    Returns:
        None
    Prints:
        Formatted output with title, separator lines, and for each score entry:
        - The question
        - The truncated document excerpt
        - The relevance score (with 2 decimal places) and relevance category:
          * "(Muy Relevante)" if score > 0.75
          * "(Relevante)" if score > 0.5
          * "(Poco relevante)" if score > 0.25
          * "(Nada relevante)" otherwise
    """
    scores_relevancia = list(scores_relevancia)
    print("*"*50, "\n", titulo, "\n", "-" * 50)
    for pregunta in set([pregunta for pregunta, _, _ in scores_relevancia]):
        maximo_score = max(score for p, _, score in scores_relevancia if p == pregunta)
        minimo_score = min(score for p, _, score in scores_relevancia if p == pregunta)
        cantidad_score = len([score for p, _, score in scores_relevancia if p == pregunta])
        promedio = sum(score for p, _, score in scores_relevancia if p == pregunta) / cantidad_score
        logger.info(f"Pregunta: {pregunta}, Maximo score: {maximo_score:.2f}, Minimo score: {minimo_score:.2f}, Promedio score: {promedio:.2f}, Cantidad de scores: {cantidad_score}")
    
        print(f"Pregunta: {pregunta}")
        print(f"Documentos recuperados: {cantidad_score}")
        print(f"Maximo score: {maximo_score:.2f}")
        print(f"Minimo score: {minimo_score:.2f}")
        print(f"Promedio score: {promedio:.2f}")
        for p, doc, score in scores_relevancia:
            if p == pregunta:
                texto_score = f"Score: {score:.2f} - "
                texto_score += "(Muy Relevante)" if score > 0.75 else "(Relevante)" if score > 0.5 else "(Poco relevante)" if score > 0.25 else "(Nada relevante)"
                logger.info(f"Extracto recuperado: {doc}...")
                logger.info(texto_score)
                print(f"Extracto recuperado: {doc[:longitud_documento]}...")
                print(texto_score)
                print("-" * 50)

        print("=" * 50)


def main(args=None):
    # --- Configuración Opcional: Cargar variables de entorno desde un archivo .env ---
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # Crear Documentos de ejemplo
    documento1 = Document(page_content="París es la capital de Francia y una ciudad hermosa.")
    documento2 = Document(page_content="El río Sena fluye a través de París.")
    documento3 = Document(page_content="Berlín es la capital de Alemania.")
    documento4 = Document(page_content="Los beneficios de la energía solar incluyen la reducción de la huella de carbono y el ahorro de costos a largo plazo.")
    documentos_de_prueba = [documento1, documento2, documento3, documento4]

    pregunta_filtrado = "¿Cuál es la capital de Francia?"
    umbral = 0.8

    documentos_filtrados = filtrar_documentos_por_relevancia(
        pregunta_filtrado,
        documentos_de_prueba,
        umbral,
        reranker="local",
    )

    print(f"Pregunta: '{pregunta_filtrado}'")
    print(f"Umbral de Relevancia: {umbral}")
    print("\nDocumentos Relevantes Encontrados:")
    if documentos_filtrados:
        for i, doc in enumerate(documentos_filtrados):
            print(f"  {i+1}. {doc.page_content}")
    else:
        print("  Ningún documento cumple con el umbral de relevancia.")

    pregunta_filtrado_energia = "¿Cuáles son las ventajas de la energía solar?"
    umbral_energia = 0.5
    documentos_filtrados_energia = filtrar_documentos_por_relevancia(
        pregunta_filtrado_energia,
        documentos_de_prueba,
        umbral_energia,
        reranker="llm",
    )
    print(f"\nPregunta: '{pregunta_filtrado_energia}'")
    print(f"Umbral de Relevancia: {umbral_energia}")
    print("\nDocumentos Relevantes Encontrados:")
    if documentos_filtrados_energia:
        for i, doc in enumerate(documentos_filtrados_energia):
            print(f"  {i+1}. {doc.page_content}")
    else:
        print("  Ningún documento cumple con el umbral de relevancia.")

# --- Código para Pruebas ---
if __name__ == "__main__":
    main()
