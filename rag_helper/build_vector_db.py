import logging
import math
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from ragas import EvaluationDataset
from transformers import AutoTokenizer

from .custom_docling_loader import CustomDoclingLoader, ExportType

logger = logging.getLogger()

OLLAMA_DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_VECTOR_SIZE = 1024
OLLAMA_SMALL_EMBEDDING_MODEL = "all-minilm:latest"
SMALL_VECTOR_SIZE = 384
DENSE_VECTOR_NAME = "denso"
SPARSE_VECTOR_NAME = "disperso"
MAX_TOKENS_OLLAMA_DEFAULT_EMBEDDING = 8192
MAX_TOKENS_SMALL_EMBEDDING = 256
HF_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HF_SMALL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_embeddings(model:str) -> OllamaEmbeddings:
    """Crea una instancia de embeddings de Ollama.

    Args:
        model (str): El nombre del modelo de embeddings a utilizar en Ollama.

    Returns:
        OllamaEmbeddings: Una instancia de Langchain OllamaEmbeddings.
    """
    return OllamaEmbeddings(model=model)

def cargar_documentos(ruta_archivos: list[str],
                      tipo_exportacion: Literal["doc_chunks", "custom_chunks"] = "doc_chunks",
                      tokenizer_hf_model: str = HF_DEFAULT_EMBEDDING_MODEL,
                      max_chunk_token_size: int = MAX_TOKENS_OLLAMA_DEFAULT_EMBEDDING,
                      tipo_ocr_pdf: Literal["local", "vlm"] = "local",
                      num_threads_ocr_pdf_local: int = 4,
                      url_ocr_pdf_vlm: str = "",
                      api_key_ocr_pdf_vlm: str = "",
                      model_name_ocr_pdf_vlm: str = ""
                      )->list[Document]:
    """Carga documentos desde una lista de rutas de archivos utilizando CustomDoclingLoader.
     Permite configurar el tipo de exportación, el modelo de tokenización, el tamaño máximo 
     de los chunks, y opciones para OCR de PDFs tanto local como a través de VLM. 
     Devuelve una lista de objetos Document listos para ser procesados y almacenados en un vector store.

    Args:
        ruta_archivos (list[str]): Lista de rutas de archivos a cargar. Cada ruta debe ser un string que apunte a un archivo válido.
         Puede incluir urls, archivos locales PDFs, o cualquier formato soportado por CustomDoclingLoader
        tipo_exportacion (Literal[doc_chunks, custom_chunks], optional): Especifica cómo se exportarán los documentos 
            doc_chunks es el default de Docling y custom_chunks tiene un chunker personalizado. Por defecto es "doc_chunks".
        tokenizer_hf_model (str, optional): Modelo de tokenización a utilizar para la separación de los chunks. Por defecto es HF_DEFAULT_EMBEDDING_MODEL.
        max_chunk_token_size (int, optional): Tamaño máximo de tokens por chunk. Por defecto es MAX_TOKENS_OLLAMA_DEFAULT_EMBEDDING.
        tipo_ocr_pdf (Literal[local, vlm], optional): Tipo de OCR a utilizar para PDFs, local usa la libreria EasyOCR y vlm un endpoint de VLM. Por defecto es "local".
        num_threads_ocr_pdf_local (int, optional): Número de hilos a utilizar para el OCR local. Por defecto es 4.
        url_ocr_pdf_vlm (str, optional): URL del endpoint del VLM a usar para el OCR. Por defecto es "".
        api_key_ocr_pdf_vlm (str, optional): API key del endpoint del VLM a usar para el OCR. Por defecto es "".
        model_name_ocr_pdf_vlm (str, optional): Nombre del modelo del VLM a usar para el OCR. Por defecto es "".

    Returns:
        list[Document]: Una lista de objetos Document que representan los documentos cargados y procesados, listos para ser almacenados en un vector store.
    """
    if tipo_exportacion == "doc_chunks":
        export_type = ExportType.DOC_CHUNKS
    elif tipo_exportacion == "custom_chunks":
        export_type = ExportType.MARKDOWN
    else:
        raise ValueError(f"Tipo de exportación no válido: {tipo_exportacion}")
    
    documentos=[]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_model)

    for ruta_archivo in ruta_archivos:
        loader = CustomDoclingLoader(
            file_path=ruta_archivo,
            export_type=export_type,
            contextualize_chunk=False,
            tokenizer=tokenizer,
            max_tokens=max_chunk_token_size,
            num_threads_ocr_pdf_local = num_threads_ocr_pdf_local,
            tipo_ocr_pdf = tipo_ocr_pdf,
            url_ocr_pdf_vlm = url_ocr_pdf_vlm,
            api_key_ocr_pdf_vlm = api_key_ocr_pdf_vlm,
            model_name_ocr_pdf_vlm = model_name_ocr_pdf_vlm
        )
        documentos += loader.load()
        logger.info(f"Se cargaron {len(documentos)} documentos desde {ruta_archivo}")
        
    return documentos

def dividir_lista(lista_original: list, tamano_sublista: int):
    """
    Divide una lista en sublistas de un tamaño específico.

    Args:
        lista_original (list): La lista que se desea dividir.
        tamano_sublista (int): El número de elementos en cada sublista.

    Returns:
        list: Una lista de sublistas.
    """
    # Verifica si el tamaño de la sublista es válido
    if tamano_sublista <= 0:
        raise ValueError(f"El tamaño de la sublista debe ser un número entero positivo: {tamano_sublista}")

    lista_dividida = []
    # Itera sobre la lista original en pasos de 'tamano_sublista'
    for i in range(0, len(lista_original), tamano_sublista):
        # Utiliza slicing para obtener la sublista
        sublista = lista_original[i:i + tamano_sublista]
        lista_dividida.append(sublista)
        
    return lista_dividida

def crear_cliente_qdrant(url = None, api_key = None ) -> QdrantClient:
    """Crea una instancia del cliente QDrant
    Args:
        url: la ruta a tu instancia de QDrant, si es None apunta a la variable de entorno "QDRANT_CLUSTER_URI"
        api_key: la API Key de Qdrant, si es None apunta a la variable de entorno "QDRANT_API_KEY"

    Returns:
        QdrantClient: Nueva intancia del cliente de Qdrant
    """    

    # Cargar las variables de entorno desde el archivo .env (aca debe ir el API Key del Proveedor del LLM)
    load_dotenv()

    if url is None:
        url = os.environ["QDRANT_CLUSTER_URI"] 
    if api_key is None:
        api_key = os.environ["QDRANT_API_KEY"]

    return QdrantClient(
        url=url,
        api_key=api_key,
    )

def crear_vector_store(
        nombre_coleccion: str,
        cliente_qdrant = crear_cliente_qdrant(),
        embeddings = None,
        modo_recuperacion: Literal["dense", "sparse", "hybrid"] = "hybrid",
        nombre_vector_denso = "denso",
        embeddings_disperso = None,
        nombre_vector_disperso = "disperso") -> QdrantVectorStore:
    """
    Crea un vector store de Qdrant con soporte para modos de recuperación denso, disperso o híbrido.

    Args:
        nombre_coleccion (str): Nombre de la colección de Qdrant a crear o usar.
        cliente_qdrant: Instancia del cliente de Qdrant. Por defecto, se crea uno nuevo con `crear_cliente_qdrant()`.
        embeddings: Modelo de embedding denso. Si es None y el modo no es "sparse", 
                    se usará por defecto OLLAMA_DEFAULT_EMBEDDING_MODEL. Por defecto es None.
        modo_recuperacion (Literal["dense", "sparse", "hybrid"]): Modo de recuperación para el vector store.
                                                                    Por defecto es "hybrid".
        nombre_vector_denso (str): Nombre del campo de vector denso en Qdrant. Por defecto es "denso".
        embeddings_disperso: Modelo de embedding disperso. Por defecto es `FastEmbedSparse` con el modelo BM25.
                            Por defecto es None.
        nombre_vector_disperso (str): Nombre del campo de vector disperso en Qdrant. 
                                        Por defecto es "disperso".
    Returns:
        QdrantVectorStore: Una instancia configurada de QdrantVectorStore con el modo de recuperación 
                            y los modelos de embedding especificados.
    """


    if embeddings is None and modo_recuperacion != "sparse":
        embeddings = build_embeddings(OLLAMA_DEFAULT_EMBEDDING_MODEL)
        
    if embeddings_disperso is None:
        embeddings_disperso = FastEmbedSparse(model_name="Qdrant/bm25")

    if modo_recuperacion == "hybrid":
        modo_recuperacion_qdrant = RetrievalMode.HYBRID
    elif modo_recuperacion == "dense":
        modo_recuperacion_qdrant = RetrievalMode.DENSE
    elif modo_recuperacion == "sparse":
        modo_recuperacion_qdrant = RetrievalMode.SPARSE

    return QdrantVectorStore(
        client=cliente_qdrant,
        collection_name=nombre_coleccion,
        embedding=embeddings,
        retrieval_mode=modo_recuperacion_qdrant,
        sparse_embedding=embeddings_disperso,
        vector_name=nombre_vector_denso,
        sparse_vector_name=nombre_vector_disperso,
    )    

def agregar_documentos_a_vector_store(
                                vector_store, 
                                documents: list[Document],
                                ):
    """
    Agrega documentos a un vector store en lotes de 100.
    Args:
        vector_store: Instancia del vector store donde se almacenarán los documentos.
        documents (list[Document]): Lista de documentos a agregar al vector store.
    Returns:
        None
    Note:
        Los documentos se dividen en sublistas de 100 elementos para procesamiento por lotes.
        Se registra un mensaje de información para cada lote procesado.
    """
    docs = dividir_lista(documents, 100)
    for sublista_documentos in docs:
        logger.info(f"Agregando {len(sublista_documentos)} documentos a colección {vector_store.collection_name}.")
        vector_store.add_documents(documents=sublista_documentos)

def crear_coleccion(
                    nombre_coleccion: str, 
                    modo_recuperacion: Literal["dense", "sparse", "hybrid"] = "hybrid",
                    nombre_vector_denso = DENSE_VECTOR_NAME,
                    nombre_vector_disperso = SPARSE_VECTOR_NAME, 
                    dimensiones_vector_denso = DEFAULT_VECTOR_SIZE,
                    recrear_coleccion: bool = False,
                    ) -> QdrantClient:
    """Crea o recupera una colección de Qdrant con configuraciones de vectores específicas.

    Esta función establece una conexión con Qdrant y crea una nueva colección
    con vectores densos, dispersos o ambos, dependiendo del modo de recuperación.
    Si la colección ya existe y `recrear_coleccion` es True, será eliminada y
    recreada.

    Args:
        nombre_coleccion (str): El nombre de la colección a crear o recuperar.
        modo_recuperacion (Literal["dense", "sparse", "hybrid"], optional): 
            El modo de recuperación que determina qué tipos de vectores configurar.
            - "dense": Solo vectores densos.
            - "sparse": Solo vectores dispersos.
            - "hybrid": Vectores densos y dispersos.
            Por defecto es "hybrid".
        nombre_vector_denso (str, optional): 
            El nombre del campo del vector denso. Por defecto es DENSE_VECTOR_NAME.
        nombre_vector_disperso (str, optional): 
            El nombre del campo del vector disperso. Por defecto es SPARSE_VECTOR_NAME.
        dimensiones_vector_denso (int, optional): 
            La dimensionalidad de los vectores densos. Por defecto es DEFAULT_VECTOR_SIZE.
        recrear_coleccion (bool, optional): 
            Si es True y la colección existe, será eliminada y recreada.
            Por defecto es False.

    Returns:
        QdrantClient: Una instancia autenticada del cliente de Qdrant conectada a la colección.

    Raises:
        Exception: If the Qdrant client cannot be created or collection operations fail.

    Note:
        - Registra un mensaje cuando una colección es eliminada debido a `recrear_coleccion=True`.
        - Registra un mensaje cuando una nueva colección es creada exitosamente.
    """
    client = crear_cliente_qdrant()
    coleccion_existe = client.collection_exists(nombre_coleccion)
    if recrear_coleccion and coleccion_existe:
        client.delete_collection(
            collection_name=nombre_coleccion,
        )
        coleccion_existe = False
        logger.info(f"Se eliminó la colección '{nombre_coleccion}'")

    if not coleccion_existe:
        args_coleccion = {}

        if modo_recuperacion == "hybrid" or modo_recuperacion == "dense":
            args_coleccion["vectors_config"] = {
                nombre_vector_denso: models.VectorParams(
                    size=dimensiones_vector_denso,
                    distance=models.Distance.COSINE
                )
            }

        if modo_recuperacion == "hybrid" or modo_recuperacion == "sparse":
            args_coleccion["sparse_vectors_config"] = {nombre_vector_disperso: models.SparseVectorParams()}

        client.create_collection(
            collection_name=nombre_coleccion,
            **args_coleccion,
        )
        logger.info(f"Se creó la colección '{nombre_coleccion}', con vector denso '{nombre_vector_denso}' y vector disperso '{nombre_vector_disperso}'")
    return  client

def poblar_coleccion(
                    nombre_coleccion: str, 
                    documents: list[Document],
                    modo_recuperacion: Literal["dense", "sparse", "hybrid"] = "hybrid",
                    nombre_vector_denso = DENSE_VECTOR_NAME,
                    embeddings_disperso = None,
                    nombre_vector_disperso = SPARSE_VECTOR_NAME, 
                    embedding_model = OLLAMA_DEFAULT_EMBEDDING_MODEL,
                    dimensiones_vector_denso = DEFAULT_VECTOR_SIZE,
                    recrear_coleccion: bool = False,
                    ) -> QdrantVectorStore:
    """Puebla una colección de vectores de Qdrant con documentos usando recuperación densa, dispersa o híbrida.

    Crea o recupera una colección de vector store, genera los embeddings para los documentos proporcionados
    y los agrega a la colección para capacidades de búsqueda semántica.

    Args:
        nombre_coleccion (str): Nombre de la colección de vectores a poblar.
        documents (list[Document]): Lista de documentos para agregar al vector store.
        modo_recuperacion (Literal["dense", "sparse", "hybrid"], optional): 
            Modo de recuperación a utilizar. Por defecto es "hybrid".
        nombre_vector_denso (str, optional): Nombre del vector Denso a usar. 
            Por defecto es DENSE_VECTOR_NAME.
        embeddings_disperso (optional): Nombre del modelo de Embeddings Disperso a usar en Ollama. Por defecto es None.
        nombre_vector_disperso (str, optional): Nombre del Vector Disperso a usar. 
            Por defecto es SPARSE_VECTOR_NAME.
        embedding_model (str, optional): Nombre del modelo de Embeddings Denso a usar en Ollama. 
            Por defecto es OLLAMA_DEFAULT_EMBEDDING_MODEL.
        dimensiones_vector_denso (int, optional): Dimensiones para el vector denso. 
            Por defecto es DEFAULT_VECTOR_SIZE.
        recrear_coleccion (bool, optional): Si se debe recrear la colección si ya existe. 
            Por defecto es False.

    Returns:
        QdrantVectorStore: Objeto de vector store configurado con la colección poblada.
    """
    client = crear_coleccion(
        nombre_coleccion,
        modo_recuperacion=modo_recuperacion,
        nombre_vector_denso=nombre_vector_denso,
        nombre_vector_disperso=nombre_vector_disperso,
        dimensiones_vector_denso=dimensiones_vector_denso,
        recrear_coleccion=recrear_coleccion,
    )

    embeddings = build_embeddings(embedding_model)
    logger.info(f"Se aplicó vector denso con modelo de embeddings '{embedding_model}' y dimensiones: {dimensiones_vector_denso}")

    vector_store = crear_vector_store(
        nombre_coleccion,
        cliente_qdrant = client,
        embeddings = embeddings,
        modo_recuperacion=modo_recuperacion,
        nombre_vector_denso=nombre_vector_denso,
        embeddings_disperso = embeddings_disperso,
        nombre_vector_disperso=nombre_vector_disperso,
    )
    
    agregar_documentos_a_vector_store(vector_store, documents)
    return vector_store

def buscar_documentos_similares(vector_store, query: str, k: int = 3, imprimir_resultados: bool = True, longitud_resumen: int = 100):
    """Busca documentos similares a una consulta en un vector store.

    Args:
        vector_store: La instancia del vector store donde se realizará la búsqueda.
        query (str): La consulta para buscar documentos similares.
        k (int, optional): El número de documentos similares a devolver. Por defecto es 3.
        imprimir_resultados (bool, optional): Si es True, imprime los resultados en la consola. Por defecto es True.
        longitud_resumen (int, optional): La longitud máxima del resumen del contenido de la página a imprimir. Por defecto es 100.

    Returns:
        list[Document]: Una lista de documentos similares encontrados.
    """
    results = vector_store.similarity_search(query, k=k)

    i=1
    if imprimir_resultados:
        print("\n==============================================================\n", 
            "Query:", query, "\n------------------------------------------")
        
    for res in results:
        if imprimir_resultados:
            print(f"*{i} - id:{res.metadata['_id']}\n{res.page_content[:longitud_resumen]}...")
            print("------------------------------------------")
        logger.info(f"*{i} - id:{res.metadata['_id']} - Documento:\n{res.page_content}")
        i+=1

    return results

def format_docs(relevant_docs):
    """Formatea una lista de documentos relevantes en un string concatenado."""
    return "\n\n".join(doc.page_content for doc in relevant_docs)

def crear_dataset_evaluacion(vector_store, 
                             llm, 
                             consultas_por_evaluar: list[str] , 
                             respuestas_esperadas: list[str] , 
                             respuestas_generadas: list[str]  = [],
                             max_docs_recuperados: int = 3,
    )-> EvaluationDataset:
    """Crea un dataset de evaluación para RAG utilizando consultas, respuestas esperadas y respuestas generadas.
    Args:
    vector_store: El vector store donde se buscarán los documentos relevantes para cada consulta.
    llm: El modelo de lenguaje a utilizar para generar respuestas basadas en los documentos recuperados.
    consultas_por_evaluar: Una lista de consultas que se desean evaluar.
    respuestas_esperadas: Una lista de respuestas esperadas para cada consulta.
    respuestas_generadas: Una lista de respuestas generadas para cada consulta.
    max_docs_recuperados: El número máximo de documentos recuperados para cada consulta.
    Returns:
    EvaluationDataset: Un dataset de evaluación que contiene las consultas, respuestas generadas, respuestas esperadas y los contextos recuperados para cada consulta.
    """

    dataset = []
    system_prompt = '''
Eres un asistente de IA útil creado para responder preguntas sobre la contexto relevante que se te proporciona. 
Tus respuestas deben ser precisas, exactas y provenir exclusivamente de la información proporcionada.
Por favor, sigue estas pautas:
* Utiliza únicamente información del contexto proporcionada. Evita opiniones, especulaciones o suposiciones.
* Utiliza la terminología y las descripciones exactas que se encuentran en el contenido proporcionado.
* Mantén las respuestas concisas y relevantes para la pregunta del usuario.
* Utiliza acrónimos y abreviaturas exactamente como aparecen en el contexto o consulta.
* Aplica markdown si tu respuesta incluye listas, tablas o código.
* Responde directamente a la pregunta y luego DETENTE. Evita explicaciones adicionales a menos que sean específicamente relevantes.
* Si la información es irrelevante, simplemente responde que no tienes contexto relevante y no proporciones comentarios ni sugerencias adicionales. Ignora cualquier cosa que no pueda usarse para responder directamente a esta consulta.

Devolver directamente la respuesta final.
'''
    # prompt = """Responde la pregunta basado solo en el siguiente contexto, si no hay contexto relevante solo responde que no sabes la respuesta.
    prompt = """
Contexto:
{contexto}

Pregunta: {query}
"""
    template = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", prompt),
        ]
    )
            
    for query, reference in zip(consultas_por_evaluar, respuestas_esperadas):
        generada = respuestas_generadas.pop(0) if respuestas_generadas else None
        if not generada:
            relevant_docs = buscar_documentos_similares(vector_store, query, k=max_docs_recuperados, imprimir_resultados=False)
            response = llm.invoke(template.invoke({"contexto": format_docs(relevant_docs), "query": query})).content
        else:
            relevant_docs = []
            response = generada
            
        dataset.append(
            {
                "user_input": query,
                "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
                "response": response,
                "reference": reference,
            }
        )

    return EvaluationDataset.from_list(dataset)
    

def evaluar_documentos_recuperados(dataset, llm, imprimir_resultados: bool = False, metrics = None):
    """Evalúa la calidad de los documentos recuperados en un dataset de evaluación utilizando métricas de RAGas.
    Args:
    dataset: El conjunto de datos de evaluación que contiene las preguntas, respuestas generadas, respuestas esperadas y contextos recuperados.
    llm: El modelo de lenguaje a utilizar para la evaluación.
    imprimir_resultados: Si se desea imprimir los resultados totales de la evaluación.
    metrics: Una lista de métricas a utilizar para la evaluación. Si es None, se utilizarán las métricas por defecto: LLMContextRecall, Faithfulness y FactualCorrectness.
    """
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (FactualCorrectness, Faithfulness,
                               LLMContextRecall)

    evaluator_llm = LangchainLLMWrapper(llm)
    
    if not metrics:
        metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]

    evaluacion = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )
    evaluacion = evaluacion._scores_dict
    
    if imprimir_resultados:
        imprimir_evaluacion(dataset, evaluacion)
        
    return evaluacion

def imprimir_evaluacion(dataset, evaluacion, nivel_detalle: Literal["total", "resumen"] = "total"):
    """Imprime los resultados de la evaluación de documentos recuperados.
    Args:
        dataset: El conjunto de datos de evaluación que contiene las preguntas, respuestas generadas, respuestas esperadas y contextos recuperados.
        evaluacion: Un diccionario con las métricas de evaluación calculadas para cada pregunta.
        nivel_detalle: El nivel de detalle a imprimir. "total" imprimirá la pregunta, respuesta generada, respuesta esperada y métricas para cada pregunta y el resumen. "resumen" imprimirá solo un resumen de las métricas promedio, máximo y mínimo.
    """
    CONTEXT_RECALL = "context_recall"
    FACTUALLY_CORRECT = "factual_correctness(mode=f1)"
    FAITHFULNESS = "faithfulness"

    dataset_evaluacion = dataset.to_list()
    context_recall = None
    faithfulness = None
    factual_correctness = None

    print("Cantidad de pruebas: ", len(dataset_evaluacion), "\n", "-"*50)
    if CONTEXT_RECALL in evaluacion:
        context_recall = evaluacion["context_recall"]
    if FAITHFULNESS in evaluacion:
        faithfulness = evaluacion["faithfulness"]
    if FACTUALLY_CORRECT in evaluacion:
        factual_correctness = [float(val) for val in evaluacion["factual_correctness(mode=f1)"]]

    if nivel_detalle == "total":
        for i, data in enumerate(dataset_evaluacion):
            print(f"Pregunta {i+1}          :", data["user_input"])
            print(f"Respuesta generada {i+1}:", data["response"])
            print(f"Respuesta esperada {i+1}:", data["reference"])
            if context_recall:
                print(f"context_recall {i+1}     : {context_recall[i]:.2f} - (Contenido de los contextos recuperados vs respuesta esperada)")
            if faithfulness:
                print(f"faithfulness {i+1}       : {faithfulness[i]:.2f} - (Fidelidad de la respuesta generada vs contextos recuperados)")
            if factual_correctness:
                print(f"factual_correctness {i+1}: {factual_correctness[i]:.2f} - (Exactitud de la respuesta generada vs respuesta esperada)")
            print("-"*50)

    # Se cambian los nan por 0 para poder sumarizar los resultados
    if context_recall:
        context_recall = [val if not math.isnan(val) else 0 for val in context_recall]
        print(f"context_recall      promedio: {sum(context_recall)/len(context_recall):.2f}, maximo: {max(context_recall):.2f}, minimo: {min(context_recall):.2f}")
    if faithfulness:
        faithfulness = [val if not math.isnan(val) else 0 for val in faithfulness]
        print(f"faithfulness        promedio: {sum(faithfulness)/len(faithfulness):.2f}, maximo: {max(faithfulness):.2f}, minimo: {min(faithfulness):.2f}")
    if factual_correctness:
        factual_correctness = [val if not math.isnan(val) else 0 for val in factual_correctness]
        print(f"factual_correctness promedio: {sum(factual_correctness)/len(factual_correctness):.2f}, maximo: {max(factual_correctness):.2f}, minimo: {min(factual_correctness):.2f}")
