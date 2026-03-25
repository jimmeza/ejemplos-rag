from typing import Union, List, Any
import logging
import time
from typing import List, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document

from rag_helper.build_vector_db import crear_vector_store, crear_dataset_evaluacion, evaluar_documentos_recuperados, imprimir_evaluacion
from rag_helper.score_relevancia import filtrar_documentos_por_relevancia

logger = logging.getLogger()


class QueryRAG:
    """Clase QueryRAG que implementa un agente RAG para realizar consultas con búsqueda iterativa y opcionalmente con reranker, utilizando un modelo de lenguaje y una base de datos vectorial para recuperar documentos relevantes y generar respuestas precisas basadas en la información recuperada.
    Attributes:
        _model: El modelo de lenguaje a utilizar para generar respuestas y evaluar relevancia.
        _system_prompt: El prompt del sistema que define el comportamiento del agente RAG.
        _vector_store: La base de datos vectorial que se utilizará para recuperar documentos relevantes.
        _agente: El agente RAG que se utilizará para realizar las consultas.
        _reranker: El tipo de reranker a utilizar para filtrar los documentos recuperados en las búsquedas.
        _llm_reranker: El modelo de lenguaje a utilizar como reranker si se selecciona "llm" como tipo de reranker.
        _documentos_recuperados: Una lista de diccionarios con los documentos recuperados en cada búsqueda realizada por el agente RAG durante la última consulta, incluyendo el criterio de búsqueda utilizado y los documentos recuperados con su contenido y metadata de relevancia si está disponible.
    Methods:
        tools: Define las herramientas disponibles para el agente RAG, incluyendo la herramienta de recuperación de documentos relevantes.
        consultar: Realiza una consulta utilizando el agente RAG con búsqueda iterativa y opcionalmente con reranker, y devuelve la respuesta generada por el agente para la pregunta dada.
        evaluar_agente_rag: Evalúa el desempeño de un agente RAG en una serie de consultas y respuestas esperadas, generando un dataset de evaluación y calculando métricas de corrección factual.
        mensajes: Devuelve la lista de mensajes intercambiados con el agente RAG durante la última consulta realizada, incluyendo mensajes del usuario, respuestas del agente y llamadas a herramientas con sus argumentos.
        metadata_uso_api: Devuelve un diccionario con la metadata de uso de tokens en las llamadas a la API del modelo durante la última consulta RAG, incluyendo detalles de tokens de entrada y salida, tokens de razonamiento y tokens leídos desde caché.
        documentos_recuperados: Devuelve una lista de diccionarios con los documentos recuperados   en cada búsqueda realizada por el agente RAG durante la última consulta, incluyendo el criterio de búsqueda utilizado y los documentos recuperados con su contenido y metadata de relevancia si está disponible.
        reranker: Devuelve el tipo de reranker utilizado para filtrar los documentos recuperados en las búsquedas del agente RAG, que puede ser "local" para un reranker basado en similitud local o "llm" para un reranker basado en un modelo de lenguaje, o None si no se está utilizando ningún reranker. También permite establecer el tipo de reranker a utilizar.
    """
    _model = None
    _system_prompt = None
    _vector_store = None
    _agente = None
    _reranker = None
    _mensajes = None
    _metadata_uso_api = None
    _llm_reranker = None
    _documentos_recuperados = []
    _documentos_unicos = set()
    _default_rag_agent_prompt = '''
Eres un asistente de IA útil creado para responder preguntas sobre la documentación relevante que se te proporciona. 
Tus respuestas deben ser precisas, exactas y provenir exclusivamente de la información proporcionada.
Por favor, sigue estas pautas:
* Utiliza únicamente información de la documentación proporcionada. Evita opiniones, especulaciones o suposiciones.
* Utiliza la terminología y las descripciones exactas que se encuentran en el contenido proporcionado.
* Mantén las respuestas concisas y relevantes para la pregunta del usuario.
* Utiliza acrónimos y abreviaturas exactamente como aparecen en la documentación o consulta.
* Aplica markdown si tu respuesta incluye listas, tablas o código.
* Responde directamente a la pregunta y luego DETENTE. Evita explicaciones adicionales a menos que sean específicamente relevantes.
* Si la información es irrelevante, simplemente responde que no tienes documentación relevante y no proporciones comentarios ni sugerencias adicionales. Ignora cualquier cosa que no pueda usarse para responder directamente a esta consulta.
* Si la información es insuficiente para responder la pregunta completa, responde con solo lo que puedes afirmar según el contexto proporcionado, indicando que no obtuviste suficiente información. Ignora cualquier cosa que no pueda usarse para responder directamente a esta consulta.

Devolver directamente la respuesta final y un listado de documentos relevantes para la respuesta(indicando su url o nombre del archivo de origen si es  posible).
Ejemplo:
respuesta final

<documentos>
Título de la metadata del documento1 | url o nombre del archivo de origen
Título de la metadata del documento2 | url o nombre del archivo de origen
</documentos>
'''


    def __init__(self, model, vector_store, 
                 system_prompt = _default_rag_agent_prompt,
                 reranker: Literal["local", "llm"] | None = None,
                 llm_reranker = None
                 ):
        """Inicializa el agente RAG con el modelo de lenguaje, la base de datos vectorial para recuperación de documentos, el prompt del sistema y el tipo de reranker a utilizar para filtrar los documentos recuperados en las búsquedas.
        Args:
            model: El modelo de lenguaje a utilizar para generar respuestas y evaluar relevancia.
            vector_store: La base de datos vectorial que se utilizará para recuperar documentos relevantes.
            system_prompt: El prompt del sistema que define el comportamiento del agente RAG. Por defecto es un prompt que indica al agente que debe responder solo con información relevante proveniente de los documentos recuperados.
            reranker: El tipo de reranker a utilizar para filtrar los documentos recuperados en las búsquedas. Puede ser "local" para un reranker basado en similitud local, "llm" para un reranker basado en un modelo de lenguaje, o None para no utilizar ningún reranker. Por defecto es None.
            llm_reranker: El modelo de lenguaje a utilizar como reranker si se selecciona "llm" como tipo de reranker. Si se selecciona "llm" pero no se proporciona un modelo, se utilizará como default gpt-oss-20b de Groq(se necesita su API Key en archivo .env).
        """
        self._model = model
        self._system_prompt = system_prompt
        self._vector_store = vector_store
        self._reranker = reranker
        self._agente = create_agent(
            model=self._model,
            tools=self.tools(),
            system_prompt=system_prompt,
        )
        self._llm_reranker = llm_reranker

    def tools(self):
        @tool
        def retriever_relevante(pregunta:str,
                                filtro:str, 
                                k:int = 5, 
                                umbral:float=0,
                                primera_busqueda: bool = True,
                                )-> List[Document]: 
            """Busca documentos relevantes para responder a la pregunta del usuario, usa criterios relevantes o frases clave.

            Args:
                pregunta (str): La pregunta del usuario para la cual se buscan documentos relevantes.
                filtro (str): Criterio de búsqueda a utilizar en la búqueda.
                k (int, optional): Cantidad de documentos máximos a recuperar. Por defecto 5.
                umbral (float, optional): Umbral de relevancia entre 0 y 1. Por defecto 0.75.
                primera_busqueda (bool, optional): Si es la primera vez que se hace la búsqueda. Por defecto True.
            Returns:
                List[Document]: Lista de documentos relevantes
            """
            
            logger.info(f"Buscar: {filtro[:50]}..., Max docs recuperados:{k}")

            doc_recuperados = self._vector_store.similarity_search(filtro, k=k)
            logger.info(f"Documentos recuperados: {len(doc_recuperados)}")
            for i, doc in enumerate(doc_recuperados):
                logger.info(f"Documento {i}:\n{doc.page_content}\n" + "="*50 + "\n")
            
            if self._reranker:
                # Se aplicará un filtrado permisivo en la primera busqueda para obtener un contexto inicial
                # Si se filtrara todo en la primera busqueda, no se tendrá un contexto para mejorar las busquedas siguientes
                doc_recuperados = filtrar_documentos_por_relevancia(
                    pregunta,
                    doc_recuperados,
                    umbral_relevancia=umbral,
                    reranker=self._reranker,
                    busqueda_permisiva=primera_busqueda,
                    llm_reranker=self._llm_reranker,
                    )

                logger.info(f"Documentos rerankeados (umbral {umbral}): {len(doc_recuperados)}")
            
            for doc in doc_recuperados:
                if doc.page_content in self._documentos_unicos:
                    doc.metadata["repetido"] = True
                
            self._documentos_recuperados.append({"busqueda": filtro, "documentos_recuperados": doc_recuperados})
            # Se eliminan documentos recuperados que ya se hayan devuelto en búsquedas anteriores.
            doc_recuperados = [doc for doc in doc_recuperados if doc.page_content not in self._documentos_unicos]
            for doc in doc_recuperados:
                self._documentos_unicos.add(doc.page_content)
            logger.info(f"Documentos nuevos devueltos: {len(doc_recuperados)}")
            
            return doc_recuperados if doc_recuperados else [Document(page_content="No hay nuevos documentos relevantes.")]

        return [retriever_relevante]

    def consultar(self,
                    pregunta: str,
                    max_docs_recuperados: int = 5,
                    umbral: float = 0,
                    max_busquedas: int = 2,
                    limite_recursion: int = 10,
                    imprimir_mensajes: bool = False,
                    imprimir_respuesta: Literal["resumen", "total"] | None = None,
                    )->str:
        """Realiza una consulta utilizando un agente RAG con búsqueda iterativa y opcionalmente con reranker.
        Args:
            pregunta (str): La pregunta o consulta que se desea responder.
            max_docs_recuperados (int, optional): El número máximo de documentos a recuperar en cada búsqueda. Por defecto 5.
            umbral (float, optional): El umbral de relevancia para filtrar los documentos recuperados. Por defecto 0.
            max_busquedas (int, optional): El número máximo de búsquedas iterativas que el agente puede realizar. Por defecto 2.
            limite_recursion (int, optional): El límite de recursión para evitar bucles infinitos en la búsqueda iterativa. Por defecto 10.
            imprimir_mensajes (bool, optional): Si se deben imprimir los mensajes intercambiados con el agente. Por defecto False.
            imprimir_respuesta (Literal["resumen", "total"] | None, optional): Si se debe imprimir la respuesta final y el detalle de los documentos recuperados. "resumen" para solo cantidad de documentos recuperados y "total" para detalle completo. Por defecto None.
        Returns:
            str: La respuesta generada por el agente RAG para la pregunta dada.
        """


        prompt = ("Puedes buscar los documentos relevantas usando las herramientas disponibles "
            f"hasta en {max_busquedas} ocasiones pero con criterios de búsqueda diferentes (ASEGURATE DE NO SUPERAR ESTE LIMITE).\n"
            "Si aún te falta contexto para responder la pregunta, vuelve a hacer búsquedas "
            "asegurandote de usar criterios de búsqueda diferentes y de no superar el límite.\n"
            # "Antes de hacer una busqueda, asegurate de que usas un criterio de búsqueda diferente.\n"
            f"Con un máximo de {max_docs_recuperados} documentos recuperados por cada búsqueda, y un umbral de relevancia de {umbral},"
            f"responder a la siguiente pregunta:\n{pregunta}"
        )

        start_time = time.time()
        events = self._agente.stream(
            {"messages": [{"role": "user", 
                           "content": prompt}]},
            {"recursion_limit": limite_recursion},
            stream_mode="values",
        )    
        self._documentos_recuperados = []
        self._documentos_unicos = set()
        respuesta_final = ""
        ultimo_mensaje = None
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        cache_read = 0
        total_pasos = 0
        mensajes = []
        for event in events:
            total_pasos += 1
            ultimo_mensaje = event["messages"][-1]
            # respuesta_final = ultimo_mensaje.content
            respuesta_final = ultimo_mensaje.text #if hasattr(ultimo_mensaje, "text") else ultimo_mensaje.content
            mensajes.append(ultimo_mensaje)
            if ultimo_mensaje.type == "ai":
                total_tokens += ultimo_mensaje.usage_metadata["total_tokens"]
                input_tokens += ultimo_mensaje.usage_metadata["input_tokens"]
                output_tokens += ultimo_mensaje.usage_metadata["output_tokens"]
                output_token_details = ultimo_mensaje.usage_metadata.get("output_token_details", None)
                input_token_details = ultimo_mensaje.usage_metadata.get("input_token_details", None)
                if output_token_details:
                    reasoning_tokens += output_token_details.get("reasoning", 0)
                if input_token_details:
                    cache_read += input_token_details.get("cache_read", 0)

            if imprimir_mensajes:
                ultimo_mensaje.pretty_print()
                print("-"*50)

        self._mensajes = mensajes
        self._metadata_uso_api = {
            "input_tokens": input_tokens,
            "input_tokens_details": {
                "cache_read": cache_read,
            },
            "output_tokens": output_tokens,
            "output_tokens_details":{
                "reasoning":reasoning_tokens,
            },
            "total_tokens": total_tokens,
        }
        
        if imprimir_respuesta:
            print(f"Máximo de {max_busquedas} búsquedas, cada una con un máximo de {max_docs_recuperados} chunks recuperados.")
            print(f"Reranker: {self.reranker}", f", Relevancia mínima: {umbral}" if self.reranker else "")
            print("Pregunta:\n", pregunta)
            print("\nRespuesta final:\n", respuesta_final, "\n")
            print(f"Metadata API Inferencia: input_tokens: {input_tokens:,} (cache_read:{cache_read:,}), output_tokens: {output_tokens:,} (reasoning:{reasoning_tokens:,}), total_tokens: {total_tokens:,}")
            print(f"Tiempo de respuesta total: {(time.time() - start_time):.2f} segundos")

            cantidad_docs_unicos = len(self._documentos_unicos)
            cantidad_busquedas = len(self.documentos_recuperados)
            cantidad_docs = sum(len(busqueda["documentos_recuperados"]) for busqueda in self.documentos_recuperados)
            for i, busqueda in enumerate(self.documentos_recuperados):
                if imprimir_respuesta == "total":
                    print(f"Búsqueda {i+1}: {busqueda['busqueda']}, documentos recuperados: {len(busqueda['documentos_recuperados'])}", f", Relevancia mínima: {umbral}" if self.reranker else "")
                for j, doc_recuperado in enumerate(busqueda["documentos_recuperados"]):
                    if imprimir_respuesta == "total":
                        repetido = doc_recuperado.metadata.get("repetido", False)
                        print(f"  Chunk {j+1} ({'repetido' if repetido else 'nuevo'}: {len(doc_recuperado.page_content)} caracteres):{doc_recuperado.page_content[:100].replace('\n',' ')}...", 
                            f", relevancia: {doc_recuperado.metadata["score_relevancia"]:.4f}" if "score_relevancia" in doc_recuperado.metadata else "")
                if imprimir_respuesta == "total":
                    print("-"*50)
            print(f"\nPasos realizados: {total_pasos}, Búsquedas: {cantidad_busquedas}, Chunks recuperados: {cantidad_docs}")
            print(f"Chunks únicos: {cantidad_docs_unicos}, Chunks repetidos: {cantidad_docs-cantidad_docs_unicos}")
        return respuesta_final

    # def _string_to_list(self, s: str) -> Union[List[Any], None]:
    #     """
    #     Convierte una cadena que representa una lista de Python en un objeto list real.
        
    #     Args:
    #         s: Cadena de texto que contiene la representación de una lista
            
    #     Returns:
    #         Lista convertida o None si falla la conversión
    #     """
    #     try:
    #         # ast.literal_eval es seguro para evaluar literales de Python
    #         result = eval(s)
            
    #         # Verificar que el resultado sea efectivamente una lista
    #         if isinstance(result, list):
    #             return result
    #         else:
    #             print(f"Advertencia: La cadena se evaluó como {type(result).__name__}, no como list")
    #             return None
    #     except (ValueError, SyntaxError) as e:
    #         print(f"Error al convertir la cadena a lista: {e}")
    #         return None
    #     except Exception as e:
    #         print(f"Error inesperado: {type(e).__name__}: {e}")
    #         return None

    def evaluar_agente_rag(self, 
                           coleccion, 
                           querys, 
                           respuestas_ok, 
                           nivel_detalle: Literal["total", "resumen"] = "total",
                           max_docs_recuperados=3, 
                           umbral:float=0, 
                           max_busquedas=3, 
                           limite_recursion=12,
                           imprimir_mensajes=False,
                           imprimir_respuestas: Literal["resumen", "total"] | None = None,
                           ):
        """Evalúa el desempeño de un agente RAG en una serie de consultas y respuestas esperadas, generando un dataset de evaluación y calculando métricas de corrección factual.
        Args:            coleccion: El nombre de la colección o base de datos vectorial utilizada para la recuperación de documentos.
            querys: Una lista de consultas que se desean evaluar.
            respuestas_ok: Una lista de respuestas esperadas para cada consulta.
            nivel_detalle: El nivel de detalle a mostrar en la impresión de los resultados de la evaluación. "resumen" para solo mostrar métricas agregadas y cantidad de documentos recuperados, "total" para mostrar detalle completo de cada consulta, respuesta generada, respuesta esperada y documentos recuperados.
            max_docs_recuperados: El número máximo de documentos a recuperar en cada búsqueda durante la evaluación.
            umbral: El umbral de relevancia para filtrar los documentos recuperados durante la evaluación.
            max_busquedas: El número máximo de búsquedas iterativas que el agente puede realizar durante la evaluación.
            limite_recursion: El límite de recursión para evitar bucles infinitos en la búsqueda iterativa durante la evaluación.
            imprimir_mensajes: Si se deben imprimir los mensajes intercambiados con el agente durante la evaluación.
            imprimir_respuestas: Si se debe imprimir la respuesta generada y el detalle de los documentos recuperados para cada consulta durante la evaluación. "resumen" para solo cantidad de documentos recuperados y "total" para detalle completo.
        Returns:
            None
        """
        from ragas.metrics import FactualCorrectness
        respuestas_generadas = []
        tokens_totales=0

        print("Evaluando agente RAG en colección:", coleccion)
        for query in querys:
            respuesta=self.consultar(query, 
                                        max_docs_recuperados=max_docs_recuperados, 
                                        umbral=umbral, 
                                        max_busquedas=max_busquedas, 
                                        limite_recursion=limite_recursion,
                                        imprimir_mensajes=imprimir_mensajes,
                                        imprimir_respuesta=imprimir_respuestas,
                                        )
            respuesta = respuesta.split("<documentos>")[0].strip()
            respuestas_generadas.append(respuesta)
            tokens_totales += self.metadata_uso_api["total_tokens"] if self.metadata_uso_api else 0
            if imprimir_respuestas:
                print("="*50)
        
        print(f"Tokens totales usados por el agente RAG para generar las respuestas de la evaluación: {tokens_totales:,}")
        print("-"*100)

        dataset = crear_dataset_evaluacion(vector_store = None,
                            llm=self._model, 
                            consultas_por_evaluar=querys,
                            respuestas_esperadas=respuestas_ok,
                            respuestas_generadas=respuestas_generadas,
                            )
        #Solo evaluaremos la respuesta generada vs respuesta esperada (FactualCorrectness)
        evaluacion = evaluar_documentos_recuperados(dataset, 
                                    llm=self._model, 
                                    imprimir_resultados=False,
                                    metrics=[FactualCorrectness(),]
                                    )
        
        imprimir_evaluacion(dataset, evaluacion, nivel_detalle=nivel_detalle)

    @property
    def mensajes(self):
        """Devuelve la lista de mensajes intercambiados con el agente RAG durante la última consulta realizada, incluyendo mensajes del usuario, respuestas del agente y llamadas a herramientas con sus argumentos."""
        return self._mensajes
    
    @property
    def metadata_uso_api(self):
        """Devuelve un diccionario con la metadata de uso de tokens en las llamadas a la API del modelo durante la última consulta RAG, incluyendo detalles de tokens de entrada y salida, tokens de razonamiento y tokens leídos desde caché."""
        return self._metadata_uso_api
    
    @property
    def documentos_recuperados(self):
        """Devuelve una lista de diccionarios con los documentos recuperados en cada búsqueda realizada por el agente RAG durante la última consulta, incluyendo el criterio de búsqueda utilizado y los documentos recuperados con su contenido y metadata de relevancia si está disponible."""
        return self._documentos_recuperados
    
    @property
    def reranker(self):
        """Devuelve el tipo de reranker utilizado para filtrar los documentos recuperados en las búsquedas del agente RAG, que puede ser "local" para un reranker basado en similitud local o "llm" para un reranker basado en un modelo de lenguaje, o None si no se está utilizando ningún reranker."""
        return self._reranker
    @reranker.setter
    def reranker(self, value: Literal["local", "llm"] | None):
        """Permite establecer el tipo de reranker a utilizar para filtrar los documentos recuperados en las búsquedas del agente RAG. El valor debe ser "local" para un reranker basado en similitud local, "llm" para un reranker basado en un modelo de lenguaje, o None para no utilizar ningún reranker."""
        self._reranker = value
    
    @property
    def llm_reranker(self):
        """Devuelve el modelo de lenguaje a utilizar como reranker si se selecciona "llm" como tipo de reranker para filtrar los documentos recuperados en las búsquedas del agente RAG."""
        return self._llm_reranker
    @llm_reranker.setter
    def llm_reranker(self, value):
        """Permite establecer el modelo de lenguaje a utilizar como reranker si se selecciona "llm" como tipo de reranker para filtrar los documentos recuperados en las búsquedas del agente RAG."""
        self._llm_reranker = value

def main(args=None):
    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model

    load_dotenv(override=True)

    #Configurar logging a archivo
    nombre_archivo = Path(__file__).stem
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=f"logs/{nombre_archivo}.log", filemode="w", encoding="utf-8")

    project_name = os.getenv("LANGCHAIN_PROJECT")
    print("Proyecto LangSmith:", project_name)
    COLLECTION_NAME = "futbol-peruano-hybrid-markdown-chunks"
    

    vector_store_hybrid = crear_vector_store(COLLECTION_NAME, modo_recuperacion="hybrid")


    #Pruebas con diferentes modelos LLM para el agente RAG y como reranker, se pueden probar otros modelos disponibles en init_chat_model de LangChain o en Ollama o Google Gemini o Groq (se necesita su API Key en archivo .env)
    #Para el agente RAG, se recomienda usar un modelo potente para que pueda entender la pregunta, realizar las búsquedas iterativas y generar una respuesta precisa basada en los documentos recuperados. Para el reranker, se puede usar un modelo más pequeño pero que sea capaz de evaluar la relevancia de los documentos recuperados para la pregunta dada.
    #Para pruebas rápidas se puede usar un modelo pequeño para el agente RAG y ninguno o un modelo pequeño para el reranker, pero para obtener mejores resultados se recomienda usar modelos potentes para ambos.
    
    #Gemini
    # llm = init_chat_model("gemini-3.1-flash-lite-preview", model_provider="google_genai", temperature=0.0)
    #Ollama
    # llm = init_chat_model("qwen3.5:9b", model_provider="ollama", temperature=0.0)
    #Groq
    llm = init_chat_model("openai/gpt-oss-120b", model_provider="groq", temperature=0.0)
    
    #OpenAI
    #llm = init_chat_model("gpt-5-mini", model_provider="openai", temperature=0.0)
    #OpenRouter
    
    # llm = init_chat_model("openai/gpt-oss-20b", model_provider="openai", 
    # llm = init_chat_model("minimax/minimax-m2.5:free", model_provider="openai", 
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    #     temperature=0.0)
    #Llama.cpp local (se necesita tener el modelo descargado y configurado en local (llama-server) y usar el base_url con el puerto correspondiente)
    # llm = init_chat_model(model="local", 
    #                     model_provider="openai", 
    #                     base_url="http://localhost:8080/v1",
    #                     api_key="no-necesita",
    #                     temperature=0.0,
    #                     )


    query = """Identifica las copas, campeonatos o juegos internacionales en las que participó la selección de Perú (de mayores)
    y muestrame las 11 ocasiones en que ganó en una tabla, indicando el año y la competencia (muestra los datos disponibles)."""
    # Asegurate de haber obtenido todo el historial de campeonatos donde participó Perú para responder con precisión a esta consulta.
    # Realiza búsquedas hasta llegar la máximo de búquedas indicado para obtener todo el contexto posible."""
    # y muestrame solo las 9 ocasiones en que ganó."""

    # query = """Identifica a los clubes del Callao que han ganado campeonatos nacionales, 
    # luego muestrame una tabla indicando el nombre del club, el campeonato ganado y el año de la victoria.
    # Asegurate de haber obtenido todo el historial de campeonatos y sus ganadores para responder con precisión a esta consulta."""

    respuestas_ok = [
        """Las ocasiones en que la selección de Perú ganó en copas, campeonatos o juegos internacionales son:
        | Año | Competición ganada |
        |-----|--------------------|
        | 1938 | Juegos Bolivarianos |
        | 1939 | Campeonato Sudamericano (Copa América) |
        | 1947-1948 | Juegos Bolivarianos |
        | 1961 | Juegos Bolivarianos |
        | 1973 | Juegos Bolivarianos |
        | 1975 | Copa América |
        | 1981 | Juegos Bolivarianos |
        | 1999 | Copa Kirin (empate con Bélgica)|
        | 2000 | Copa Oro de la CONCACAF |
        | 2005 | Copa Kirin |
        | 2011 | Copa Kirin |
        """,
    ]

    # llm_reranker = init_chat_model("openai/gpt-oss-20b", model_provider="groq", temperature=0.0)

    # TODO: Quitar chunks duplicados
    query_rag_hybrid = QueryRAG(llm, 
                                vector_store_hybrid, 
                                # reranker="llm",
                                # llm_reranker=llm_reranker,
                                )

    print("Colección:", COLLECTION_NAME)
    respuesta = query_rag_hybrid.consultar(query, max_docs_recuperados=5, max_busquedas=5, limite_recursion=15, imprimir_respuesta="total")
    print("="*100)
    print("No hubo respuesta del agente RAG." if respuesta.strip() == "" else "Respuesta generada por el agente RAG:\n", respuesta)
    # print(query_rag_hybrid.documentos_recuperados)
    # print("metadata: ", query_rag_hybrid.metadata_uso_api)
    # print("mensajes: ", query_rag_hybrid.mensajes)
    # query_rag_hybrid.consultar(query, max_docs_recuperados=5, umbral=0.5, max_busquedas=5, limite_recursion=15, imprimir_respuesta="total")
    
if __name__ == "__main__":
    main()

