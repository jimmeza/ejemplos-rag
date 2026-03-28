#Personalizacion del Docling Loader para limpiar el resultado del formato MD y que convierta las tablas md a json
#Obtenido de: https://github.com/docling-project/docling-langchain/blob/main/langchain_docling/loader.py
#
# Copyright IBM Corp. 2025 - 2025
# SPDX-License-Identifier: MIT
#

"""Docling LangChain loader module."""

import json
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Literal, Optional, Union
from dotenv import load_dotenv

import semchunk
from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.accelerator_options import (AcceleratorDevice,
                                                   AcceleratorOptions)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import (EasyOcrOptions,
                                                PdfPipelineOptions,
                                                TableStructureOptions,
                                                VlmPipelineOptions)
from docling.datamodel.pipeline_options_vlm_model import (ApiVlmOptions,
                                                          ResponseFormat)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveJsonSplitter
from transformers import AutoTokenizer

load_dotenv(override=True)
Environment = Literal['development', 'testing', 'production']
ENVIRONMENT: Environment = os.getenv('ENVIRONMENT', 'development') 

class ExportType(str, Enum):
    """Enumeration of available export types."""

    MARKDOWN = "markdown"
    DOC_CHUNKS = "doc_chunks"


class BaseMetaExtractor(ABC):
    """BaseMetaExtractor."""

    @abstractmethod
    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        raise NotImplementedError()

    @abstractmethod
    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract Docling document meta."""
        raise NotImplementedError()


class MetaExtractor(BaseMetaExtractor):
    """MetaExtractor."""

    def extract_chunk_meta(self, file_path: str, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        return {
            "source": file_path,
            "dl_meta": chunk.meta.export_json_dict(),
        }

    def extract_dl_doc_meta(
        self, file_path: str, dl_doc: DoclingDocument
    ) -> dict[str, Any]:
        """Extract Docling document meta."""
        return {"source": file_path}

def markdown_to_plain_text(md_text):
    """
    Convierte una cadena en formato Markdown a texto sin formato.
    Elimina o simplifica los elementos de formato.
    """
    if not isinstance(md_text, str):
        return ""

    text = md_text

    # 1. Eliminar bloques de código multilínea (```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

    # 2. Eliminar bloques de código en línea (`)
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # 3. Eliminar encabezados (### Título -> Título)
    text = re.sub(r'^(\s*)#{1,6}\s+', r'\1', text, flags=re.MULTILINE)

    # 4. Eliminar negritas (**texto** o __texto__)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)

    # 5. Eliminar cursivas (*texto* o _texto_)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)

    # 6. Eliminar tachado (~~texto~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)

    # 7. Convertir enlaces [texto](url) -> texto
    text = re.sub(r'\[([^\]]+)\]\([^)]+\){1,2}', r'\1', text)
    
    # 8. Eliminar marcadores de lista al inicio de línea (-, *, +, 1., 2., etc.)
    text = re.sub(r'^\s*[-+*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # 9. Eliminar citas (> texto -> texto)
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)

    # 10. Eliminar líneas horizontales (---, ***, ___)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 11. Reemplazar múltiples saltos de línea por solo dos
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 12. Eliminar citas de wikipedia [[texto]](url)
    text = re.sub(r'\[\[([^\]]+)\]\]\([^)]+\)', '', text)
    return text

def md_table_to_json(md_table: str, titulo_tabla: str = "título")->list[dict]:
    """
    Convierte una tabla en formato Markdown a una estructura JSON agrupada por títulos.
    La función procesa una tabla Markdown que puede contener filas de títulos/agrupadores y filas de datos.
    Las filas de títulos se identifican como aquellas que contienen una sola columna con valores distintos (pueden exitir varios títulos o ninguno), 
    mientras que las filas de datos contienen el mismo número de columnas que los encabezados detectados. 
    Las filas de separación (---) son ignoradas.
    Args:
        md_table (str): Cadena que representa la tabla en formato Markdown.
            Ejemplo:
                |---------|------|-----------|
                | Nombre  | Edad | Ciudad    |
                |---------|------|-----------|
                | Titulo 1     |||
                | Ana     | 23   | Lima      |
                | Titulo 2  |Titulo 2||
                | Luis    | 31   | Cusco     |
                | Titulo 3 |Titulo 3|Titulo 3|
                | Marta   | 28   | Arequipa  |
    Returns:
        list: Lista de diccionarios, cada uno con las claves 'titulo_tabla' (nombre del grupo) y 'datos_tabla' (lista de filas de datos como diccionarios).
    Ejemplo de retorno:
        [
            {
                "titulo_tabla": "Titulo 1",
                "datos_tabla": [
                {
                    "Nombre": "Ana",
                    "Edad": "23",
                    "Ciudad": "Lima"
                }
                ]
            },
            {
                "titulo_tabla": "Titulo 2",
                "datos_tabla": [
                {
                    "Nombre": "Luis",
                    "Edad": "31",
                    "Ciudad": "Cusco"
                }
                ]
            },
            {
                "titulo_tabla": "Titulo 3",
                "datos_tabla": [
                {
                    "Nombre": "Marta",
                    "Edad": "28",
                    "Ciudad": "Arequipa"
                }
                ]
            }
        ]
    """
    texto = md_table.strip()
    lines = [line.strip() for line in texto.split('\n') if line.strip()]
    if not lines:
        return []

    # Buscar la primera fila con más de una columna diferente como encabezado, ignorando filas de separación
    headers = []
    header_idx = -1
    cantidad_columnas = lines[0].count('|')-1
    cantidad_separadores = 0
    encontro_encabezado = False

    for idx, line in enumerate(lines):
        # Ignorar filas de separación
        if re.match(r'^\s*\|?\s*-+\s*\|?(\s*-+\s*\|?)*\s*$', line):
            cantidad_separadores += 1
            continue
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if len(set(cols)) > 1 and cantidad_columnas == len(set(cols)):
            headers = cols
            header_idx = idx
            break

    if not headers:
        #columnas no vacias de la primera fila
        cols0 = [c.strip() for c in lines[0].split('|') if c.strip()]
        if cantidad_separadores == 1 and len(cols0) == cantidad_columnas:
            #Uso de encabezados la primera fila si existe una fila de separadores
            headers = cols0
            encontro_encabezado = True
        else:
            #Si no se encontro encabezados se genera uno por defecto
            headers = [f"Col {i+1}" for i in range(cantidad_columnas)]

    #Se elimina la fila de encabezados (si existe)
    lineas_sin_headers = lines.copy()
    if header_idx != -1:
        lineas_sin_headers.remove(lines[header_idx])

    # Filtrar líneas de separación (---)
    data_lines = [line for line in lineas_sin_headers if not re.match(r'^\s*\|?\s*-+\s*\|?(\s*-+\s*\|?)*\s*$', line)]
    current_title = titulo_tabla
    current_group = []
    result = []

    for line in data_lines:
        cols = [c.strip() for c in line.split('|') if c.strip()]
        if len(cols) == 0:
            continue

        #Si toda la linea es una tabla anidada, se maneja como una subtabla md
        if line.startswith("| |") and line.endswith("| |"):
            texto_subtabla = line[2:-2]
            if texto_subtabla.strip() != "":
                result.append({'titulo_tabla': current_title,
                               'subtabla': 
                    md_table_to_json(corregir_tabla_markdown(texto_subtabla), current_title)
                })
            continue
        
        if len(set(cols)) == 1: #and len(cols) != 1:
            # Es una fila de título/agrupador (solo columnas con el mismo encabezado), se agrega la subtabla anterior
            if current_title and current_group:
                result.append({'titulo_tabla': current_title, 'datos_tabla': current_group})
            current_title = titulo_tabla + "/" + cols[0]
            current_group = []
        elif len(cols) <= len(headers):
            # Es una fila de datos (se agrega a la subtabla actual)
            row = dict(zip(headers, cols))
            if len(row) > 0:
                current_group.append(row)

    # Agregar el último grupo como nueva tabla si existe
    if current_group:
        result.append({'titulo_tabla': current_title, 'datos_tabla': current_group})

    if not result:
        if encontro_encabezado:
            # solo se encontro una fila de encabezados
            result = [{"titulo_tabla": current_title, "datos_tabla": headers}]
        elif titulo_tabla != current_title:
            # Solo encontro una fila título/agrupador ()
            result = [{"titulo_tabla": titulo_tabla, "datos_tabla": "/".join(current_title.split("/")[1:])}]

    return result

def cadena_json(objeto)->str:
    """Convierte un objeto Python en una cadena en formato json de fácil lectura para humanos
    Args:
        objeto: objeto Python a convertir

    Returns:
        str: Cadena en formato json de fácil lectura para humanos
    """
    return json.dumps(objeto, indent=2, ensure_ascii=False)

def extraer_tablas_md(texto)->list[tuple[str, int, int]]:
    """
    Busca todas las tablas en formato Markdown dentro de un texto y devuelve una lista de tuplas:
    (tabla_encontrada, posición_inicial, posición_final)
    """
    texto_original_termina_salto_linea = texto.endswith('\n')
    #Se agrega \n en caso no tenga un salto de linea final para que se reconozca la ultima fila de la tabla
    if not texto_original_termina_salto_linea:
        texto+="\n"
    # Expresión regular para detectar tablas Markdown
    # Busca líneas que empiezan y terminan con | y que tengan al menos una fila de separación |---| y alguna fila de datos
    patron_tabla = re.compile(
        r'((?:\s*\|.*\n)+?\s*\|[\s\-|]+\|\s*\n(?:\s*\|.*\n)+)', re.MULTILINE
    )
    tablas_encontradas = patron_tabla.findall(texto)
    if len(tablas_encontradas) == 0:
        # Busca una tabla md solo con encabezado y fila de separación (sin filas de datos)
        patron_tabla = re.compile(
            r'((?:\s*\|.*\n)+?\s*\|[\s\-|]+\|\s*)', re.MULTILINE
        )

    resultados = []
    for match in patron_tabla.finditer(texto):
        #si se encuentran mas de un salto de linea seguido, se separa el texto en tablas diferentes
        tablas = match.group(0).split('\n\n')
        inicio_tabla = match.start()
        for tabla in tablas:
            if tabla.strip() != "":
                inicio = inicio_tabla
                fin = inicio_tabla + len(tabla)
                resultados.append((tabla, inicio, fin))
            inicio_tabla += len(tabla) + 2 #Se suma 2 para considerar los saltos de linea eliminados

    if not texto_original_termina_salto_linea and len(resultados)>0:
        resultados[-1] = (resultados[-1][0][:-1], resultados[-1][1], resultados[-1][2]-1)
    return resultados

#Funcion para extrar de un texto md los titulos encadenados y separados por "/"
def extraer_titulos_encadenados(texto)->str:
    """
    Extrae los titulos encadenados y separados por "/" de un texto md.
    Args:
        texto (str): Texto md.
    Returns:
        str: Titulos encadenados y separados por "/".
    """
    patron_titulo = re.compile(r'^#{1,6}\s*(.*)$', re.MULTILINE)
    titulos_encadenados = ""
    #Se encadenan los titulos 
    for match in patron_titulo.finditer(texto):
        titulo = markdown_to_plain_text(match.group(0).strip())
        #titulo = match.group(0).strip()
        titulos_encadenados += "/" + titulo if titulos_encadenados != "" else titulo
    return titulos_encadenados

def separar_texto_y_tablas(texto)->list[tuple[str, str, str]]:
    """
    Divide un texto en partes, separando las tablas Markdown del resto del contenido.
    Devuelve una lista de tuplas (tipo, contenido, titulos_encadenados), donde tipo es 'texto' o 'tabla'.
    """
    tablas = extraer_tablas_md(texto)
    if not tablas:
        return [("texto", texto, extraer_titulos_encadenados(texto))]

    partes = []
    idx = 0
    ultimo_texto = ""
    ultimo_titulo = ""

    for tabla, inicio, fin in tablas:
        # Añadir el texto antes de la tabla (si existe)
        if inicio > idx:
            ultimo_texto = texto[idx:inicio].strip()
            if ultimo_texto:
                ultimo_titulo = extraer_titulos_encadenados(ultimo_texto)
                partes.append(("texto", ultimo_texto, ultimo_titulo))
        # Añadir la tabla
        partes.append(("tabla", tabla, ultimo_titulo))
        idx = fin

    # Añadir el texto después de la última tabla (si existe)
    if idx < len(texto):
        partes.append(("texto", texto[idx:], extraer_titulos_encadenados(texto[idx:])))

    return partes

def reemplazar_tablas_md_a_json(texto, titulo_tabla:str="Default")->str:
    """
    Reemplaza todas las tablas en formato Markdown dentro de un texto por su representación en JSON.

    Args:
        texto (str): Texto que contiene tablas en formato Markdown.
        titulo_tabla (str, optional): Titulo de la tabla. Defaults to "Default".

    Returns:
        str: Texto modificado donde cada tabla Markdown ha sido reemplazada por un bloque de código JSON.
    """
    tablas = extraer_tablas_md(texto)
    texto_modificado = texto
    for tabla, inicio, fin in reversed(tablas):
        json_tabla = md_table_to_json(tabla, titulo_tabla)
        #json_str = f"\n```json\n{cadena_json(json_tabla)}\n```\n"
        json_str = f"{cadena_json(json_tabla)}\n"
        texto_modificado = texto_modificado[:inicio] + json_str + texto_modificado[fin:]
    return texto_modificado

def reemplazar_espacios_multiples(texto):
    """
    Reemplaza dos o más espacios seguidos por uno solo en la cadena dada.
    """
    return re.sub(r' {2,}', ' ', texto)

def completar_titulo_tabla(lista_diccionario:list[dict])->list[dict]:
    """
    Agrega el atributo "titulo_tabla" del primer elemento de la lista a los demas elementos.
    """
    if len(lista_diccionario) > 1:
        titulo_tabla = lista_diccionario[0]["titulo_tabla"]
        for diccionario in lista_diccionario[1:]:
            if "titulo_tabla" not in diccionario:
                diccionario["titulo_tabla"] = titulo_tabla
    return lista_diccionario

def corregir_titulo(titulo:str, texto:str)->str:
    """
    Corrige un titulo de un texto, el titulo esta compuesto por una cadena de subtitulos concatenados con "/"
    se debe eliminar un subtitulo si no esta presente en el texto, si no quedan subtitulos, devolver el ultimo
    """
    subtitulos = titulo.split("/")
    if len(subtitulos) == 1:
        return titulo
    ultimo_subtitulo = subtitulos[-1]
    titulo_corregido = ""
    for subtitulo in subtitulos:
        if subtitulo in texto:
            titulo_corregido += "/" + subtitulo if titulo_corregido != "" else subtitulo
    
    if titulo_corregido == "":
        titulo_corregido = ultimo_subtitulo

    return titulo_corregido

def corregir_tabla_markdown(texto_md: str) -> str:
    """
    Corrige una tabla markdown a la que se le eliminaron todos los saltos de línea.

    La función usa la fila separadora (celdas con '---') como ancla principal,
    dividiendo la cadena en tres zonas: encabezado, separador y datos. A continuación
    aplica el número de columnas del separador para subdividir la zona de datos.

    Algoritmo:
    1. Localizar la fila sep mediante la regex ``\\|[-|:]+\\|`` (solo guiones/pipes).
    2. Retroceder desde el inicio de la sep para detectar el cierre del encabezado.
    3. Subdividir la zona de datos contando (N_cols + 1) pipes por fila.

    Args:
        texto_md: Cadena con la tabla markdown sin saltos de línea.

    Returns:
        Cadena con la tabla markdown correctamente formateada con saltos de línea.
        Si no se detecta fila separadora, se devuelve el texto sin cambios.
    """
    if not texto_md or not texto_md.strip():
        return texto_md

    texto = texto_md.strip()

    # ── 1. Localizar la fila separadora ─────────────────────────────────────
    # Patrón: secuencia de pipes con solo guiones y dos puntos entre ellos.
    sep_match = re.search(r'\|[-|:]+\|', texto)
    if sep_match is None:
        # Sin separador: no es una tabla estándar; devolver sin cambios.
        return texto

    sep_start = sep_match.start()
    sep_end   = sep_match.end()
    sep_text  = sep_match.group()
    n_cols    = sep_text.count('|') - 1  # número de columnas detectadas

    # ── 2. Separar el encabezado del separador ───────────────────────────────
    # La fila sep puede estar precedida por espacios y el '|' de cierre del enc.
    k = sep_start - 1
    while k >= 0 and texto[k] == ' ':
        k -= 1

    if k >= 0 and texto[k] == '|':
        enc_end = k + 1   # incluir el '|' de cierre del encabezado
    else:
        enc_end = sep_start

    enc_text   = texto[:enc_end].strip()
    datos_text = texto[sep_end:].strip()

    # ── 3. Reconstruir las filas ──────────────────────────────────────────────
    lineas = []

    if enc_text:
        lineas.append(enc_text)

    lineas.append(sep_text)

    # Subdividir la zona de datos: cada fila tiene exactamente n_cols + 1 pipes.
    if datos_text:
        pipes_por_fila = n_cols + 1
        pipe_count = 0
        inicio = 0

        for pos, ch in enumerate(datos_text):
            if ch == '|':
                pipe_count += 1
                if pipe_count == pipes_por_fila:
                    lineas.append(datos_text[inicio:pos + 1].strip())
                    pipe_count = 0
                    inicio = pos + 1

        # Sobrante (tabla incompleta o mal formada)
        sobrante = datos_text[inicio:].strip()
        if sobrante:
            lineas.append(sobrante)

    return '\n'.join(lineas)


class CustomDoclingLoader(BaseLoader):
    """Docling Loader personalizado para limpiar el resultado del formato Markdown y que se pueda elegir 
    si la jerarquia de referencias del documento original va al inicio del documento"""
    HF_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    MAX_TOKENS_OLLAMA_DEFAULT_EMBEDDING = 8192

    def __init__(
        self,
        file_path: Union[str, Iterable[str]],
        *,
        converter: Optional[DocumentConverter] = None,
        convert_kwargs: Optional[Dict[str, Any]] = None,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        md_export_kwargs: Optional[dict[str, Any]] = None,
        chunker: Optional[BaseChunker] = None,
        meta_extractor: Optional[BaseMetaExtractor] = None,
        #Nuevos argumentos
        contextualize_chunk: Optional[bool] = False,
        tokenizer = None,
        max_tokens = None,
        tipo_ocr_pdf: Literal["local", "vlm"] = "local",
        num_threads_ocr_pdf_local: int = 4,
        url_ocr_pdf_vlm: str = "",
        api_key_ocr_pdf_vlm: str = "",
        model_name_ocr_pdf_vlm: str = ""

    ):
        """Initialize with a file path.

        Args:
            file_path: File source as single str (URL or local file) or Iterable
                thereof.
            converter: Any specific `DocumentConverter` to use. Defaults to `None` (i.e.
                converter defined internally).
            convert_kwargs: Any specific kwargs to pass to conversion invocation.
                Defaults to `None` (i.e. behavior defined internally).
            export_type: The type to export to: either `ExportType.MARKDOWN` (outputs
                Markdown of whole input file) or `ExportType.DOC_CHUNKS` (outputs chunks
                based on chunker).
            md_export_kwargs: Any specific kwargs to pass to Markdown export (in case of
                `ExportType.MARKDOWN`). Defaults to `None` (i.e. behavior defined
                internally).
            chunker: Any specific `BaseChunker` to use (in case of
                `ExportType.DOC_CHUNKS`). Defaults to `None` (i.e. chunker defined
                internally).
            meta_extractor: The extractor instance to use for populating the output
                document metadata; if not set, a system default is used.
            contextualize_chunk: True for add hierarchical references at the beginning of document chunk
            tokenizer: Tokenizer to use. Defaults to `None` (i.e. tokenizer defined
                internally).
            max_tokens: Maximum number of tokens to use. Defaults to `None` (i.e. behavior defined
                internally).
            num_threads_ocr_pdf_local: Número de hilos a usar en el conversor de PDF, debe personalizarse según los núcleos del procesador
            tipo_ocr: Tipo de OCR a usar en el conversor de PDF, puede ser "local" (con la libreria EasyOCR) o "vlm" (con un VLM compatible con OpenAI)
            url_ocr_pdf_vlm: URL del VLM compatible con OpenAI a usar en el conversor de PDF, debe ser una URL válida
            api_key_ocr_pdf_vlm: API key del VLM a usar en el conversor de PDF, debe ser una API key válida
            model_name_ocr_pdf_vlm: Nombre del modelo a usar en el conversor de PDF, debe ser un modelo válido
        """

        self._file_paths = (
            file_path
            if isinstance(file_path, Iterable) and not isinstance(file_path, str)
            else [file_path]
        )

        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = (
            md_export_kwargs
            if md_export_kwargs is not None
            else {"image_placeholder": ""}
        )
        if max_tokens is None:
            if chunker:
                max_tokens = chunker.max_tokens
            else:
                max_tokens = MAX_TOKENS_OLLAMA_DEFAULT_EMBEDDING
        self._max_tokens = max_tokens
        if not tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(self.HF_DEFAULT_EMBEDDING_MODEL, padding_side='left')
        self._tokenizer = tokenizer
        self._chunker: BaseChunker = chunker or HybridChunker(tokenizer=self._tokenizer, max_tokens=max_tokens)
        self._meta_extractor = meta_extractor or MetaExtractor()
        self._contextualize_chunk = contextualize_chunk
        self._num_threads_ocr_pdf_local = num_threads_ocr_pdf_local
        self._tipo_ocr_pdf = tipo_ocr_pdf
        self._url_ocr_pdf_vlm = url_ocr_pdf_vlm
        self._api_key_ocr_pdf_vlm = api_key_ocr_pdf_vlm
        self._model_name_ocr_pdf_vlm = model_name_ocr_pdf_vlm
        self._converter: DocumentConverter = converter or self._build_pdf_converter()

    def _split_docling_document(self, docling_document, file_path) -> Iterator[Document]:
        """
        Separa el documento de Docling en chunks y los retorna como Documentos de LangChain
        
        Args:
            docling_document: Documento de Docling
            file_path: Ruta o nombre del archivo origen
        
        Returns:
            Iterator[Document]: Iterador de Documentos de LangChain
        """
        chunk_iter = self._chunker.chunk(docling_document)
        for chunk in chunk_iter:
            content = self._chunker.contextualize(chunk=chunk) if self._contextualize_chunk else chunk.text
            content = markdown_to_plain_text(content.strip())
            if content == "":
                continue

            yield Document(
                page_content = content,
                metadata=self._meta_extractor.extract_chunk_meta(
                    file_path=file_path,
                    chunk=chunk,
                ),
            )

    def _build_pdf_converter(self):
        """
        Construye el conversor de Docling con un pipeline para PDF usando EasyOCR o un VLM según el atributo self._tipo_ocr_pdf

        Returns:
            DocumentConverter: Conversor de documentos de Docling con pipeline para PDF.
        """
        if self._tipo_ocr_pdf == "local":
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options = TableStructureOptions(
                mode="accurate",
                do_cell_matching=True
            )

            pipeline_options.ocr_options.lang = ["es"]
            accelerator_options = AcceleratorOptions(
                num_threads=self._num_threads_ocr_pdf_local,
                device=AcceleratorDevice.CPU
            )
            ocr_options = EasyOcrOptions(force_full_page_ocr=True)
            pipeline_options.ocr_options = ocr_options
            pipeline_options.accelerator_options = accelerator_options

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        elif self._tipo_ocr_pdf == "vlm":
            vlm_options = ApiVlmOptions(
                url=self._url_ocr_pdf_vlm,
                headers={"Authorization": f"Bearer {self._api_key_ocr_pdf_vlm}"},
                params={"model": self._model_name_ocr_pdf_vlm},
                prompt="""
                Convierte está página a markdown, no uses Latex u otro formato.
                Si hay texto que tenga algun cuadro, imagen o columna a la derecha, ignora todo lo que este a la derecha y solo procesa el texto de la izquierda. 
                No agregues subtitulos que no existan en los textos del documento.
                Asegurate de mantener el orden de los textos y tablas.
                Considera que puedes encontrar tablas que no tienen encabezado, en ese caso, crea uno basado en el contexto.
                Considera que en las tablas, algunas celdas pueden tener varias líneas de texto.
                Solo responde con el markdown.
                """,
                response_format=ResponseFormat.MARKDOWN,
                timeout=60,
                scale=3.0,
                temperature=0.0,
            )


            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=True,
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        pipeline_cls=VlmPipeline,
                    )
                }
            )
        else:
            raise ValueError(f"Tipo de OCR no soportado: {self._tipo_ocr_pdf}")

        return converter

    def lazy_load(
        self,
    ) -> Iterator[Document]:

        max_char_size = max(self._max_tokens, self._max_tokens*2 - 600) #Se ajusta para aproximar la longitud de los chunks en caracteres a la longitud en tokens
        min_char_size = max(int(max_char_size/2), min(100, max_char_size))
        #Crear el chunker semantico

        semantic_text_splitter = semchunk.chunkerify(
            self._tokenizer, 
            self._max_tokens
        )
        #Crear el splitter de json
        json_splitter = RecursiveJsonSplitter(max_chunk_size=max_char_size,
                                            min_chunk_size=min_char_size)

        """Lazy load documents."""
        for file_path in self._file_paths:
            conv_res = self._converter.convert(
                source=file_path,
                **self._convert_kwargs,
            )
            dl_doc = conv_res.document
            if self._export_type == ExportType.MARKDOWN:
                content = reemplazar_espacios_multiples(dl_doc.export_to_markdown(**self._md_export_kwargs))
                segmentos = separar_texto_y_tablas(content)

                if ENVIRONMENT != "production":
                    #Guardar cada segmento separado en un archivo de texto
                    ruta_archivo = "./temp/segmentos_texto.txt"
                    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
                    f = open(ruta_archivo, "w", encoding="utf-8")
                    ruta_archivo_json = "./temp/segmentos_json.txt"
                    fj = open(ruta_archivo_json, "w", encoding="utf-8")

                #Por cada segmento crear Documentos
                for i, (tipo, segmento, titulo) in enumerate(segmentos):
                #for tipo, segmento, titulo in segmentos:
                    if segmento == "":
                        continue
                    segmento = markdown_to_plain_text(segmento)

                    if ENVIRONMENT != "production":
                        #Guarda segmentos en texto sin Markdown (excepto las tablas)
                        f.write(f"Indice: {i}\n")
                        f.write(f"Tipo: {tipo}\n")
                        f.write(f"Titulo: {titulo}\n")
                        f.write(f"Segmento: \n{segmento}\n")
                        f.write("-"*50+"\n")

                    # si es tabla, convertir a json, luego convertir todo a texto plano
                    if tipo == "tabla":
                        #Reemplazar las tablas md por json
                        titulo_tabla = corregir_titulo(titulo, segmento)
                        segmento = reemplazar_tablas_md_a_json(segmento, titulo_tabla).strip()
                        
                        if ENVIRONMENT != "production":
                            #Guarda segmentos en JSON
                            fj.write(f"Indice: {i}\n")
                            fj.write(f"Tipo: {tipo}\n")
                            fj.write(f"Titulo: {titulo}\n")
                            fj.write(f"Segmento: \n{segmento}\n")
                            fj.write("-"*50+"\n")
                        
                        #Convertir cadena json a objeto json
                        lista_json = json.loads(segmento)
                        documentos =[]
                        #Separa cada json en caso supere el maximo aproximado de tokens
                        for elemento in lista_json:
                            documentos.extend(
                                completar_titulo_tabla(
                                    json_splitter.split_json(
                                        json_data=elemento, 
                                        convert_lists=True
                                    )
                                )
                            )
                        documentos = [cadena_json(documento) for documento in documentos]
                    else:
                        #Separar el segmento de texto semanticamente en chunks
                        documentos = semantic_text_splitter(segmento)
                        
                    for documento in documentos:
                        titulo_corregido = corregir_titulo(titulo, documento)
                        yield Document(
                            page_content=documento,
                            metadata={"source": file_path, "seccion": titulo_corregido}
                        )

            elif self._export_type == ExportType.DOC_CHUNKS:
                yield from self._split_docling_document(dl_doc, file_path)

            else:
                raise ValueError(f"Unexpected export type: {self._export_type}")
    
