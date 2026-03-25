
import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
#     VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat

# from docling.datamodel.vlm_engine_options import (
#     ApiVlmEngineOptions,
#     VlmEngineType,
# )
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

def check_and_pull_ollama_model(model_name: str) -> bool:
    """Check if model exists in Ollama and attempt to pull if not.

    Args:
        model_name: The model name to check/pull

    Returns:
        True if model exists or successfully pulled, False otherwise
    """
    try:
        # Check if model exists
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            # Check for exact match or with :latest tag
            if model_name in model_names or f"{model_name}:latest" in model_names:
                print(f"✓ Model '{model_name}' is already available in Ollama")
                return True

            # Try to pull the model using Ollama API
            print(f"Attempting to pull model '{model_name}' in Ollama...")
            print("This may take a few minutes...")

            # Ollama pull API endpoint
            pull_response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300,
            )

            if pull_response.status_code == 200:
                # Stream the response to show progress
                for line in pull_response.iter_lines():
                    if line:
                        import json

                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if status:
                                print(f"  {status}", end="\r")
                        except json.JSONDecodeError:
                            pass
                print()  # New line after progress
                print(f"✓ Successfully pulled model '{model_name}'")
                return True
            else:
                print(f"✗ Failed to pull model: HTTP {pull_response.status_code}")
                return False
        return False
    except requests.exceptions.Timeout:
        print("✗ Timeout while trying to pull model (this can take a while)")
        print("Please try pulling manually: ollama pull", model_name)
        return False
    except Exception as e:
        print(f"✗ Error checking/pulling model: {e}")
        return False

def run_ollama_example(input_doc_path: Path) -> bool:
    """Example 2: Using Granite-Docling preset with Ollama.

    Returns:
        True if example ran successfully, False if skipped
    """
    #model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    model_name = "meta-llama/llama-4-maverick-17b-128e-instruct"
    #model_name = "qwen3-vl:2b"

    # print("\n" + "=" * 70)
    # print(f"Example 2: {model_name} with Ollama (pre-configured API type)")
    # print("=" * 70)
    # print("\nPrerequisites:")
    # print("- Install Ollama: https://ollama.ai")
    # print(f"- Pull model: ollama pull {model_name}")
    # print()

    # # Check if Ollama is running
    # try:
    #     response = requests.get("http://localhost:11434/api/tags", timeout=2)
    #     if response.status_code != 200:
    #         print("WARNING: Ollama server not responding correctly")
    #         print("Skipping Ollama example.\n")
    #         return False
    # except requests.exceptions.RequestException:
    #     print("WARNING: Ollama server not running at http://localhost:11434")
    #     print("Skipping Ollama example.\n")
    #     return False

    # # Check and pull the model
    # if not check_and_pull_ollama_model(model_name):
    #     print("Skipping Ollama example.\n")
    #     return False

    # vlm_options = ApiVlmOptions(
    #     url="http://localhost:11434/v1/chat/completions",
    #     params={"model": model_name},
    #     prompt="Convert this page to docling.",
    #     response_format=ResponseFormat.MARKDOWN,
    #     timeout=90,
    #     #scale=2.0,
    #     temperature=0.0,
    # )

    vlm_options = ApiVlmOptions(
        url=os.getenv('GROQ_API_URL'),
        headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
        params={"model": model_name},
        prompt="""
        Convierte está página a markdown, no uses Latex u otro formato.
        Si hay texto que tenga algun cuadro o columna a la derecha, ignora dicho cuadro y solo procesa el texto de la izquierda. 
        No agregues subtitulos que no existan en los textos del documento.
        Asegurate de mantener el orden de los textos y tablas.
        Considera que puedes encontrar tablas que no tienen encabezado, en ese caso, crea uno basado en el contexto.
        Considera que en las tablas, algunas celdas pueden tener varias líneas de texto.
        Solo responde con el markdown.
        """,
        response_format=ResponseFormat.MARKDOWN,
        timeout=60,
        scale=2.0,
        temperature=0.0,
    )


    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True,
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                pipeline_cls=VlmPipeline,
            )
        }
    )

    result = doc_converter.convert(input_doc_path)
    nombre_archivo = input_doc_path.split("/")[-1]
    ruta_archivo = f"./temp/vlm/{nombre_archivo}.md"
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write(result.document.export_to_markdown())
    #print(result.document.export_to_markdown())
    return True

def main():
    logging.basicConfig(level=logging.INFO)

    from dotenv import load_dotenv
    load_dotenv(override=True)

    input_doc_path = "./PDF/Primera_División_del_Perú.pdf"
    #input_doc_path = "./PDF/vacaciones.pdf"

    # Track which examples ran
    results = {
        # "LM Studio": run_lmstudio_example(input_doc_path),
        "Ollama": run_ollama_example(input_doc_path),
        # "VLLM": run_vllm_example(input_doc_path),
        # "watsonx.ai": run_watsonx_example(input_doc_path),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ran = [name for name, success in results.items() if success]
    skipped = [name for name, success in results.items() if not success]

    if ran:
        print(f"\n✓ Examples that ran successfully ({len(ran)}):")
        for name in ran:
            print(f"  - {name}")

    if skipped:
        print(f"\n⊘ Examples that were skipped ({len(skipped)}):")
        for name in skipped:
            reason = "Server not running"
            if name == "watsonx.ai":
                if os.environ.get("CI"):
                    reason = "Running in CI environment"
                else:
                    reason = "Credentials not found (WX_API_KEY, WX_PROJECT_ID)"
            print(f"  - {name}: {reason}")

    print()


if __name__ == "__main__":
    main()