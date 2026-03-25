
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    EasyOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions


def main():
    input_doc_path = "./PDF/vacaciones.pdf"

    # Docling Parse with EasyOCR
    # ----------------------------
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )

    pipeline_options.ocr_options.lang = ["es"]
    accelerator_options = AcceleratorOptions(
        num_threads=6, #Personalizar segun los nucleos de tu procesador
        device=AcceleratorDevice.CPU
    )
    ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options
    pipeline_options.accelerator_options = accelerator_options

    #Mostrar el tiempo que demora la conversion
    import time
    start_time = time.time()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(input_doc_path).document
    end_time = time.time()
    md = doc.export_to_markdown()
    print(f"Tiempo de conversion: {end_time - start_time:.2f} segundos")
    #ruta_archivo = "./temp/Primera_División_del_Perú.md"
    ruta_archivo = "./temp/vacaciones.md"
    import os
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write("# Generado con EasyOCR\n")
        f.write(f"*Tiempo de conversion:* {end_time - start_time:.2f} segundos\n")
        f.write("---\n")
        f.write(md)

if __name__ == "__main__":
    main()