"""
 collect user input: document and questions
    - document: pdf, png, jpg, jpeg
    - question: delineated by new line in the text box

Call model
"""
import gradio as gr
import os
from datetime import datetime
from pdf2image import convert_from_path, convert_from_bytes
import torch
from functools import partial
from PIL import Image
from transformers import Pix2StructForConditionalGeneration as psg
from transformers import Pix2StructProcessor as ps

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache model and processor so that they are only loaded once.
MODEL_NAME = "google/pix2struct-docvqa-large"
model = None
processor = None

def generate(model, processor, img, questions):
  inputs = processor(images=[img for _ in range(len(questions))],
           text=questions, return_tensors="pt").to(DEVICE)
  predictions = model.generate(**inputs, max_new_tokens=1028)
  return zip(questions, processor.batch_decode(predictions, skip_special_tokens=True))

# def convert_pdf_to_image(filename, page_no):
#     return convert_from_path(filename)[page_no-1]
from pathlib import Path
from io import BytesIO

def convert_to_image(input_source, page_no=1):
    """
    Load a page from a PDF or an image file and return it as a PIL.Image.

    Args:
        input_source (str | bytes | Path):
            - If a str or Path ending in .pdf, treated as a PDF file path.
            - If bytes or a path to an image (.jpg, .png, etc.), treated as an image.
        page_no (int): 1-based page number for PDF (ignored for images).

    Returns:
        PIL.Image.Image: the requested page/image.

    Raises:
        ValueError: if the file type is unsupported or page_no > 1 on image input.
    """
    # If raw bytes, try opening as image first
    if isinstance(input_source, (bytes, bytearray)):
        return Image.open(BytesIO(input_source))

    # Otherwise, work with filesystem path
    path = Path(input_source)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = convert_from_path(str(path))
        try:
            return pages[page_no - 1]
        except IndexError:
            raise ValueError(f"PDF only has {len(pages)} pages; page_no={page_no} is out of range.")

    # Common image extensions
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        if page_no != 1:
            raise ValueError("page_no > 1 not supported for image files.")
        return Image.open(path)

    else:
        raise ValueError(f"Unsupported file extension '{suffix}'. "
                         "Expected .pdf or an image format.")


def load_model(progress=gr.Progress()):
    """Load the model and processor if they haven't been loaded yet."""
    global model, processor
    if model is None or processor is None:
        progress(0, desc="Loading model...")
        model = psg.from_pretrained(MODEL_NAME).to(DEVICE)
        progress(0.5, desc="Loading processor...")
        processor = ps.from_pretrained(MODEL_NAME)
        progress(1, desc="Model loaded")


def run_doc_vqa(file_path, questions, page_no=1, progress=gr.Progress()) -> str:
    """Run document VQA and return answers."""
    load_model(progress)

    progress(0.6, desc="Preparing image...")
    image = convert_to_image(file_path, page_no)

    progress(0.8, desc="Running inference...")
    generator = partial(generate, model, processor)
    completions = generator(image, questions)

    progress(1, desc="Done")
    return [i for i in completions]



def collect_inputs(file_obj, texts, log_history, progress=gr.Progress()):
    """
    Process the uploaded file and questions.
    """
    if file_obj is None:
        return "Please upload a file first.", log_history

    # Get file extension
    file_path = file_obj.name
    file_ext = os.path.splitext(file_path)[1].lower()

    # Validate file type
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    if file_ext not in allowed_extensions:
        return f"Invalid file type. Please upload one of: {', '.join(allowed_extensions)}", log_history

    # Process questions
    questions = [line.strip() for line in texts.splitlines() if line.strip()]
    if not questions:
        return "Please enter at least one question.", log_history

    # Create log entry
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # log_entry = f"[{timestamp}] Questions asked:\n" + "\n".join(f"- {q}" for q in questions) + "\n\n"
    # new_log = log_entry + log_history

    results = run_doc_vqa(file_path, questions, page_no=1, progress=progress)
    return results
    # return "\n".join(f"{q}: {a}" for q, a in results)

    # return f"File uploaded: {file_path}\n\nQuestions:\n" + "\n".join(questions), new_log


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Document QA"):
            gr.Markdown("### 📄 Upload a file (PDF, PNG, or JPG):")
            file_input = gr.File(
                label="File Upload",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath"
            )

            gr.Markdown("### ❓ Enter your question(s), one per line:")
            questions = gr.Textbox(
                label="Questions",
                placeholder="Type each question on its own line…",
                lines=5,               # show 5 rows by default
                max_lines=10          # allow up to 10 lines
            )
            submit = gr.Button("Submit")
            output = gr.Textbox(label="Results", lines=10)

        with gr.TabItem("Log"):
            gr.Markdown("### 📝 Activity Log")

    # Connect the submit button after all components are created
    submit.click(
        fn=collect_inputs,
        inputs=[file_input, questions],
        outputs=[output,]
    )

demo.launch()
