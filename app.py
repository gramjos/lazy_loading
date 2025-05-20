import gradio as gr
import torch
from functools import partial
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration as psg,
    Pix2StructProcessor as ps
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/pix2struct-docvqa-large"

model = None
processor = None

def load_model(progress=gr.Progress()):
    global model, processor
    if model is None or processor is None:
        progress(0, desc="Loading model…")
        model = psg.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
        progress(0.5, desc="Loading processor…")
        processor = ps.from_pretrained(MODEL_NAME)
        progress(1, desc="Ready!")

def generate(model, processor, img, questions):
    inputs = processor(images=[img]*len(questions),
                       text=questions,
                       return_tensors="pt").to(DEVICE)
    model.eval()
    with torch.inference_mode():
        preds = model.generate(**inputs, max_new_tokens=1028)
    return list(zip(
        questions,
        processor.batch_decode(preds, skip_special_tokens=True),
    ))

def run_doc_vqa(file_path, questions, page_no=1, progress=gr.Progress()):
    load_model(progress)
    progress(0.6, desc="Converting to image…")
    image = convert_to_image(file_path, page_no)
    progress(0.8, desc="Running inference…")
    return generate(model, processor, image, questions)

def collect_inputs(file_obj, texts, progress=gr.Progress()):
    if file_obj is None:
        return "Please upload a file first."
    questions = [q for q in texts.splitlines() if q.strip()]
    if not questions:
        return "Please enter at least one question."
    return run_doc_vqa(file_obj.name, questions, progress=progress)

with gr.Blocks() as demo:
    demo.load(fn=load_model, inputs=None, outputs=None)
    with gr.Tabs():
        with gr.TabItem("Document QA"):
            file_input = gr.File(type="filepath")
            questions  = gr.Textbox(lines=5, max_lines=10)
            submit     = gr.Button("Submit")
            output     = gr.Textbox(lines=10)
        with gr.TabItem("Log"):
            gr.Markdown("📝 Nothing here… for now!")

    submit.click(
        fn=collect_inputs,
        inputs=[file_input, questions],
        outputs=[output],
        show_progress="full",
        show_progress_on=[output],
        concurrency_limit=1,
        queue=True
    )

    demo.queue(
        default_concurrency_limit=8,
        max_size=16
    )

demo.launch()
