import gradio as gr
import torch
from transformers import pipeline
from PIL import Image
import signal

# Defin device
device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Timeout handler

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Model inference timed out.")

signal.signal(signal.SIGALRM, timeout_handler)

# Load MedGemma model

def load_medgemma_model():

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=dtype,
        device=device
    )
    return pipe

pipe = load_medgemma_model()
# System prompt
SYSTEM_PROMPT = (
    "You are MedGemma, an AI model specialized in analyzing ECG images "
    "and providing clinical decision support for cardiac risk triage. "
    "Given an ECG image and patient symptoms, you will identify key cardiac findings, "
    "highlight red flags, and provide a triage recommendation."
)
# Triage logic
def triage_logic(analysis_text):

    text = analysis_text.lower()

    high_risk = [
        "st elevation", "acute infarction", "myocardial infarction",
        "acute ischemia", "ventricular tachycardia", "ventricular fibrillation",
        "heart block", "cardiogenic shock"
    ]

    moderate_risk = [
        "possible ischemia", "t wave inversion", "st depression",
        "arrhythmia", "borderline", "abnormal"
    ]

    if any(k in text for k in high_risk):
        return "🔴 **HIGH RISK** – urgent referral / emergency evaluation recommended"

    if any(k in text for k in moderate_risk):
        return "🟡 **MODERATE RISK** – clinical correlation and further evaluation advised"

    return "🟢 **LOW RISK** – no immediate red flags detected (monitor & reassess)"


# Analyze function
def analyze(ecg_image, symptoms):
    if ecg_image is None:
        return "Please upload an ECG image!"

    try:
        signal.alarm(90)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                                Patient symptoms: {symptoms}
                                Analyze the ECG and list cardiac findings.
                                Focus on red flags and uncertainty.
                """
                    },
                    {
                        "type": "image",
                        "image": ecg_image
                    }
                ]
            }
        ]

        output = pipe(messages, max_new_tokens=250)
        analysis_text = output[0]["generated_text"][-1]["content"]

        triage = triage_logic(analysis_text)

        signal.alarm(0)

        return f"""## 🩺 ECG Analysis
                    {analysis_text}
                    ---
                    ## 🚦 Triage Recommendation
                    {triage}
                    ---

                    ⚠️ *This is a clinical decision support prototype, not a diagnostic tool.*
            """

    except TimeoutException:
        return "⏱️ Analysis timed out. Please try again with a clearer ECG image."

    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"




# GUI with Gradio
logo = Image.open("app/logo.png")


    
with gr.Blocks() as demo:
    gr.Image(value=logo, label="App Logo", type="pil", elem_id="app-logo", interactive=False, height=150)
    gr.Markdown("# Rural Cardiac Triage Assistant")
    gr.Markdown(
        "**AI-powered decision support for early cardiac risk detection**--"
        "⚠️ Not a diagnostic device. For educational and clinical support use only."
    )

    with gr.Row():
        ecg = gr.Image(type="pil", label="Upload ECG image")
        symptoms = gr.Textbox(
            lines=5,
            label="Patient symptoms",
            placeholder="e.g. chest pain, shortness of breath, dizziness"
        )

    analyze_btn = gr.Button("Analyze ECG", variant="primary")

    output = gr.Markdown()

    analyze_btn.click(
        fn=analyze,
        inputs=[ecg, symptoms],
        outputs=output
    )

demo.launch()