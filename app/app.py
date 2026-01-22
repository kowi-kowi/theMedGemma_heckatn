import gradio as gr
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from PIL import Image
import json
import time

# Defin device
DEVICE = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def load_medgemma_model():

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=dtype,
        device=DEVICE
    )
    return pipe

BASE_MODEL = "google/gemma-2b-it"
LORA_PATH = "./triage_lora"

triage_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

triage_model = PeftModel.from_pretrained(base_model, LORA_PATH)
triage_model.eval()
pipe = load_medgemma_model()

def run_medgemma(image):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert cardiologist."}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this 12-lead ECG.\n"
                        "Provide a concise clinical summary including:\n"
                        "- rhythm\n"
                        "- heart rate\n"
                        "- ST-segment changes\n"
                        "- T-wave abnormalities\n"
                        "- conduction abnormalities\n"
                        "- other notable findings\n\n"
                        "Do not provide a diagnosis."
                    )
                },
                {"type": "image", "image": image}
            ]
        }
    ]

    output = medgemma_pipe(text=messages, max_new_tokens=200)
    summary = output[0]["generated_text"][-1]["content"]
    return summary




def run_triage(symptoms, ecg_summary):
    prompt = f"""<user>
    PATIENT SYMPTOMS:
{symptoms}

ECG SUMMARY:
{ecg_summary}

</user>
TASK:
Assess cardiac risk and assign triage level.
List red flags.
Suggest next clinical actions.
Do not provide a diagnosis.
</user>
<assistant>
"""

    inputs = triage_tokenizer(prompt, return_tensors="pt").to(triage_model.device)

    with torch.no_grad():
        outputs = triage_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=False
        )

    text = triage_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # wyciągnij tylko część JSON
    if "<assistant>" in text:
        text = text.split("<assistant>")[-1].strip()

    return text




def parse_triage_output(text):
    try:
        data = json.loads(text)
    except:
        return {
            "triage_level": "UNKNOWN",
            "red_flags": [],
            "suggested_actions": [],
            "raw": text
        }

    return data


def triage_badge(level):
    if level == "GREEN":
        return "🟢 LOW RISK"
    if level == "YELLOW":
        return "🟡 MODERATE RISK"
    if level == "RED":
        return "🔴 HIGH RISK"
    return "⚪ UNKNOWN"


# =========================
# MAIN PIPELINE FUNCTION
# =========================

def full_pipeline(symptoms, image):
    start = time.time()

    # 1. MedGemma
    ecg_summary = run_medgemma(image)

    # 2. Triage model
    triage_text = run_triage(symptoms, ecg_summary)

    # 3. Parse
    triage_data = parse_triage_output(triage_text)

    badge = triage_badge(triage_data.get("triage_level"))

    flags = "\n".join([f"- {x}" for x in triage_data.get("red_flags", [])])
    actions = "\n".join([f"- {x}" for x in triage_data.get("suggested_actions", [])])

    elapsed = f"{time.time() - start:.1f}s"

    return (
        ecg_summary,
        badge,
        flags,
        actions,
        elapsed
    )




with gr.Blocks() as demo:


    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("logo.png", show_label=False, height=120)

        with gr.Column(scale=4):
            gr.Markdown(
            """
            # ❤️🩺📈 Rural Cardiac Triage Assistant  
            ### AI-assisted cardiac triage for rural and low-resource settings  

            **Multimodal ECG analysis + clinical triage support**  

            ⚠️ *Decision support only. Not a diagnostic or medical device.*
            """
            )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="ECG Image")
            symptoms_input = gr.Textbox(
                label="Patient symptoms",
                placeholder="e.g. chest pain, shortness of breath, dizziness"
            )
            run_btn = gr.Button("Run Triage Assessment")

        with gr.Column():
            ecg_summary_out = gr.Textbox(label="ECG Summary (MedGemma)", lines=8)
            triage_badge_out = gr.Textbox(label="Triage Level")
            red_flags_out = gr.Textbox(label="Red Flags", lines=6)
            actions_out = gr.Textbox(label="Suggested Actions", lines=6)
            time_out = gr.Textbox(label="Processing time")

    run_btn.click(
        fn=full_pipeline,
        inputs=[symptoms_input, image_input],
        outputs=[
            ecg_summary_out,
            triage_badge_out,
            red_flags_out,
            actions_out,
            time_out
        ]
    )

demo.launch()