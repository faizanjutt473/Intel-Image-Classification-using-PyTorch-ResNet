"""Gradio web interface: upload an image and see the predicted scene class.

Run with:  python app.py   (then open the printed local URL)
"""
import gradio as gr
import torch

import config
from dataset import build_transforms
from model import load_checkpoint
from predict import predict_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = load_checkpoint(config.BEST_MODEL_PATH, device)
classes = checkpoint["classes"]
transform = build_transforms(checkpoint["img_size"])[1]


def classify(image):
    if image is None:
        return {}
    return dict(predict_image(model, image, classes, transform, device, top_k=len(classes)))


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a scene image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Intel Scene Classifier",
    description=(f"Classifies natural/urban scenes into: {', '.join(classes)}. "
                 "The model only knows these classes, so unrelated images (people, animals, objects) "
                 "will still be forced into one of them."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
