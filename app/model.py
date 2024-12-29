from transformers import pipeline

vqa_pipeline = pipeline("visual-question-answering", model="microsoft/git-base-vqa")
