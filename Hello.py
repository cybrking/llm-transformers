import streamlit as st
from transformers import AutoTokenizer, AutoModel
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the model and processor
model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

def predict(image, question):
    # Preprocess the image and question
    encoding = processor(image, question, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            pixel_values=encoding.pixel_values,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return generated_text

def main():
    st.title("Document Visual Question Answering")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Get the question from the user
        question = st.text_input("Ask a question about the image:")

        if question:
            # Generate the answer
            answer = predict(image, question)
            st.success(answer)

if __name__ == "__main__":
    main()