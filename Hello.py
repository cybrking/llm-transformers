import streamlit as st
import torch
from transformers import Phi3ForSequenceClassification, Phi3Tokenizer

# Load the Phi-3 model and tokenizer
try:
    model = Phi3ForSequenceClassification.from_pretrained("phi/phi-3")
    tokenizer = Phi3Tokenizer.from_pretrained("phi/phi-3")
except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")

# Create a Streamlit app
st.title("Phi-3 Text Classification App")

# Add a text input field
input_text = st.text_input("Enter some text:")

# Add a button to trigger the model prediction
if st.button("Classify"):
    if not input_text.strip():  # Check for empty input
        st.error("Please enter some text.")
    else:
        try:
            # Tokenize the input text
            inputs = tokenizer(input_text, return_tensors="pt")

            # Run the model prediction
            outputs = model(**inputs)

            # Get the predicted class label
            predicted_label = torch.argmax(outputs.logits)

            # Map the predicted label to a human-readable class label
            class_labels = ["label1", "label2", ...]  # Define your class labels here
            predicted_class = class_labels[predicted_label]

            # Display the predicted label
            st.write(f"Predicted label: {predicted_class}")
        except Exception as e:
            st.error(f"Error during model prediction: {e}")
