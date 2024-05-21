import streamlit as st

# Create a Streamlit app
st.title("Hello World App")

# Add a header
st.header("This is a header")

# Add some text
st.write("Hello, World!")

# Add a button
if st.button("Click me!"):
    st.write("Button clicked!")
