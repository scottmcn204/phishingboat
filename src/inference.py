import onnxruntime as rt
import numpy as np
import pandas as pd
import parsing
import requests

def run_inference(features) : 
    input_data = np.array(features, dtype=np.float32).reshape(1, 15)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: input_data})
    prediction = result[0][0] 
    print("Predicted Output:", prediction)

# Load the ONNX model
session = rt.InferenceSession("models/model.onnx")

url = input("Enter website url: ")
response = requests.get(url)
if response.status_code == 200:
    html_content = response.text
    html_features = parsing.html_features_from_text(html_content)
    url_features = parsing.url_features_inference(url)
    features = [0] + url_features + html_features
    print("HTML Features:", html_features)
    print("URL Features:", url_features)
    print("Combined Features:", features)
    run_inference(features)
else:
    print(f"Couldn't fetch page, Status : {response.status_code} :(")


# url_features = parsing.url_features(url)
# html_features = parsing.hmtl_features(path_to_html)