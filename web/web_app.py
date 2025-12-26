from flask import Flask, request, json, render_template, jsonify
import torch
import numpy as np
from io import BytesIO
import base64
from flask import send_from_directory

from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
import os
from PIL import Image
import sys

# ============ Path Settings ============
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(f"Project Root Path: {project_root}")
print(f"Python Path: {sys.path[:3]}")
# ============ Path Settings End ============

from image_denoising import denoising_config
from image_denoising import denoising_model
from image_classification import classification_config
from image_classification import classification_model
from image_similarity import similarity_config
from image_similarity import similarity_model


app = Flask(__name__, static_folder='../common/dataset')

@app.route('/logo/<filename>')
def serve_logo(filename):
    return send_from_directory('./logo', filename)

@app.route('/pictures/<filename>')
def serve_pictures(filename):
    return send_from_directory('./pictures', filename)

print("Startting model loading...")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading denoising model")
denoiser = denoising_model.ConvDenoiser()
denoiser.load_state_dict(torch.load(
    os.path.join('../image_denoising', denoising_config.DENOISER_MODEL_NAME), map_location=device))
denoiser.to(device)
print("Loading denoising model completed")

print("Loading classification model")
classifier = classification_model.Classifier()
classifier.load_state_dict(torch.load(
    os.path.join('../image_classification', classification_config.CLASSIFIER_MODEL_NAME), map_location=device))
classifier.to(device)
print("Loading classification model completed")

print("Loading embedding model")
encoder = similarity_model.ConvEncoder()  
encoder.load_state_dict(
    torch.load(
        os.path.join(
            '..',
            similarity_config.PACKAGE_NAME, similarity_config.ENCODER_MODEL_NAME),
        map_location=device))
encoder.to(device) 
print("Loading embedding model completed")

print("Loading vector database...")
embedding = np.load(os.path.join(
    '..',
    similarity_config.PACKAGE_NAME,
    similarity_config.EMBEDDING_NAME)
)
print("Loading vector database completed")

def compute_similar_images(image_tensor, num_images, embedding, device):
    image_tensor = image_tensor.to(device)  
    
    with torch.no_grad():  
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)  
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist() 
    return indices_list

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/denoising', methods=['POST'])
def get_denoised_image():
    image = request.files["image"]
    image = Image.open(image.stream).convert("RGB")
    t = T.Compose([T.Resize((68, 68)), T.ToTensor()])
    image_tensor = t(image)

    noisy_img = image_tensor + denoising_config.NOISE_FACTOR * torch.randn(*image_tensor.shape)
    noisy_img = torch.clip(noisy_img, 0., 1.)
    noisy_img = noisy_img.unsqueeze(0)

    with torch.no_grad():
        noisy_img = noisy_img.to(device)
        denoised_image = denoiser(noisy_img)

    denoised_image = denoised_image.squeeze(0).cpu()  
    denoised_image = denoised_image.permute(1, 2, 0).numpy() * 255 
    noisy_img = noisy_img.squeeze(0).cpu()
    noisy_img = noisy_img.permute(1, 2, 0).numpy() * 255
    denoised_image = Image.fromarray(denoised_image.astype('uint8'))
    noisy_img = Image.fromarray(noisy_img.astype('uint8'))

    def encode_image(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return (
        json.dumps(
            {
                "noisy_img": encode_image(noisy_img),
                "denoised_image": encode_image(denoised_image)
            }),
        200,
        {"ContentType": "application/json"},
    )

@app.route("/classification", methods=["POST"])
def classification():
    image = request.files["image"]
    image = Image.open(image.stream).convert("RGB")
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    image_tensor = t(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        classification = classifier(image_tensor)

    return "The product category you searched for is: " + classification_config.classification_names[np.argmax(classification.cpu().detach().numpy())]

@app.route("/simimages", methods=["POST"])
def simimages():
    image = request.files["image"]
    image = Image.open(image.stream).convert("RGB")
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    image_tensor = t(image)
    image_tensor = image_tensor.unsqueeze(0)
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    return (
        json.dumps({"indices_list": indices_list[0]}),
        200,
        {"ContentType": "application/json"},
    )


if __name__ == "__main__":
    app.run(debug=False, port=7860)
