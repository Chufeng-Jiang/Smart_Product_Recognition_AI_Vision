import os
import sys
import torch
import numpy as np
from io import BytesIO
import base64
from flask import Flask, request, json, render_template, jsonify, send_from_directory

from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from image_denoising import denoising_config, denoising_model
    from image_classification import classification_config, classification_model
    from image_similarity import similarity_config, similarity_model
    print("Successfully imported custom modules")
except ImportError as e:
    print(f"Import error: {e}")


app = Flask(__name__)

# ============ Path Settings（Hugging Face Spaces） ============
# Hugging Face Spaces  /home/user/app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATASET_DIR = os.path.join(BASE_DIR, "common", "dataset")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Models directory: {MODELS_DIR}")


DENOISER_PATH = os.path.join(MODELS_DIR, "denoiser.pt")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "classifier.pt")
ENCODER_PATH = os.path.join(MODELS_DIR, "deep_encoder.pt")
EMBEDDING_PATH = os.path.join(EMBEDDINGS_DIR, "data_embedding.npy")

# ============ Static Files Route ============
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route('/static/logo/<filename>')
def serve_logo(filename):
    logo_dir = os.path.join(BASE_DIR, "logo")
    if not os.path.exists(logo_dir):
        os.makedirs(logo_dir, exist_ok=True)
    return send_from_directory(logo_dir, filename)

@app.route('/static/pictures/<filename>')
def serve_pictures(filename):
    pictures_dir = os.path.join(BASE_DIR, "pictures")
    if not os.path.exists(pictures_dir):
        os.makedirs(pictures_dir, exist_ok=True)
    return send_from_directory(pictures_dir, filename)

@app.route('/dataset/<filename>')
def serve_dataset_short(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/common/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory(DATASET_DIR, filename)

@app.route('/common/dataset/<filename>')
def serve_dataset_image(filename):
    dataset_dir = os.path.join(BASE_DIR, "common", "dataset")
    return send_from_directory(dataset_dir, filename)

print("Starting model loading...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

denoiser = None
classifier = None
encoder = None
embedding = None

def load_models():
    global denoiser, classifier, encoder, embedding
    
    try:
        print("Loading denoising model...")
        denoiser = denoising_model.ConvDenoiser()
        denoiser.load_state_dict(torch.load(DENOISER_PATH, map_location=device))
        denoiser.to(device)
        denoiser.eval()
        print("Denoising model loaded")
        
        print("Loading classification model...")
        classifier = classification_model.Classifier()
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
        classifier.to(device)
        classifier.eval()
        print("Classification model loaded")
        
        print("Loading embedding model...")
        encoder = similarity_model.ConvEncoder()
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
        encoder.to(device)
        encoder.eval()
        print("Embedding model loaded")
        
        print("Loading vector database...")
        embedding = np.load(EMBEDDING_PATH)
        print(f"Embedding shape: {embedding.shape}")
        print("Vector database loaded")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure all model files are in the correct location:")
        print(f"  - Denoiser: {DENOISER_PATH}")
        print(f"  - Classifier: {CLASSIFIER_PATH}")
        print(f"  - Encoder: {ENCODER_PATH}")
        print(f"  - Embedding: {EMBEDDING_PATH}")


load_models()


def compute_similar_images(image_tensor, num_images, embedding, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        image_embedding = encoder(image_tensor).cpu().detach().numpy()
        
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)
    _, indices = knn.kneighbors(flattened_embedding)
    return indices.tolist()[0]

def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/denoising', methods=['POST'])
def get_denoised_image():
    if denoiser is None:
        return jsonify({"error": "Denoising model not loaded"}), 500
    
    try:
        image = request.files["image"]
        image = Image.open(image.stream).convert("RGB")
        
        t = T.Compose([T.Resize((68, 68)), T.ToTensor()])
        image_tensor = t(image)
        
        noise_factor = denoising_config.NOISE_FACTOR if hasattr(denoising_config, 'NOISE_FACTOR') else 0.3
        noisy_img = image_tensor + noise_factor * torch.randn(*image_tensor.shape)
        noisy_img = torch.clip(noisy_img, 0., 1.)
        noisy_img = noisy_img.unsqueeze(0)
        
        with torch.no_grad():
            noisy_img = noisy_img.to(device)
            denoised_image = denoiser(noisy_img)
        
        denoised_image = denoised_image.squeeze(0).cpu()
        denoised_image = denoised_image.permute(1, 2, 0).numpy() * 255
        noisy_img = noisy_img.squeeze(0).cpu()
        noisy_img = noisy_img.permute(1, 2, 0).numpy() * 255
        
        denoised_pil = Image.fromarray(denoised_image.astype('uint8'))
        noisy_pil = Image.fromarray(noisy_img.astype('uint8'))
        
        return jsonify({
            "noisy_img": encode_image(noisy_pil),
            "denoised_image": encode_image(denoised_pil),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/classification", methods=["POST"])
def classification():
    if classifier is None:
        return jsonify({"error": "Classification model not loaded"}), 500
    
    try:
        image = request.files["image"]
        image = Image.open(image.stream).convert("RGB")
        
        t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        image_tensor = t(image)
        image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            predictions = classifier(image_tensor)

        class_idx = np.argmax(predictions.cpu().detach().numpy())
        
        if hasattr(classification_config, 'classification_names'):
            class_name = classification_config.classification_names[class_idx]
        else:
            class_name = f"Category {class_idx}"
        
        return jsonify({
            "category": class_name,
            "category_id": int(class_idx),
            "confidence": float(predictions.cpu().detach().numpy()[0][class_idx]),
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/simimages", methods=["POST"])
def simimages():
    try:
        user_image = request.files["image"]
        user_image = Image.open(user_image.stream).convert("RGB")

        t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        image_tensor = t(user_image)
        image_tensor = image_tensor.unsqueeze(0)
        
        indices = compute_similar_images(
            image_tensor, num_images=5, embedding=embedding, device=device
        )
        
        image_paths = []
        image_urls = []
        
        for idx in indices:
            img_filename = f"{idx}.jpg"
            
            dataset_path = os.path.join(BASE_DIR, "common", "dataset", img_filename)
            if os.path.exists(dataset_path):
                print(f"✓ Found the pics: {dataset_path}")
            else:
                print(f"✗ Pics Not found: {dataset_path}")
                alt_names = [f"product_{idx}.jpg", f"item_{idx}.jpg", f"img_{idx}.jpg"]
                for alt in alt_names:
                    alt_path = os.path.join(BASE_DIR, "common", "dataset", alt)
                    if os.path.exists(alt_path):
                        img_filename = alt
                        print(f"  Using the alt name: {alt}")
                        break
            
            image_url = f"/common/dataset/{img_filename}"
            image_paths.append({
                "index": int(idx),
                "filename": img_filename,
                "url": image_url,
                "full_path": os.path.join("common", "dataset", img_filename)
            })
            
        
        return jsonify({
            "indices_list": indices,
            "images": image_paths,
            "count": len(indices),
            "status": "success",
            "message": f"Found # {len(indices)} similar products"
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    models_loaded = all([denoiser is not None, classifier is not None, 
                        encoder is not None, embedding is not None])
    
    return jsonify({
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "device": str(device),
        "denoiser_loaded": denoiser is not None,
        "classifier_loaded": classifier is not None,
        "encoder_loaded": encoder is not None,
        "embedding_loaded": embedding is not None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)