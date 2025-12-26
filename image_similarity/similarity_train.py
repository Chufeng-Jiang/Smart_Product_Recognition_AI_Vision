import torch
from torch import nn, optim
import torchvision.transforms as T  
from torch.utils.data import Dataset, DataLoader, random_split    

import numpy as np
from tqdm import tqdm 

from common import utils
from similarity_config import *
from similarity_data import ImageDataset
from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import train_step, test_step, create_embedding

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    utils.seed_everything(SEED)

    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])

    dataset = ImageDataset(IMG_PATH, transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    full_loader = DataLoader(dataset, batch_size=FULL_BATCH_SIZE)

    encoder = ConvEncoder()
    decoder = ConvDecoder()
    loss = nn.MSELoss()    

    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(autoencoder_params, lr=LEARNING_RATE)

    encoder.to(device)
    decoder.to(device)
    min_test_loss = 9999   

    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(encoder, decoder, train_loader, loss, optimizer, device)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss}")

        test_loss = test_step(encoder, decoder, test_loader, loss, device)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Test Loss: {test_loss}")

        if test_loss < min_test_loss:
            print("Saving the model ...")
            min_test_loss = test_loss
            torch.save(encoder.state_dict(), ENCODER_MODEL_NAME)
            torch.save(decoder.state_dict(), DECODER_MODEL_NAME)
        else:
            print("Not saving! The loss did not improve.")

    print("--- Training Done!---")

    encoder_state_dict = torch.load(ENCODER_MODEL_NAME, map_location=device)
    encoder.load_state_dict(encoder_state_dict)

    embeddings = create_embedding(encoder, full_loader, device)
    vec_embeddings = embeddings.detach().numpy().reshape(embeddings.shape[0], -1)
    np.save(EMBEDDING_NAME, vec_embeddings)

    print("Embedding shape:", embeddings.shape)
    print("Vector embedding shape:", vec_embeddings.shape)