__all__ = ["train_step", "test_step", "create_embedding"]

import torch

def train_step(encoder, decoder, train_loader, loss, optimizer, device):
    encoder.train()
    decoder.train()

    total_loss = 0.0

    for train_imgs, target_imgs in train_loader:
        train_imgs = train_imgs.to(device)
        target_imgs = target_imgs.to(device)
        en_output = encoder(train_imgs)
        outputs = decoder(en_output)
        loss_value = loss(outputs, target_imgs)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def test_step(encoder, decoder, test_loader, loss, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0.0

    with torch.no_grad():
        for test_imgs, target_imgs in test_loader:
            test_imgs = test_imgs.to(device)
            target_imgs = target_imgs.to(device) 
            en_output = encoder(test_imgs)
            outputs = decoder(en_output)
            loss_value = loss(outputs, target_imgs)
            total_loss += loss_value.item()
    return total_loss / len(test_loader)

def create_embedding(encoder, full_loader, device):
    encoder.eval()
    embeddings = torch.empty(0)
    with torch.no_grad():
        for train_img, target_img in full_loader:
            train_img = train_img.to(device)
            encoded_img = encoder(train_img).cpu()
            embeddings = torch.cat((embeddings, encoded_img), dim=0)
    return embeddings