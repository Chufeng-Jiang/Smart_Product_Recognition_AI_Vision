__all__ = ["train_step", "test_step"]

import torch
def train_step(classifier, train_loader, loss, optimizer, device):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)    
        loss_value = loss(outputs, labels)  
        loss_value.backward()   
        optimizer.step()    
        optimizer.zero_grad()   
        total_loss += loss_value.item()
    return total_loss / len(train_loader)

def test_step(classifier, test_loader, loss, device):
    total_loss = 0
    correct_num = 0    

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)   
            loss_value = loss(outputs, labels) 
            total_loss += loss_value.item()
            pred = outputs.argmax(dim=1)
            correct_num += pred.eq(labels).sum()

    return total_loss / len(test_loader), correct_num