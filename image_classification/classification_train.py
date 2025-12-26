import torch
from torch import nn, optim
import torchvision.transforms as T  
from torch.utils.data import Dataset, DataLoader, random_split   

from tqdm import tqdm    

from common import utils
from classification_config import *
from classification_data import ImageLabelDataset
from classification_model import Classifier
from classification_engine import train_step, test_step

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(SEED)

    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])

    dataset = ImageLabelDataset(IMG_PATH, FASHION_LABELS_PATH, transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    classifier = Classifier()
    loss = nn.CrossEntropyLoss()  
    optimizer = optim.AdamW(classifier.parameters(), lr=LEARNING_RATE)

    classifier.to(device)
    min_test_loss = 9999    

    for epoch in tqdm(range(EPOCHS)):
        train_loss = train_step(classifier, train_loader, loss, optimizer, device)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss}")

        test_loss, test_correct_num = test_step(classifier, test_loader, loss, device)
        accuracy = test_correct_num / len(test_dataset)
        print(f"\nEpoch {epoch+1}/{EPOCHS}, Test Loss: {test_loss:.6f}, Accuracy: {accuracy:.6f}")

        if test_loss < min_test_loss:
            print("Saving the model ...")
            min_test_loss = test_loss
            torch.save(classifier.state_dict(), CLASSIFIER_MODEL_NAME)
        else:
            print("No Save！Loss did not improve.")

    print("--- Training Done! ---")