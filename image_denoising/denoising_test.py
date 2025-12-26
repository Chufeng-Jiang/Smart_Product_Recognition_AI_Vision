import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split   

import matplotlib.pyplot as plt


from common import utils
from denoising_config import *
from denoising_data import ImageDataset
from denoising_model import ConvDenoiser


def test(denoiser, test_loader, device):
    denoiser.eval()

    data_iter = iter(test_loader)
    noise_images, original_images = next(data_iter)
    print("Testing the noise image shape：", noise_images.shape)

    denoiser = denoiser.to(device)
    noise_images = noise_images.to(device)

    outputs = denoiser(noise_images)
    print("Reconstrcted image shape：", outputs.shape)

    noise_imgs = noise_images.permute(0, 2, 3, 1).cpu().numpy()
    print("Transformed image shape：", noise_imgs.shape)

    output_imgs = outputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    print("Transformed denoised image shape：", output_imgs.shape)

    original_imgs = original_images.permute(0, 2, 3, 1).cpu().numpy()
    print("Transformed original image shape：", original_imgs.shape)
    fig, axes = plt.subplots(3, 10, figsize=(25, 4))
    for imgs, row in zip([noise_imgs, output_imgs, original_imgs], axes):
        for img, ax in zip(imgs, row):
            ax.imshow(img)
            ax.axis('off')
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.seed_everything(SEED)
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor()
    ])

    dataset = ImageDataset(IMG_PATH, transform)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_RATIO, TEST_RATIO])
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    loaded_denoiser = ConvDenoiser()
    model_state_dict = torch.load(DENOISER_MODEL_NAME, map_location=device)
    loaded_denoiser.load_state_dict(model_state_dict)
    loaded_denoiser.to(device)
    test(loaded_denoiser, test_loader, device)