IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 68
IMG_WIDTH = 68

SEED = 42              
TRAIN_RATIO = 0.75      
TEST_RATIO = 1 - TRAIN_RATIO
NOISE_FACTOR = 0.5     

LEARNING_RATE = 0.001   
EPOCHS = 30             
TRAIN_BATCH_SIZE = 32  
TEST_BATCH_SIZE = 32

PACKAGE_NAME = 'image_denoising'
DENOISER_MODEL_NAME = 'denoiser.pt'