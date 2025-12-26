IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 64
IMG_WIDTH = 64

SEED = 42               
TRAIN_RATIO = 0.75      
TEST_RATIO = 1 - TRAIN_RATIO

LEARNING_RATE = 0.001   
EPOCHS = 30             
TRAIN_BATCH_SIZE = 32   
TEST_BATCH_SIZE = 32
FULL_BATCH_SIZE = 32

PACKAGE_NAME = 'image_similarity'
ENCODER_MODEL_NAME = 'deep_encoder.pt'
DECODER_MODEL_NAME = 'deep_decoder.pt'
EMBEDDING_NAME = 'data_embedding.npy'