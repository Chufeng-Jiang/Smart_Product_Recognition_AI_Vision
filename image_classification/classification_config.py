FASHION_LABELS_PATH = "../common/fashion-labels.csv"
IMG_PATH = "../common/dataset/"
IMG_HEIGHT = 64
IMG_WIDTH = 64

SEED = 42               
TRAIN_RATIO = 0.75     
TEST_RATIO = 1 - TRAIN_RATIO

LEARNING_RATE = 0.001   
EPOCHS = 20             
TRAIN_BATCH_SIZE = 128   
TEST_BATCH_SIZE = 128

PACKAGE_NAME = 'image_classification'
CLASSIFIER_MODEL_NAME = 'classifier.pt'

classification_names = {
    0: 'Top',  
    1: 'Shoes',     
    2: 'Bag',       
    3: 'Bottom',  
    4: 'Watch'      
}