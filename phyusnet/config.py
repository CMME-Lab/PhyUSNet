# Datasets Related Variables that I can use if the CLI arguments are not provided

DATASET_PATH = (
    "/home/user/data/phyusformer_data/physics_data/paths.npz"  # npz file path
)
ROOT_US30K_PATH = "/home/user/data/phyusformer_data/post_miccai_exps/data/US30k/US30K/"  # root path for the US30K dataset
VISUALIZE_DATASET = False
SHIFT_BBOX = 10
THRESHOLD = 0.5
DEVICE_ID = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
INPUT_CHANNELS = 3
NUM_CLASSES = 1

# Model Related Variables that I can use if the CLI arguments are not provided
MODEL_NAME = "unet"
LAMBDA_WEIGHT = 0.5
SMOOTH = 1e-8
FROM_LOGITS = True
# Training Related Variables that I can use if the CLI arguments are not provided
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
DROPOUT_RATE = 0.5
DROPOUT_PROB = 0.5
DROPOUT_TYPE = "dropout"
TRAIN_METRICS_PRINT_FREQUENCY = 10
PATIENCE_EPOCH = 10

# Evaluation Related Variables that I can use if the CLI arguments are not provided


# Wandb Related Variables that I can use if the CLI arguments are not provided
WANDB_PROJECT = f"phyusformer_experiments"
WANDB_ENTITY = "user"
WANDB_MODE = "online"
STEP_SIZE = 10
GAMMA = 0.1
