import os

class Config:
    DATA_BASE = "./data/"
    EMBEDDINGS_FILE = "embeddings/ml20_pca128.pkl"
    RATINGS_FILE = "ml-20m/ratings.csv"
    CACHE_FILE = "cache/frame_env.pkl"

    FRAME_SIZE = 10
    BATCH_SIZE = 25
    N_EPOCHS = 100
    HIDDEN_SIZE = 256

    LEARNING_RATE = 1e-5
    GAMMA = 0.99
    SOFT_TAU = 0.001
    POLICY_STEP = 10

    MIN_VALUE = -10
    MAX_VALUE = 10

    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    LOG_DIR = './runs'
    MODEL_SAVE_PATH = 'models/ddpg_model.pth'

    @classmethod
    def get_data_path(cls):
        return cls.DATA_BASE

    @classmethod
    def get_embeddings_path(cls):
        return os.path.join(cls.DATA_BASE, cls.EMBEDDINGS_FILE)

    @classmethod
    def get_ratings_path(cls):
        return os.path.join(cls.DATA_BASE, cls.RATINGS_FILE)

    @classmethod
    def get_cache_path(cls):
        return os.path.join(cls.DATA_BASE, cls.CACHE_FILE)
