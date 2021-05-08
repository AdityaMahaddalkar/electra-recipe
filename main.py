from scripts.model_predictor import predict_test
from scripts.model_trainer import train_model

if __name__ == '__main__':
    history, model = train_model()
    predict_test(model)