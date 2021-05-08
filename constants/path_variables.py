import os

INPUT_FILE_PATHS = {
    "train.x": os.path.join(os.getcwd(), "data", "train.txt"),
    "train.y": os.path.join(os.getcwd(), "data", "train.labels"),
    "val.x": os.path.join(os.getcwd(), "data", "val.txt"),
    "val.y": os.path.join(os.getcwd(), "data", "val.labels"),
    "test.x": os.path.join(os.getcwd(), "data", "test.txt"),
    "submission": os.path.join(os.getcwd(), "submission.txt")
}