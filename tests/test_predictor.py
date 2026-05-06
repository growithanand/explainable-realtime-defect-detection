from src.data.prepare_data import create_train_val_test_split
from src.inference.predictor import DefectPredictor


DATA_DIR = "data/mvtec_ad/bottle"
MODEL_PATH = "models/resnet18_bottle_binary.pth"


splits = create_train_val_test_split(DATA_DIR)

# Pick one test sample
image_path, true_label = splits["test"][0]

predictor = DefectPredictor(
    model_path=MODEL_PATH,
    defect_threshold=0.5,
)

result = predictor.predict(image_path)

class_names = ["normal", "defective"]

print("Image path:", image_path)
print("True label:", class_names[true_label])
print("Predicted:", result["predicted_class"])
print("Confidence:", round(result["confidence"], 4))
print("Normal probability:", round(result["normal_probability"], 4))
print("Defective probability:", round(result["defective_probability"], 4))
print("Threshold:", result["defect_threshold"])
print("Device:", result["device"])