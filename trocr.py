import difflib
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2

# Preprocess with OpenCV
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh).convert("RGB")

# Load and preprocess image
image_path = './3.jpg'
image = preprocess_image(image_path)

# Load model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# Predict
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def accuracy(gt, pred):
    gt = gt.strip().lower()
    pred = pred.strip().lower()
    correct = sum(1 for a, b in zip(gt, pred) if a == b)
    return round((correct / len(gt)) * 100, 2) if gt else 0

with open("./handwritten3.txt", "r") as f:
    ground_truth = f.read()

acc = accuracy(ground_truth, generated_text)

print("Text:", generated_text)
print(f"Accuracy: {acc:.2f}%")
