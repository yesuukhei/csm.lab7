from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

image = Image.open("./5.jpg").convert("RGB")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values, num_beams=4, early_stopping=True)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def accuracy(gt, pred):
    gt = gt.strip().lower()
    pred = pred.strip().lower()
    correct = sum(1 for a, b in zip(gt, pred) if a == b)
    return round((correct / len(gt)) * 100, 2) if gt else 0

with open("./handwritten4.txt", "r") as f:
    ground_truth = f.read()

acc = accuracy(ground_truth, generated_text)

print("Танигдсан текст:", generated_text)
print(f"Accuracy: {acc:.2f}%")
