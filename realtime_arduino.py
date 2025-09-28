import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import serial
import time
from collections import deque, Counter

# ---------------------------
# Arduino setup
# ---------------------------
arduino = None
try:
    arduino = serial.Serial(port='COM8', baudrate=9600, timeout=1)
    time.sleep(2)
    print("✅ Connected to Arduino on COM8")
except Exception as e:
    print("⚠️ Could not connect to Arduino:", e)

# ---------------------------
# Define Model
# ---------------------------
class WasteCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(WasteCNN, self).__init__()
        self.base_model = models.resnet18(weights=None)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load model
device = torch.device("cpu")
model = torch.load("cnn_full.pth", map_location=device, weights_only=False)
model.eval()
print("✅ Model loaded successfully")

# ---------------------------
# Transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Waste categories
# ---------------------------
classes = ["Biodegradable", "Recyclable", "Residual"]

# ---------------------------
# Voting system
# ---------------------------
PREDICTION_BUFFER_SIZE = 5   # smaller = faster response
CONFIDENCE_THRESHOLD = 0.70
ACTION_COOLDOWN = 3          # seconds between bin actions

pred_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
last_final_class = None
last_action_time = 0

# ---------------------------
# Camera capture
# ---------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0)

    # Prediction with probabilities
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().tolist()

    pred_buffer.append(probs)

    # Once enough frames are collected
    if len(pred_buffer) == PREDICTION_BUFFER_SIZE:
        avg_probs = [sum(p[i] for p in pred_buffer) / len(pred_buffer) for i in range(len(classes))]
        max_idx = int(avg_probs.index(max(avg_probs)))
        majority_class = classes[max_idx]
        avg_conf = avg_probs[max_idx]

        # Show on screen
        text = f"Decision: {majority_class} ({avg_conf:.2f})"
        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        prob_text = f"B:{avg_probs[0]:.2f} R:{avg_probs[1]:.2f} S:{avg_probs[2]:.2f}"
        cv2.putText(frame, prob_text, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Decision → send to Arduino
        if avg_conf >= CONFIDENCE_THRESHOLD:
            current_time = time.time()
            if majority_class != last_final_class and (current_time - last_action_time) > ACTION_COOLDOWN:
                print(f"✅ Classified: {majority_class} ({avg_conf:.2f})")

                # Send short code
                code = {"Biodegradable": "B", "Recyclable": "R", "Residual": "S"}[majority_class]
                if arduino:
                    arduino.write((code + "\n").encode())
                    print(f"➡️ Sent to Arduino: {code}")

                last_final_class = majority_class
                last_action_time = current_time

    # Show video
    cv2.imshow("Smart Waste Segregator", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
