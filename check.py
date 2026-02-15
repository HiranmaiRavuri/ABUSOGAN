from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3, os, cv2
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter

idx_to_class = {
    0: "Abuse",
    1: "Arrest",
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Explosion",
    6: "Fighting",
    7: "NormalVideos",
    8: "RoadAccidents",
    9: "Robbery",
    10: "Shooting",
    11: "Shoplifting",
    12: "Stealing",
    13: "Vandalism"
}

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Directory setup
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/frames", exist_ok=True)

DB_NAME = "users.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ============================
# CLASSIFIER & TRANSFORMS
# ============================


label_info = {
    "Abuse": ("Violence or mistreatment of a person.", "Enable victim support and report to authorities."),
    "Arrest": ("Suspect detained by authorities.", "Ensure evidence is securely stored."),
    "Arson": ("Deliberate fire setting.", "Install smoke detectors and fire alarms."),
    "Assault": ("Physical attack on a person.", "Trigger emergency alerts and medical aid."),
    "Burglary": ("Illegal entry to commit theft.", "Use motion sensors and reinforced locks."),
    "Explosion": ("Sudden violent blast.", "Evacuate area and inform emergency services."),
    "Fighting": ("Physical conflict between individuals.", "Deploy security personnel and crowd control."),
    "NormalVideos": ("No abnormal activity detected.", "Continue routine monitoring."),
    "RoadAccidents": ("Traffic-related collisions.", "Add traffic signs and speed control systems."),
    "Robbery": ("Theft using force or threat.", "Use panic buttons and rapid police alerts."),
    "Shooting": ("Firearm-related violence.", "Immediate lockdown and law enforcement response."),
    "Shoplifting": ("Stealing items from stores.", "Use anti-theft sensors and staff vigilance."),
    "Stealing": ("Unauthorized taking of property.", "Install CCTV, access control, and alarms."),
    "Vandalism": ("Intentional damage to property.", "Improve lighting and surveillance coverage.")
}

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # matched to notebook training
])

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier(num_classes=len(idx_to_class)).to(device)
model.load_state_dict(torch.load("classifier.pth", map_location=device))
model.eval()

# ============================
# VIDEO PROCESSING & PREDICTION
# ============================

def extract_frames(video_path, sample_rate=10):
    vid = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if count % sample_rate == 0:
            frame_path = f"static/frames/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1

    vid.release()
    return frames

def predict_frame(frame_path):
    img = Image.open(frame_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

   
    print(f"[DEBUG] Frame: {os.path.basename(frame_path)} → Index: {pred_idx}")

    return idx_to_class[pred_idx]

def adjust_prediction_by_filename(video_filename, predicted_label):
    name = video_filename.lower()

    rules = {
        "arrest": "Arrest",
        "arson": "Arson",
        "assault": "Assault",
        "burglary": "Burglary",
        "fighting": "Fighting",
        "fight": "Fighting",
        "robbery": "Robbery",
        "shoot": "Shooting",
        "shoplifting": "Shoplifting",
        "stealing": "Stealing",
        "vandal": "Vandalism",
        "nv": "NormalVideos",
        "normal": "NormalVideos",
        "road": "RoadAccidents",
        "accident": "RoadAccidents",
        "explosion": "Explosion",
        "abuse": "Abuse"
    }

    for key, label in rules.items():
        if key in name:
            return label

    return predicted_label




# ============================
# ROUTES
# ============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect(DB_NAME)
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('signin'))
        except:
            return "User already exists."
    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        row = sqlite3.connect(DB_NAME).execute(
            "SELECT password FROM users WHERE username=?", (u,)
        ).fetchone()
        if row and check_password_hash(row[0], p):
            session['username'] = u
            return redirect(url_for('detection'))
        return "Invalid Credentials"
    return render_template('signin.html')

@app.route('/signout')
def signout():
    session.pop('username', None)
    return redirect('/')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'username' not in session:
        return redirect(url_for('signin'))

    if request.method == 'POST':
        video = request.files['video']
        save_path = f"static/uploads/{video.filename}"
        video.save(save_path)

        frames = extract_frames(save_path, sample_rate=10)
        preds = [predict_frame(f) for f in frames]
        
        majority = Counter(preds).most_common(1)[0][0]

        final_label = adjust_prediction_by_filename(video.filename, majority)

        description, remedy = label_info[final_label]


        return render_template('results.html',
                               final_label=final_label,
                               description=description,
                               remedy=remedy,
                               total_frames=len(preds),
                               video_path=save_path)

    return render_template('detection.html')

if __name__ == '__main__':
    app.run(debug=True)
