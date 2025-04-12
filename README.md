# Eye Blink Counter – Left vs Right Eye Detection with OpenCV & MediaPipe 👁️👁️

Ever wondered which eye you blink more often? This fun computer vision project uses Python, OpenCV, and MediaPipe to track your facial landmarks and count how often your **left** and **right** eyes blink.

![Demo GIF](media/demo.gif)

## 📦 Features

- Real-time eye tracking using your webcam
- Detects individual left and right eye closures
- Sound notification on blink
- Simple visual interface with FPS and blink counters

## 🧠 How It Works

This project uses the Eye Aspect Ratio (EAR) method to detect whether your eye is open or closed. Based on six landmark points around each eye, we track and count blink events per eye.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/eye-blink-counter.git
cd eye-blink-counter
```

### 2. Install dependencies

```bash
pip install opencv-python mediapipe pygame numpy matplotlib
```

### 3. Place your click sound

Download a click sound like [this one](https://pixabay.com/sound-effects/click-124467/) and put it in the root folder.

Update the file path in `Eyeblinker.py` if necessary:
```python
click_sound = pygame.mixer.Sound("click-124467.mp3")
```

### 4. Run the application

```bash
python Eyeblinker.py
```

Press `Q` to exit.

---

## 🖼️ Where to Put the GIF

- Create a folder called `media` in the project root.
- Save your recorded GIF there as `demo.gif`.
- Your folder structure should look like this:
```
eye-blink-counter/
│
├── FaceMeshModule.py
├── Eyeblinker.py
├── click-124467.mp3
├── README.md
└── media/
    └── demo.gif
```

---

## 🛠️ Technologies Used

- [OpenCV](https://opencv.org/) – image processing & webcam capture
- [MediaPipe](https://google.github.io/mediapipe/) – facial landmark detection
- [NumPy](https://numpy.org/) – distance calculations
- [PyGame](https://www.pygame.org/) – sound playback

---

## 📈 Possible Improvements

- Track blink duration
- Visualize blink rate over time
- Add fatigue detection logic
- Deploy to a web or mobile app

---

## 🧑‍💻 Author

**[Mert Akgül]** – *Computer Vision & AI Enthusiast*  
[Portfolio](https://medium.com/@Mert.A/list/projects-6f9bb92a3c21) | [Blog](https://medium.com/@Mert.A) | [LinkedIn](https://www.linkedin.com/in/mert-akgül)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
