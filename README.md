<h1 align="center">GOSSIP</h1>
<p align="center">
  <b>General Observation System for Spontaneous Interpretation of Pedestrians</b>
</p>

Recently, while visiting a museum in London, I came across an interactive art installation that captured live video of pedestrians walking below, detecting them in real-time and assigning humorous, internal-monologue-style speech bubbles over their heads as they moved. 

I was completely fascinated by it. The blend of computer vision and playful, context-free humor made it genuinely fun to watch. I thought it would be an excellent engineering challenge to recreate the exact same system from scratch—so I built **GOSSIP**.

<div align="center">

![GOSSIP Demo](./assets/demo.gif)

</div>

## Project Philosophy & Performance

A massive priority for this project was to prove that you **do not need expensive hardware** or top-tier GPUs to run playful, stable computer vision installations. 

This repository leverages entirely CPU-friendly techniques to maintain a visually pleasing frame rate:
*   **Nano-scale Object Inference:** GOSSIP runs Ultralytics YOLOv8 Nano (`yolov8n`), the smallest and fastest model of the family. Instead of doing inference on every single frame, we run it asynchronously every 3rd frame—slashing hardware requirement by ~66% while sacrificing almost no visual fidelity for pedestrians.
*   **Lightweight Identity Tracking:** Instead of using heavy spatial-temporal sequence models (like DeepSORT), it uses mathematically cheap Intersection-over-Union (IoU) overlap physics. If two bounding boxes overlap across consecutive frames, it assumes they are the same person.
*   **The "Director" Cooldown System:** Rather than bombarding the screen by detecting *everyone*, GOSSIP's algorithmic `Director` intentionally filters candidates to maintain order. It establishes maximum limits (e.g. max 3 people), ensures candidates have been consistently tracked for 10 frames before promotion, and enforces global quote cooldowns so jokes never repeat too frequently.
*   **Anti-Aliased Typography:** Unlike standard, jittery OpenCV annotations, GOSSIP maps bounding boxes through an Exponential Moving Average (EMA) to float the UI smoothly, rendering everything using clean `Pillow` alpha-composited fonts.

The result is a highly polished aesthetic that runs smoothly even on old laptops or embedded low-end CPUs without screaming fans.

## Reproducibility & Installation

I built the project rigorously modular so you can easily replicate the exact effect on your own webcams or security footage.

### Prerequisites

Ensure you have Python 3.8+ installed. You can install all minimum dependencies quickly (uses standard CPU-only dependencies without massive PyTorch stacks):

```bash
pip install -r requirements.txt
```

### Running the App

To run GOSSIP on your default live webcam:

```bash
python main.py
```

*Note: On first run, it will quickly download the ~6MB `yolov8n.pt` weights file.*

To test GOSSIP on a pre-recorded video:

```bash
python main.py --source "video/test2.mp4" --save "assets/output.mp4"
```

Enjoy observing the pedestrians!
