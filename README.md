# Dwellink iGo Zoom

Dwellink iGo Zoom is a Tkinter-based video recording application that allows you to preview and record video from your webcam with various filters and adjustments. The application supports zooming, filtering, adjusting contrast, brightness, sharpness, and saturation. It also includes features for freezing frames, taking pictures, and performing OCR (Optical Character Recognition) on the frozen frames.

## Features

- **Camera Device Selection**: Automatically detects and lists available camera devices.
- **Video Preview**: Real-time video preview with various filters and adjustments.
- **Zooming**: Zoom in and out using keyboard shortcuts.
- **Filters**: Apply various filters like greyscale, sepia, negative, and high contrast.
- **Adjustments**: Adjust contrast, brightness, sharpness, and saturation using sliders.
- **Frame Freezing**: Freeze and unfreeze the current frame.
- **Take Picture**: Capture and save the current frame as an image.
- **Record Video**: Record video and save it to a file.
- **OCR**: Perform OCR on the frozen frame to extract text.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Required Python Packages

Install the required packages using pip:

```sh
pip install opencv-python-headless pillow numpy pytesseract

```
### Clone the repository:
```sh
git clone https://github.com/ShairALiMughal/CameraApp.git
cd CameraApp
```

### Run the application
```sh
python app.py
```
### Controls
- Camera Selection: Use the dropdown to select a camera device.
- Zoom In: Ctrl + Up
- Zoom Out: Ctrl + Down
- Next Filter: Ctrl + Right
- Previous Filter: Ctrl + Left
- Freeze Frame: Ctrl + f
- Save Image: Ctrl + s
- Open Image: Ctrl + o
- Perform OCR: Ctrl + t
- Select Camera by Shortcut: Press 0-9 to select the corresponding camera device.

### Sliders
- Contrast: Adjust the contrast of the video.
- Brightness: Adjust the brightness of the video.
- Sharpness: Adjust the sharpness of the video.
- Saturation: Adjust the saturation of the video.

### code Structure
- app.py: Main application file.