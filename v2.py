import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import pytesseract
import time
import shutil

class VideoRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dwellink iGo Zoom")
        self.root.geometry("1200x700")
        self.root.state('zoomed')  # Start in maximized mode
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle the close button

        self.cap = None
        self.is_recording = False
        self.is_previewing = False

        self.setup_ui()
        threading.Thread(target=self.populate_devices).start()

    def setup_ui(self):
        self.left_frame = tk.Frame(self.root, width=200, bg='lightgrey')
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = tk.Frame(self.root, width=200, bg='lightgrey')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.device_label = tk.Label(self.left_frame, text="Select Camera Device:")
        self.device_label.pack(pady=10)

        self.device_combo = ttk.Combobox(self.left_frame, state="readonly")
        self.device_combo.pack(pady=10)
        self.device_combo.bind("<<ComboboxSelected>>", lambda event: self.start_preview())
        self.announce(self.device_combo, "Select the camera device from this dropdown")

        self.searching_label = tk.Label(self.left_frame, text="Searching for camera devices...", fg="blue")
        self.searching_label.pack(pady=10)

        self.start_button = tk.Button(self.left_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)
        self.announce(self.start_button, "Start recording the video")

        self.stop_button = tk.Button(self.left_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        self.announce(self.stop_button, "Stop recording the video")

        self.take_picture_button = tk.Button(self.left_frame, text="Take Picture", command=self.take_picture)
        self.take_picture_button.pack(pady=10)
        self.announce(self.take_picture_button, "Take a picture from the video feed")

        self.zoom_in_button = tk.Button(self.left_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(pady=10)
        self.announce(self.zoom_in_button, "Zoom in on the video feed")

        self.zoom_out_button = tk.Button(self.left_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(pady=10)
        self.announce(self.zoom_out_button, "Zoom out of the video feed")

        self.next_filter_button = tk.Button(self.left_frame, text="Next Filter", command=self.next_filter)
        self.next_filter_button.pack(pady=10)
        self.announce(self.next_filter_button, "Apply the next filter to the video feed")

        self.prev_filter_button = tk.Button(self.left_frame, text="Previous Filter", command=self.prev_filter)
        self.prev_filter_button.pack(pady=10)
        self.announce(self.prev_filter_button, "Apply the previous filter to the video feed")

        self.freeze_frame_button = tk.Button(self.left_frame, text="Freeze Frame", command=self.freeze_frame)
        self.freeze_frame_button.pack(pady=10)
        self.announce(self.freeze_frame_button, "Freeze the current frame of the video feed")

        self.save_image_button = tk.Button(self.left_frame, text="Save Image", command=self.save_image)
        self.save_image_button.pack(pady=10)
        self.announce(self.save_image_button, "Save the current frame as an image")

        self.open_image_button = tk.Button(self.left_frame, text="Open Image", command=self.open_image)
        self.open_image_button.pack(pady=10)
        self.announce(self.open_image_button, "Open an image file")

        self.ocr_button = tk.Button(self.left_frame, text="OCR", command=self.ocr)
        self.ocr_button.pack(pady=10)
        self.announce(self.ocr_button, "Perform OCR on the frozen frame")

        self.contrast_slider = tk.Scale(self.right_frame, from_=0, to_=4, resolution=0.1, orient=tk.HORIZONTAL, label="Contrast", command=self.update_contrast_brightness)
        self.contrast_slider.pack(pady=10)
        self.contrast_slider.set(1.0)
        self.announce(self.contrast_slider, "Adjust the contrast of the video feed")

        self.brightness_slider = tk.Scale(self.right_frame, from_=-100, to_=100, orient=tk.HORIZONTAL, label="Brightness", command=self.update_contrast_brightness)
        self.brightness_slider.pack(pady=10)
        self.brightness_slider.set(0)
        self.announce(self.brightness_slider, "Adjust the brightness of the video feed")

        self.sharpness_slider = tk.Scale(self.right_frame, from_=0, to_=2, resolution=0.1, orient=tk.HORIZONTAL, label="Sharpness", command=self.update_sharpness)
        self.sharpness_slider.pack(pady=10)
        self.sharpness_slider.set(1.0)
        self.announce(self.sharpness_slider, "Adjust the sharpness of the video feed")

        self.saturation_slider = tk.Scale(self.right_frame, from_=0, to_=3, resolution=0.1, orient=tk.HORIZONTAL, label="Saturation", command=self.update_saturation)
        self.saturation_slider.pack(pady=10)
        self.saturation_slider.set(1.0)
        self.announce(self.saturation_slider, "Adjust the saturation of the video feed")

        self.canvas = tk.Canvas(self.main_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def announce(self, widget, description):
        widget.focus()
        widget.bind("<FocusIn>", lambda e: self.root.after(10, lambda: self.root.bell()))
        widget.bind("<FocusIn>", lambda e: self.root.after(20, lambda: self.root.event_generate('<<announce>>', data=description)))

    def on_closing(self):
        self.is_previewing = False
        self.is_recording = False

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.root.quit()
        self.root.update_idletasks()
        sys.exit()

    def populate_devices(self):
        devices = []
        index = 0
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                cap.release()
                break
            devices.append(f"Camera {index}")
            cap.release()
            index += 1

        print("Devices found:", devices)  # Debug print

        if devices:
            self.device_combo['values'] = devices
            self.device_combo.current(0)
        else:
            messagebox.showerror("Error", "No camera devices found")
        
        # Remove the searching label
        self.searching_label.pack_forget()

        # Automatically select the first device if available
        if devices:
            self.device_combo.current(0)
            self.start_preview()

        # Set focus to the main window or another widget to ensure shortcuts work
        self.root.focus()

    def start_preview(self):
        selected_device = self.device_combo.get()
        print("Selected device:", selected_device)  # Debug print
        if selected_device:
            try:
                device_index = int(selected_device.split(' ')[1])
                self.cap = cv2.VideoCapture(device_index)

                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Failed to open camera")
                    return

                self.is_previewing = True
                threading.Thread(target=self.update_frame).start()
            except IndexError:
                messagebox.showerror("Error", "No valid camera device selected")
        else:
            messagebox.showerror("Error", "No camera device selected")

    def start_recording(self):
        if not self.is_previewing:
            threading.Thread(target=self.start_preview).start()

        self.is_recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Get the default resolutions and frame rate
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))  # Get the frame rate of the capture device

        # Define the codec and create VideoWriter object
        self.video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))

    def stop_recording(self):
        self.is_recording = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.video_writer:
            # Release the video writer
            self.video_writer.release()
            
            # Ask the user to specify a filename for the video
            file_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
            if not file_path:
                messagebox.showwarning("Save Video", "No filename specified. Video not saved.")
            else:
                # Rename the temporary file to the user-specified filename
                import os
                shutil.move('output.avi', file_path)
                messagebox.showinfo("Save Video", f"Video saved to {file_path}")

    def zoom_in(self):
        self.zoom_level = min(4.0, self.zoom_level + 0.1)

    def zoom_out(self):
        self.zoom_level = max(1.0, self.zoom_level - 0.1)

    def next_filter(self):
        self.filter_index = (self.filter_index + 1) % len(self.filters)

    def prev_filter(self):
        self.filter_index = (self.filter_index - 1) % len(self.filters)

    def freeze_frame(self):
        if self.frame is not None:
            if self.frozen_frame is None:
                self.frozen_frame = self.frame.copy()
            else:
                self.frozen_frame = None

    def save_image(self):
        if self.frozen_frame is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.frozen_frame)
                messagebox.showinfo("Save Image", f"Image saved to {file_path}")

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.frozen_frame = image
                self.display_frame(image)

    def ocr(self):
        if self.frozen_frame is not None:
            try:
                text = pytesseract.image_to_string(self.frozen_frame)
                messagebox.showinfo("OCR Result", text)
            except pytesseract.TesseractNotFoundError:
                messagebox.showerror("OCR Error", "Tesseract is not installed or not in your PATH.")
            except Exception as e:
                messagebox.showerror("OCR Error", f"An error occurred during OCR: {str(e)}")

    def take_picture(self):
        if self.frame is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.frame)
                messagebox.showinfo("Take Picture", f"Picture saved to {file_path}")

    def process_frame(self, frame):
        if self.frozen_frame is not None:
            frame = self.frozen_frame
        else:
            frame = self.apply_filter(frame)
            frame = self.apply_zoom(frame)
            frame = self.apply_contrast_brightness(frame)
            frame = self.apply_sharpness(frame)  # Apply sharpness
            frame = self.apply_saturation(frame)
            self.frame = frame
        self.display_frame(frame)

    def update_frame(self):
        while self.is_previewing:
            try:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.root.after(1, self.process_frame, frame)
                time.sleep(0.01)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                self.cap.release()
                break

    def apply_zoom(self, frame):
        if len(frame.shape) == 2:  # Grayscale image
            height, width = frame.shape
            center_x, center_y = width // 2, height // 2
            new_width, new_height = int(width / self.zoom_level), int(height / self.zoom_level)
            x1 = max(0, center_x - new_width // 2)
            x2 = min(width, center_x + new_width // 2)
            y1 = max(0, center_y - new_height // 2)
            y2 = min(height, center_y + new_height // 2)
            frame = frame[y1:y2, x1:x2]
            return cv2.resize(frame, (width, height))
        else:  # Color image
            height, width, _ = frame.shape
            center_x, center_y = width // 2, height // 2
            new_width, new_height = int(width / self.zoom_level), int(height / self.zoom_level)
            x1 = max(0, center_x - new_width // 2)
            x2 = min(width, center_x + new_width // 2)
            y1 = max(0, center_y - new_height // 2)
            y2 = min(height, center_y + new_height // 2)
            frame = frame[y1:y2, x1:x2]
            return cv2.resize(frame, (width, height))

    def apply_filter(self, frame):
        return self.filters[self.filter_index](frame)

    def no_filter(self, frame):
        return frame

    def inverted_colors(self, frame):
        return cv2.bitwise_not(frame)

    def yellow_on_black(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 255, 255)  # Yellow
        frame[mask == 0] = (0, 0, 0)  # Black
        return frame

    def yellow_on_blue(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 255, 255)  # Yellow
        frame[mask == 0] = (255, 0, 0)  # Blue
        return frame

    def black_on_yellow(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 255, 255)  # Yellow
        frame[mask == 0] = (0, 0, 0)  # Black
        frame = cv2.bitwise_not(frame)
        return frame

    def green_on_black(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 255, 0)  # Green
        frame[mask == 0] = (0, 0, 0)  # Black
        return frame

    def blue_on_white(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (255, 0, 0)  # Blue
        frame[mask == 0] = (255, 255, 255)  # White
        return frame

    def red_on_black(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 0, 255)  # Red
        frame[mask == 0] = (0, 0, 0)  # Black
        return frame

    def normal(self, frame):
        return frame

    def greyscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sepia(self, frame):
        frame = cv2.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
        frame[np.where(frame > 255)] = 255
        return frame

    def high_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_contrast_brightness(self, frame):
        contrast = self.contrast_slider.get()
        brightness = self.brightness_slider.get()
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        saturation = self.saturation_slider.get()
        if saturation == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = 0  # Set saturation to 0, resulting in grayscale
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        sharpness = self.sharpness_slider.get()
        if sharpness != 1.0:
            kernel = np.array([[0, -1, 0],
                            [-1, 5 + (sharpness - 1) * 4, -1],
                            [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        
        return frame

    def update_contrast_brightness(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_contrast_brightness(self.frame)
            self.display_frame(self.frame)

    def update_sharpness(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_sharpness(self.frame)
            self.display_frame(self.frame)

    def update_saturation(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_saturation(self.frame)
            self.display_frame(self.frame)

    def display_frame(self, frame):
        # Get the dimensions of the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Resize the frame to fit the canvas while maintaining the aspect ratio
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if canvas_width / aspect_ratio <= canvas_height:
            new_width = canvas_width
            new_height = int(canvas_width / aspect_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * aspect_ratio)
        
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Convert the frame to RGB and display it
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")  # Clear the canvas before drawing the new image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRecorderApp(root)
    root.mainloop()
