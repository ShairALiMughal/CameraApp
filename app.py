import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import pytesseract
import time
import shutil
import sys
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.tooltip = None

    def show_tooltip(self, event):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1, font=("Arial", 10, "normal"))
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

class VideoRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dwellink iGo Zoom")
        self.root.geometry("1200x700")
        self.root.state('zoomed')  # Start in maximized mode
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle the close button

        self.root.bind('<Control-Up>', lambda event: self.zoom_in())
        self.root.bind('<Control-Down>', lambda event: self.zoom_out())
        self.root.bind('<Control-Right>', lambda event: self.next_filter())
        self.root.bind('<Control-Left>', lambda event: self.prev_filter())
        self.root.bind('<Control-f>', lambda event: self.freeze_frame())
        self.root.bind('<Control-s>', lambda event: self.save_image())
        self.root.bind('<Control-o>', lambda event: self.open_image())
        self.root.bind('<Control-t>', lambda event: self.ocr())
        for i in range(10):
            self.root.bind(str(i), self.select_camera_by_shortcut)

        self.cap = None
        self.zoom_level = 1.0
        self.filters = [
            self.no_filter,
            self.greyscale,
            self.sepia,
            self.negative,
            self.high_contrast
        ]
        self.filter_index = 0
        self.frozen_frame = None
        self.frame = None

        self.setup_ui()

        self.is_recording = False
        self.is_previewing = False
        self.video_writer = None

        threading.Thread(target=self.populate_devices).start()
    
    def announce(self, widget, description):
        widget.focus()
        widget.bind("<FocusIn>", lambda e: self.root.after(10, lambda: self.root.bell()))
        widget.bind("<FocusIn>", lambda e: self.root.after(20, lambda: self.root.event_generate('<<announce>>', data=description)))

    def on_closing(self):
        self.is_previewing = False
        self.is_recording = False

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.root.destroy()
        sys.exit()


    def select_camera_by_shortcut(self, event):
        index = int(event.char)
        if index < len(self.device_combo['values']):
            self.device_combo.current(index)
            self.start_preview()

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
        self.start_button.config(takefocus=True)
        self.start_button.config(text="Start Recording", underline=0)
        ToolTip(self.start_button, "Start recording the video")
        self.announce(self.start_button, "Start recording the video")

        self.stop_button = tk.Button(self.left_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        self.stop_button.config(takefocus=True)
        self.stop_button.config(text="Stop Recording", underline=0)
        ToolTip(self.stop_button, "Stop recording the video")
        self.announce(self.stop_button, "Stop recording the video")


        self.take_picture_button = tk.Button(self.left_frame, text="Take Picture", command=self.take_picture)
        self.take_picture_button.pack(pady=10)
        self.take_picture_button.config(takefocus=True)
        self.take_picture_button.config(text="Take Picture", underline=0)
        ToolTip(self.take_picture_button, "Take a picture from the video feed")
        self.announce(self.take_picture_button, "Take a picture from the video feed")


        self.zoom_in_button = tk.Button(self.left_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(pady=10)
        self.zoom_in_button.config(takefocus=True)
        self.zoom_in_button.config(text="Zoom In", underline=0)
        ToolTip(self.zoom_in_button, "Zoom in on the video feed")
        self.announce(self.zoom_in_button, "Zoom in on the video feed")

        self.zoom_out_button = tk.Button(self.left_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(pady=10)
        self.zoom_out_button.config(takefocus=True)
        self.zoom_out_button.config(text="Zoom Out", underline=0)
        ToolTip(self.zoom_out_button, "Zoom out of the video feed")
        self.announce(self.zoom_out_button, "Zoom out of the video feed")

        self.next_filter_button = tk.Button(self.left_frame, text="Next Filter", command=self.next_filter)
        self.next_filter_button.pack(pady=10)
        self.next_filter_button.config(takefocus=True)
        self.next_filter_button.config(text="Next Filter", underline=0)
        ToolTip(self.next_filter_button, "Apply the next filter to the video feed")
        self.announce(self.next_filter_button, "Apply the next filter to the video feed")


        self.prev_filter_button = tk.Button(self.left_frame, text="Previous Filter", command=self.prev_filter)
        self.prev_filter_button.pack(pady=10)
        self.prev_filter_button.config(takefocus=True)
        self.prev_filter_button.config(text="Previous Filter", underline=0)
        ToolTip(self.prev_filter_button, "Apply the previous filter to the video feed")
        self.announce(self.prev_filter_button, "Apply the previous filter to the video feed")


        self.freeze_frame_button = tk.Button(self.left_frame, text="Freeze Frame", command=self.freeze_frame)
        self.freeze_frame_button.pack(pady=10)
        self.freeze_frame_button.config(takefocus=True)
        self.freeze_frame_button.config(text="Freeze Frame", underline=0)
        ToolTip(self.freeze_frame_button, "Freeze the current frame of the video feed")
        self.announce(self.freeze_frame_button, "Freeze the current frame of the video feed")


        self.save_image_button = tk.Button(self.left_frame, text="Save Image", command=self.save_image)
        self.save_image_button.pack(pady=10)
        self.save_image_button.config(takefocus=True)
        self.save_image_button.config(text="Save Image", underline=0)
        ToolTip(self.save_image_button, "Save the current frame as an image")
        self.announce(self.save_image_button, "Save the current frame as an image")

        self.open_image_button = tk.Button(self.left_frame, text="Open Image", command=self.open_image)
        self.open_image_button.pack(pady=10)
        self.open_image_button.config(takefocus=True)
        self.open_image_button.config(text="Open Image", underline=0)
        ToolTip(self.open_image_button, "Open an image file")
        self.announce(self.open_image_button, "Open an image file")

        self.ocr_button = tk.Button(self.left_frame, text="OCR", command=self.ocr)
        self.ocr_button.pack(pady=10)
        self.ocr_button.config(takefocus=True)
        self.ocr_button.config(text="OCR", underline=0)
        ToolTip(self.ocr_button, "Perform OCR on the frozen frame")
        self.announce(self.ocr_button, "Perform OCR on the frozen frame")

        self.contrast_slider = tk.Scale(self.right_frame, from_=0, to_=4, resolution=0.1, orient=tk.HORIZONTAL, label="Contrast", command=self.update_contrast_brightness)
        self.contrast_slider.pack(pady=10)
        self.contrast_slider.set(1.0)
        self.contrast_slider.config(takefocus=True)
        ToolTip(self.contrast_slider, "Adjust the contrast of the video feed")
        self.announce(self.contrast_slider, "Adjust the contrast of the video feed")

        self.brightness_slider = tk.Scale(self.right_frame, from_=-100, to_=100, orient=tk.HORIZONTAL, label="Brightness", command=self.update_contrast_brightness)
        self.brightness_slider.pack(pady=10)
        self.brightness_slider.set(0)
        self.brightness_slider.config(takefocus=True)
        ToolTip(self.brightness_slider, "Adjust the brightness of the video feed")
        self.announce(self.brightness_slider, "Adjust the brightness of the video feed")

        self.sharpness_slider = tk.Scale(self.right_frame, from_=0, to_=2, resolution=0.1, orient=tk.HORIZONTAL, label="Sharpness", command=self.update_sharpness)
        self.sharpness_slider.pack(pady=10)
        self.sharpness_slider.set(1.0)
        self.sharpness_slider.config(takefocus=True)
        ToolTip(self.sharpness_slider, "Adjust the sharpness of the video feed")
        self.announce(self.sharpness_slider, "Adjust the sharpness of the video feed")

        self.saturation_slider = tk.Scale(self.right_frame, from_=0, to_=3, resolution=0.1, orient=tk.HORIZONTAL, label="Saturation", command=self.update_saturation)
        self.saturation_slider.pack(pady=10)
        self.saturation_slider.set(1.0)
        self.saturation_slider.config(takefocus=True)
        ToolTip(self.saturation_slider, "Adjust the saturation of the video feed")
        self.announce(self.saturation_slider, "Adjust the saturation of the video feed")

        self.canvas = tk.Canvas(self.main_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)


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
                try:
                    shutil.move('output.avi', file_path)
                    messagebox.showinfo("Save Video", f"Video saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Save Video", f"Failed to save video: {e}")

    def zoom_in(self):
        self.zoom_level = min(4.0, self.zoom_level + 0.1)

    def zoom_out(self):
        self.zoom_level = max(1.0, self.zoom_level - 0.1)

    def next_filter(self):
        self.filter_index = (self.filter_index + 1) % len(self.filters)

    def prev_filter(self):
        self.filter_index = (self.filter_index - 1) % len(self.filters)

    def freeze_frame(self):
        if self.frozen_frame is None and self.frame is not None:
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

    def update_frame(self):
        while self.is_previewing:
            try:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        if self.frozen_frame is not None:
                            frame = self.frozen_frame
                        else:
                            frame = self.apply_filter(frame)
                            frame = self.apply_zoom(frame)
                            frame = self.apply_contrast_brightness(frame)
                            frame = self.apply_sharpness(frame)
                            frame = self.apply_saturation(frame)
                            self.frame = frame
                        self.root.after(1, self.display_frame, frame)
                time.sleep(0.01)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                self.cap.release()
                break

    def freeze_frame(self):
        if self.frozen_frame is None and self.frame is not None:
            self.frozen_frame = self.frame.copy()
        else:
            self.frozen_frame = None


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
        if self.filter_index == 0:
            return frame
        elif self.filter_index == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif self.filter_index == 2:
            frame = cv2.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                                    [0.349, 0.686, 0.168],
                                                    [0.393, 0.769, 0.189]]))
            frame[np.where(frame > 255)] = 255
        elif self.filter_index == 3:
            frame = cv2.bitwise_not(frame)
        elif self.filter_index == 4:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return frame

    def no_filter(self, frame):
        return frame

    def greyscale(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def sepia(self, frame):
        frame = cv2.transform(frame, np.matrix([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
        frame[np.where(frame > 255)] = 255
        return frame

    def negative(self, frame):
        return cv2.bitwise_not(frame)

    def high_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_contrast_brightness(self, frame):
        contrast = int(self.contrast_slider.get())
        #print(contrast)
        brightness = int(self.brightness_slider.get())
        #print(brightness)
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        return frame

    def apply_sharpness(self, frame):
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

    
    def apply_saturation(self, frame):
        saturation = self.saturation_slider.get()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRecorderApp(root)
    root.mainloop()
