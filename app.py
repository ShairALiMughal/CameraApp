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
import pyttsx3
def speak_async(text):
        def run_speak():
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=run_speak).start()
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
        self.engine = pyttsx3.init()
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
        
        self.root.bind('<<announce>>', self.handle_announce)

        self.cap = None
        self.zoom_level = 1.0
        self.filters = [
            ("No Filter", self.no_filter),
            ("Grayscale", self.greyscale),
            ("Sepia", self.sepia),
            ("Negative", self.negative),
            ("High Contrast", self.high_contrast),
            ("Yellow on Black", self.yellow_on_black),
            ("Yellow on Blue", self.yellow_on_blue),
            ("Black on Yellow", self.black_on_yellow),
            ("Green on Black", self.green_on_black),
            ("Blue on White", self.blue_on_white),
            ("Red on Black", self.red_on_black),
            ("Inverted", self.inverted),
            ("Inverted Grayscale", self.inverted_grayscale),
            ("Blue on Yellow", self.blue_on_yellow),
            ("White on Blue", self.white_on_blue),
            ("Black on Red", self.black_on_red)
        ]
        self.filter_index = 0
        self.frozen_frame = None
        self.frame = None

        self.setup_ui()

        self.is_recording = False
        self.is_previewing = False
        self.video_writer = None

        threading.Thread(target=self.populate_devices).start()
    
    
    def apply_colored_filter(self, frame, color1, color2, thresh_low, thresh_high):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, thresh_low, thresh_high)
        frame1 = cv2.bitwise_and(frame, frame, mask=mask)
        frame2 = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        color_frame1 = np.full(frame1.shape, color1, dtype=np.uint8)  # Color for the masked region
        color_frame2 = np.full(frame2.shape, color2, dtype=np.uint8)  # Color for the non-masked region
        blended_frame1 = cv2.addWeighted(frame1, 0.5, color_frame1, 0.5, 0)
        blended_frame2 = cv2.addWeighted(frame2, 0.5, color_frame2, 0.5, 0)
        return cv2.add(blended_frame1, blended_frame2)

    def handle_announce(self, event):
        description = event.widget.cget('data')
        self.root.after(10, lambda: self.root.bell())  # Optional: to draw attention
        self.root.after(20, lambda: self.root.event_generate('<<announce-text>>', data=description))
    
    def announce(self, widget, description):
        widget.bind("<FocusIn>", lambda e: self.root.after(10, lambda: widget.focus_set()))
        widget.bind("<FocusIn>", lambda e: self.root.after(20, lambda: widget.event_generate('<<announce>>', data=description)))



    def on_closing(self):
        self.is_previewing = False
        self.is_recording = False

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        if self.video_writer is not None:
            self.video_writer.release()

        self.root.quit()
        self.root.update_idletasks()

        # Terminate all threads
        for thread in threading.enumerate():
            if thread.is_alive():
                try:
                    thread.join(timeout=1)
                except RuntimeError:
                    continue

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

        self.contrast_slider = tk.Scale(self.right_frame, from_=0, to_=4, resolution=0.1, orient=tk.HORIZONTAL, label="Contrast", command=lambda value: self.update_contrast_brightness())
        self.contrast_slider.pack(pady=10)
        self.contrast_slider.set(1.0)
        self.contrast_slider.config(takefocus=True)
        ToolTip(self.contrast_slider, "Adjust the contrast of the video feed")
        self.announce(self.contrast_slider, "Adjust the contrast of the video feed")

        self.brightness_slider = tk.Scale(self.right_frame, from_=-100, to_=100, orient=tk.HORIZONTAL, label="Brightness", command=lambda value: self.update_contrast_brightness())
        self.brightness_slider.pack(pady=10)
        self.brightness_slider.set(0)
        self.brightness_slider.config(takefocus=True)
        ToolTip(self.brightness_slider, "Adjust the brightness of the video feed")
        self.announce(self.brightness_slider, "Adjust the brightness of the video feed")

        self.sharpness_slider = tk.Scale(self.right_frame, from_=0, to_=2, resolution=0.1, orient=tk.HORIZONTAL, label="Sharpness", command=lambda value: self.update_sharpness())
        self.sharpness_slider.pack(pady=10)
        self.sharpness_slider.set(1.0)
        self.sharpness_slider.config(takefocus=True)
        ToolTip(self.sharpness_slider, "Adjust the sharpness of the video feed")
        self.announce(self.sharpness_slider, "Adjust the sharpness of the video feed")

        self.saturation_slider = tk.Scale(self.right_frame, from_=0, to_=3, resolution=0.1, orient=tk.HORIZONTAL, label="Saturation", command=lambda value: self.update_saturation())
        self.saturation_slider.pack(pady=10)
        self.saturation_slider.set(1.0)
        self.saturation_slider.config(takefocus=True)
        ToolTip(self.saturation_slider, "Adjust the saturation of the video feed")
        self.announce(self.saturation_slider, "Adjust the saturation of the video feed")

        self.canvas = tk.Canvas(self.main_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)


    def populate_devices(self):
        devices = []
        max_devices = 10  # Set a reasonable limit to avoid long wait times

        for index in range(max_devices):
            cap = cv2.VideoCapture(index)
            ret, _ = cap.read()
            cap.release()

            if ret:
                devices.append(f"Camera {index}")
            else:
                break  # Stop searching when a device fails to initialize

        print("Devices found:", devices)  # Debug print

        if devices:
            self.root.after(0, self.update_device_combo, devices)
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", "No camera devices found"))

        # Remove the searching label and set focus to main window for shortcuts
        self.root.after(0, self.finalize_device_search)

    def update_device_combo(self, devices):
        self.device_combo['values'] = devices
        self.device_combo.current(0)
        self.start_preview()  # Start preview automatically if devices are found

    def finalize_device_search(self):
        self.searching_label.pack_forget()
        self.root.focus()

    def threaded_populate_devices(self):
        thread = threading.Thread(target=self.populate_devices)
        thread.daemon = True
        thread.start()
    
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
        filter_name, _ = self.filters[self.filter_index]
        speak_async(filter_name)  # This will not block the main thread
        self.apply_filter(self.frame)
        self.display_frame(self.frame)  # Ensure the frame updates immediately after applying the filter

    def prev_filter(self):
        self.filter_index = (self.filter_index - 1 + len(self.filters)) % len(self.filters)
        filter_name, _ = self.filters[self.filter_index]
        speak_async(filter_name)  # This will not block the main thread
        self.apply_filter(self.frame)
        self.display_frame(self.frame)  # Ensure the frame updates immediately after applying the filter


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

    def process_frame(self, frame):
        if self.frozen_frame is not None:
            frame = self.frozen_frame
        else:
            frame = self.apply_filter(frame)
            frame = self.apply_zoom(frame)
            frame = self.apply_contrast_brightness(frame)
            frame = self.apply_sharpness(frame)
            frame = self.apply_saturation(frame)
        self.display_frame(frame)


    def update_frame(self):
        while self.is_previewing:
            try:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.frame = frame  # Update the current frame
                        # Process and display frame using after method
                        self.root.after(1, self.process_frame, frame)
                    else:
                        self.is_previewing = False
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
            return frame  # No filter applied
        elif self.filter_index == 1:
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif self.filter_index == 2:
            frame = cv2.transform(frame, np.array([[0.272, 0.534, 0.131],
                                                [0.349, 0.686, 0.168],
                                                [0.393, 0.769, 0.189]]))
            return np.clip(frame, 0, 255).astype(np.uint8)
        elif self.filter_index == 3:
            return cv2.bitwise_not(frame)
        elif self.filter_index == 4:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        elif self.filter_index == 5:
            return self.yellow_on_black(frame)
        elif self.filter_index == 6:
            return self.yellow_on_blue(frame)
        elif self.filter_index == 7:
            return self.black_on_yellow(frame)
        elif self.filter_index == 8:
            return self.green_on_black(frame)
        elif self.filter_index == 9:
            return self.blue_on_white(frame)
        elif self.filter_index == 10:
            return self.red_on_black(frame)
        elif self.filter_index == 11:
            return self.inverted(frame)
        elif self.filter_index == 12:
            return self.inverted_grayscale(frame)
        elif self.filter_index == 13:
            return self.blue_on_yellow(frame)
        elif self.filter_index == 14:
            return self.white_on_blue(frame)
        elif self.filter_index == 15:
            return self.black_on_red(frame)
        else:
            return frame  # Default to no filter if index is out of range


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
    
    def inverted(self, frame):
        return cv2.bitwise_not(frame)
    
    def negative(self, frame):
        return cv2.bitwise_not(frame)
    
    def inverted_grayscale(self, frame):
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Invert the grayscale frame
            inverted_frame = cv2.bitwise_not(gray_frame)
            # Convert back to BGR for display consistency with other filters
            return cv2.cvtColor(inverted_frame, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Error applying inverted grayscale: {e}")
            return frame  # Return the original frame in case of an error
        


    
    def blue_on_yellow(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame, (20, 100, 100), (30, 255, 255))
        frame[mask != 0] = [120, 255, 255]  # Blue text
        frame[mask == 0] = [30, 255, 255]  # Yellow background
        return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    def white_on_blue(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame, (100, 150, 0), (140, 255, 255))
        frame[mask != 0] = [0, 0, 255]  # White text
        frame[mask == 0] = [120, 255, 255]  # Blue background
        return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    def black_on_red(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame, (0, 70, 50), (10, 255, 255))
        frame[mask != 0] = [0, 0, 0]  # Black text
        frame[mask == 0] = [0, 255, 255]  # Red background
        return cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    
    

    def high_contrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def yellow_on_black(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 255, 255))  # Light areas (white text)
        frame[mask != 0] = (0, 255, 255)  # Yellow
        frame[mask == 0] = (0, 0, 0)  # Black
        return frame

    def yellow_on_blue(self, frame):
    # Yellow text on blue background
        return self.apply_colored_filter(frame, (0, 255, 255), (255, 0, 0), (0, 100, 100), (50, 255, 255))
    

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
        # Blue text on white background
        return self.apply_colored_filter(frame, (255, 0, 0), (255, 255, 255), (110, 50, 50), (130, 255, 255))

    def red_on_black(self, frame):
        # Red text on black background
        return self.apply_colored_filter(frame, (0, 0, 255), (0, 0, 0), (0, 70, 50), (10, 255, 255))

    def normal(self, frame):
        return frame


    def apply_contrast_brightness(self, frame):
        contrast = float(self.contrast_slider.get())
        brightness = int(self.brightness_slider.get())
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        saturation = float(self.saturation_slider.get())
        if saturation == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = 0  # Set saturation to 0, resulting in grayscale
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        sharpness = float(self.sharpness_slider.get())
        if sharpness != 1.0:
            kernel = np.array([[0, -1, 0],
                            [-1, 5 + (sharpness - 1) * 4, -1],
                            [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        
        return frame


    def apply_sharpness(self, frame):
        sharpness = int(self.sharpness_slider.get())
        if sharpness != 1.0:
            kernel = np.array([[0, -1, 0],
                            [-1, 5 + (sharpness - 1) * 4, -1],
                            [0, -1, 0]])
            frame = cv2.filter2D(frame, -1, kernel)
        return frame
    def update_sharpness(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_sharpness(self.frame)
            self.display_frame(self.frame)


    def update_contrast_brightness(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_contrast_brightness(self.frame)
            self.display_frame(self.frame)


    def update_saturation(self, _=None):
        if self.frame is not None:
            self.frame = self.apply_saturation(self.frame)
            self.display_frame(self.frame)

    
    def apply_saturation(self, frame):
        saturation = int(self.saturation_slider.get())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame

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

    def show_tooltip(self, event):
        try:
            x, y, cx, cy = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
            self.tooltip = tk.Toplevel(self.widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1, font=("Arial", 10, "normal"))
            label.pack()
        except _tkinter.TclError:
            # Handle the error or retry
            self.root.after(100, lambda: self.show_tooltip(event))


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoRecorderApp(root)
    root.mainloop()