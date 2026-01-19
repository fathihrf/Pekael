import sys
import os
import cv2
import csv
from datetime import datetime
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGroupBox, QFormLayout, QScrollArea, QMessageBox, QInputDialog)
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import main as backend
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Start defining the main class based on the UI file
class HygroScanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the UI file
        ui_file_path = os.path.join(os.path.dirname(__file__), 'ui', 'mainwindow.ui')
        if not os.path.exists(ui_file_path):
             QMessageBox.critical(self, "Error", f"UI file not found: {ui_file_path}")
             sys.exit(1)
             
        uic.loadUi(ui_file_path, self)
        
        self.templates = {}
        self.ocr_reader = None
        self.current_image_path = None
        self.current_cv_image = None
        
        self.init_ui_connections()
        self.init_backend()
        
    def init_ui_connections(self):
        # Connect UI elements from .ui file to functions
        self.btn_select_photo.clicked.connect(self.upload_image)
        self.btn_confirm.clicked.connect(self.save_result)
        
        # Navigation buttons
        self.btn_nav_dashboard.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.btn_nav_upload.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))

    def init_backend(self):
        # This might be slow, so we could thread it, but for now keep it simple
        try:
            if not os.path.exists(backend.ANNOTATIONS_FILE):
                self.lbl_status.setText("Error: Annotations file missing!")
                self.btn_select_photo.setEnabled(False)
                return

            self.lbl_status.setText("Loading annotations...")
            QApplication.processEvents()
            annotations = backend.load_annotations()
            
            self.lbl_status.setText("Extracting templates...")
            QApplication.processEvents()
            self.templates = backend.extract_templates(annotations)
            
            if not self.templates:
                self.lbl_status.setText("Error: Could not extract templates from training images.")
                self.btn_select_photo.setEnabled(False)
                return
                
            self.lbl_status.setText("Initializing OCR Engine (this may take a moment)...")
            QApplication.processEvents()
            from paddleocr import PaddleOCR
            self.ocr_reader = PaddleOCR(lang='en', use_textline_orientation=False)
            
            self.lbl_status.setText("Ready.")
            
        except Exception as e:
            self.lbl_status.setText(f"Initialization Error: {str(e)}")
            print(e)

    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        
        if file_name:
            self.current_image_path = file_name
            self.load_image_preview(file_name)
            self.process_image(file_name)
            
    def load_image_preview(self, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            # Scale if too large
            if pixmap.width() > 800:
                pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
            self.lbl_image_preview.setPixmap(pixmap)
            
    def process_image(self, path):
        if not self.ocr_reader or not self.templates:
            QMessageBox.warning(self, "Error", "Backend not ready.")
            return
            
        self.lbl_status.setText("Processing...")
        QApplication.processEvents()
        
        try:
            # Read image with CV2
            img = cv2.imread(path)
            if img is None:
                self.lbl_status.setText("Error reading image.")
                return
            
            # Analyze
            # We don't pass ground_truth here as we interpret new images
            processed_img, results = backend.analyze_image(img, self.templates, self.ocr_reader)
            self.current_cv_image = processed_img
            
            # Update Results
            temp_val = results.get("Temperature", {}).get("value", "")
            humid_val = results.get("Humidity", {}).get("value", "")
            
            self.input_temp.setText(temp_val)
            self.input_humidity.setText(humid_val)
            
            # Show processed image (convert BGR to RGB for Qt)
            rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            if pixmap.width() > 800:
                pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
            self.lbl_image_preview.setPixmap(pixmap)
            
            self.lbl_status.setText("Analysis Complete.")
            
        except Exception as e:
            self.lbl_status.setText(f"Processing Error: {str(e)}")
            print(e)
            
    def save_result(self):
        if not self.current_image_path:
             return
             
        temp = self.input_temp.text()
        hum = self.input_humidity.text()
        
        # Save to CSV
        csv_file = "results.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header if new file
                if not file_exists:
                    writer.writerow(["Timestamp", "Image Name", "Temperature", "Humidity"])
                    
                # Write data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                image_name = os.path.basename(self.current_image_path)
                writer.writerow([timestamp, image_name, temp, hum])
                
            QMessageBox.information(self, "Saved", f"Data saved to {csv_file}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save to CSV: {str(e)}")
            
        # Google Sheets Upload
        if os.path.exists('credentials.json'):
             # Prompt for sheet name
             sheet_name, ok = QInputDialog.getText(self, "Google Cloud", "Enter your Google Sheet Name to upload (Optional):")
             if ok and sheet_name:
                 try:
                     self.lbl_status.setText("Uploading to Drive...")
                     QApplication.processEvents()
                     
                     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                     creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
                     client = gspread.authorize(creds)
                     
                     # Open Sheet
                     try:
                         sheet = client.open(sheet_name).sheet1
                         sheet.append_row([timestamp, image_name, temp, hum])
                         QMessageBox.information(self, "Success", "Data uploaded to Google Sheet!")
                     except gspread.exceptions.SpreadsheetNotFound:
                          QMessageBox.warning(self, "Error", f"Spreadsheet '{sheet_name}' not found.\nPlease ensure you 'Shared' it with:\n{creds.service_account_email}")
                          
                 except Exception as e:
                     QMessageBox.warning(self, "Google Sheets Error", str(e))
                 finally:
                     self.lbl_status.setText("Ready.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HygroScanApp()
    window.show()
    sys.exit(app.exec_())
