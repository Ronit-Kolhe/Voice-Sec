import sys
import os
import shutil
import zipfile
import librosa
import numpy as np
from resemblyzer import VoiceEncoder
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import pickle
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QMessageBox, QProgressBar, QHBoxLayout, QGridLayout,
    QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt

# Secure encryption key (change this!)
SECRET_KEY = b"your-strong-password"
VAULT_FOLDER = "vault_storage"
VOICE_MODEL_PATH = "voice_model.pkl"
DEFAULT_REQUIRED_SAMPLES = 20  # Default value, now configurable

# Ensure vault folder exists
if not os.path.exists(VAULT_FOLDER):
    os.makedirs(VAULT_FOLDER)

# Initialize voice encoder
encoder = VoiceEncoder()

class VoiceModel:
    def __init__(self):
        self.embeddings = []
        self.trained = False
        self.sample_count = 0
        self.required_samples = DEFAULT_REQUIRED_SAMPLES
        
    def add_sample(self, audio_data):
        """Add a voice sample embedding to the model"""
        embedding = encoder.embed_utterance(audio_data)
        self.embeddings.append(embedding)
        self.sample_count += 1
        
    def train(self):
        """Finalize training after all samples are added"""
        self.embeddings = np.array(self.embeddings)
        self.trained = True
        
    def authenticate(self, test_embedding):
        """Compare test embedding against the trained model"""
        if not self.trained:
            return False
            
        similarities = np.dot(self.embeddings, test_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(test_embedding)
        )
        
        # Calculate average similarity score
        avg_similarity = np.mean(similarities)
        # Count how many samples have similarity > threshold
        high_similarity_count = np.sum(similarities > 0.75)
        
        # Authentication passes if average similarity is high enough
        # AND enough individual samples match well
        return avg_similarity > 0.7 and high_similarity_count >= len(self.embeddings) * 0.6
        
    def save(self, path):
        """Save the trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        """Load a trained model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class VoiceVaultApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice-Secured Vault")
        self.setGeometry(100, 100, 600, 400)
        self.voice_model = VoiceModel()
        self.init_ui()
        
        # Check if we have a trained model
        if os.path.exists(VOICE_MODEL_PATH):
            try:
                self.voice_model = VoiceModel.load(VOICE_MODEL_PATH)
                if self.voice_model.trained:
                    self.show_authentication_ui()
                else:
                    self.show_sample_config_ui()
            except:
                self.show_sample_config_ui()
        else:
            self.show_sample_config_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create header
        self.header_label = QLabel("Voice-Secured Vault")
        self.header_label.setAlignment(Qt.AlignCenter)
        font = self.header_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.header_label.setFont(font)
        self.main_layout.addWidget(self.header_label)
        
        # Create status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)
        
        # Create content area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.main_layout.addWidget(self.content_widget)

    def show_sample_config_ui(self):
        """Show UI for configuring the number of required voice samples"""
        self.clear_content()
        
        # Update status
        self.status_label.setText("Configure Voice Authentication Training")
        
        # Configuration form
        form_layout = QFormLayout()
        
        # Instructions
        instructions = QLabel("Before training, select how many voice samples to collect.\n"
                             "More samples typically provide better security but require more time to collect.")
        instructions.setWordWrap(True)
        self.content_layout.addWidget(instructions)
        
        # Sample count selector
        self.sample_count_spinner = QSpinBox()
        self.sample_count_spinner.setMinimum(5)    # Minimum number of samples
        self.sample_count_spinner.setMaximum(100)  # Maximum number of samples
        self.sample_count_spinner.setValue(DEFAULT_REQUIRED_SAMPLES)
        self.sample_count_spinner.setSingleStep(5)
        
        form_layout.addRow("Number of voice samples to collect:", self.sample_count_spinner)
        
        # Add recommendations
        recommendations = QLabel("Recommendations:\n"
                                "• Minimum security: 5-10 samples\n"
                                "• Medium security: 20-30 samples\n"
                                "• High security: 50+ samples")
        form_layout.addRow(recommendations)
        
        self.content_layout.addLayout(form_layout)
        
        # Start button
        start_btn = QPushButton("Start Training")
        start_btn.clicked.connect(self.start_training)
        self.content_layout.addWidget(start_btn)

    def show_training_ui(self):
        """Show the UI for collecting voice samples and training the model"""
        self.clear_content()
        
        # Update status
        required_samples = self.voice_model.required_samples
        self.status_label.setText(f"Please provide {required_samples} voice samples to train the system")
        
        # Add training instructions
        instructions = QLabel("Record or upload multiple voice samples of the authorized user.\n"
                             "Each sample should be a WAV file of 3-5 seconds of speech.")
        instructions.setWordWrap(True)
        self.content_layout.addWidget(instructions)
        
        # Progress tracking
        progress_layout = QHBoxLayout()
        progress_label = QLabel(f"Samples collected: {self.voice_model.sample_count}/{required_samples}")
        self.progress_label = progress_label
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(required_samples)
        self.progress_bar.setValue(self.voice_model.sample_count)
        progress_layout.addWidget(self.progress_bar)
        
        self.content_layout.addLayout(progress_layout)
        
        # Buttons
        btn_layout = QGridLayout()
        
        self.add_sample_btn = QPushButton("Add Voice Samples")
        self.add_sample_btn.clicked.connect(self.add_voice_samples)
        btn_layout.addWidget(self.add_sample_btn, 0, 0)
        
        self.finish_training_btn = QPushButton("Complete Training")
        self.finish_training_btn.clicked.connect(self.complete_training)
        self.finish_training_btn.setEnabled(self.voice_model.sample_count >= required_samples)
        btn_layout.addWidget(self.finish_training_btn, 0, 1)
        
        self.reset_training_btn = QPushButton("Reset Training")
        self.reset_training_btn.clicked.connect(self.reset_training)
        btn_layout.addWidget(self.reset_training_btn, 1, 0, 1, 2)
        
        # Add batch upload option
        self.batch_upload_btn = QPushButton("Batch Upload Samples")
        self.batch_upload_btn.clicked.connect(self.batch_upload_samples)
        btn_layout.addWidget(self.batch_upload_btn, 2, 0, 1, 2)
        
        self.content_layout.addLayout(btn_layout)
        
    def show_authentication_ui(self):
        """Show the UI for authenticating and accessing the vault"""
        self.clear_content()
        
        # Update status
        self.status_label.setText("Vault is locked. Please authenticate with your voice.")
        
        # Authentication controls
        auth_label = QLabel("Upload a voice sample to verify your identity and unlock the vault:")
        auth_label.setWordWrap(True)
        self.content_layout.addWidget(auth_label)
        
        # Show model info
        if hasattr(self.voice_model, 'required_samples'):
            model_info = QLabel(f"Voice model trained with {self.voice_model.sample_count} samples")
            self.content_layout.addWidget(model_info)
        
        btn_layout = QVBoxLayout()
        
        self.upload_auth_button = QPushButton("Upload Voice Sample")
        self.upload_auth_button.clicked.connect(self.upload_voice_sample)
        btn_layout.addWidget(self.upload_auth_button)
        
        self.auth_status = QLabel("")
        btn_layout.addWidget(self.auth_status)
        
        self.unlock_button = QPushButton("Authenticate and Unlock Vault")
        self.unlock_button.clicked.connect(self.authenticate_vault)
        btn_layout.addWidget(self.unlock_button)
        
        self.content_layout.addLayout(btn_layout)

    def show_vault_ui(self):
        """Show the UI for using the unlocked vault"""
        self.clear_content()
        
        # Update status
        self.status_label.setText("✅ Authentication Successful! Vault is unlocked.")
        
        # Vault controls
        vault_label = QLabel("Your voice has been verified. You now have access to the secure vault.")
        vault_label.setWordWrap(True)
        self.content_layout.addWidget(vault_label)
        
        button_layout = QVBoxLayout()
        
        self.encrypt_button = QPushButton("Encrypt Folder")
        self.encrypt_button.clicked.connect(self.encrypt_folder)
        button_layout.addWidget(self.encrypt_button)
        
        self.decrypt_button = QPushButton("Decrypt Folder")
        self.decrypt_button.clicked.connect(self.decrypt_folder)
        button_layout.addWidget(self.decrypt_button)
        
        self.lock_button = QPushButton("Lock Vault")
        self.lock_button.clicked.connect(self.lock_vault)
        button_layout.addWidget(self.lock_button)
        
        # Add retrain button to vault UI
        self.retrain_btn = QPushButton("Reset and Retrain Voice Model")
        self.retrain_btn.clicked.connect(self.reset_training)
        button_layout.addWidget(self.retrain_btn)
        
        self.content_layout.addLayout(button_layout)

    def clear_content(self):
        """Clear the content area to prepare for new UI"""
        # Remove all widgets from content layout
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                # If item is a layout
                while item.count():
                    subitem = item.takeAt(0)
                    subwidget = subitem.widget()
                    if subwidget:
                        subwidget.deleteLater()

    def start_training(self):
        """Start the training process with configured number of samples"""
        # Get the user-selected number of samples
        sample_count = self.sample_count_spinner.value()
        
        # Update the voice model
        self.voice_model.required_samples = sample_count
        
        # Show training UI
        self.show_training_ui()

    def add_voice_samples(self):
        """Add multiple voice samples to the training dataset"""
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Voice Samples", "", "Audio Files (*.wav)")
        if not file_paths:
            return
            
        successful = 0
        required_samples = self.voice_model.required_samples
        remaining_needed = max(0, required_samples - self.voice_model.sample_count)
        files_to_process = min(len(file_paths), remaining_needed)
        
        if files_to_process == 0:
            QMessageBox.information(self, "Samples Complete", 
                                  "You have already collected enough voice samples.")
            return
            
        for i in range(files_to_process):
            try:
                audio = load_audio(file_paths[i])
                self.voice_model.add_sample(audio)
                successful += 1
                
                # Update UI
                self.progress_label.setText(f"Samples collected: {self.voice_model.sample_count}/{required_samples}")
                self.progress_bar.setValue(self.voice_model.sample_count)
                
                # Process events to show progress
                QApplication.processEvents()
                
            except Exception as e:
                print(f"Error processing {file_paths[i]}: {str(e)}")
        
        # Enable finish button if we have enough samples
        if self.voice_model.sample_count >= required_samples:
            self.finish_training_btn.setEnabled(True)
            
        QMessageBox.information(self, "Upload Complete", 
                              f"Successfully added {successful} voice samples.\n"
                              f"Total samples: {self.voice_model.sample_count}/{required_samples}")

    def batch_upload_samples(self):
        """Upload multiple voice samples at once from a directory"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Voice Samples")
        if not folder_path:
            return
            
        # Look for WAV files in the selected folder
        wav_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                     if f.lower().endswith('.wav')]
        
        if not wav_files:
            QMessageBox.warning(self, "No Samples Found", 
                              "No WAV files found in the selected folder. Please select a folder containing voice samples.")
            return
            
        # Calculate how many we need to process
        required_samples = self.voice_model.required_samples
        remaining_needed = max(0, required_samples - self.voice_model.sample_count)
        files_to_process = min(len(wav_files), remaining_needed)
        
        if files_to_process == 0:
            QMessageBox.information(self, "Samples Complete", 
                                  "You have already collected enough voice samples.")
            return
            
        # Process the files
        successful = 0
        for i in range(files_to_process):
            try:
                audio = load_audio(wav_files[i])
                self.voice_model.add_sample(audio)
                successful += 1
                
                # Update UI
                self.progress_label.setText(f"Samples collected: {self.voice_model.sample_count}/{required_samples}")
                self.progress_bar.setValue(self.voice_model.sample_count)
                
                # Process events to show progress
                QApplication.processEvents()
                
            except Exception as e:
                print(f"Error processing {wav_files[i]}: {str(e)}")
                
        # Update UI
        self.progress_label.setText(f"Samples collected: {self.voice_model.sample_count}/{required_samples}")
        self.progress_bar.setValue(self.voice_model.sample_count)
        
        # Enable finish button if we have enough samples
        if self.voice_model.sample_count >= required_samples:
            self.finish_training_btn.setEnabled(True)
            
        QMessageBox.information(self, "Batch Upload Complete", 
                              f"Successfully added {successful} voice samples.\n"
                              f"Total samples: {self.voice_model.sample_count}/{required_samples}")

    def complete_training(self):
        """Complete the training process and save the model"""
        required_samples = self.voice_model.required_samples
        if self.voice_model.sample_count < required_samples:
            QMessageBox.warning(self, "More Samples Needed", 
                               f"Please provide {required_samples - self.voice_model.sample_count} more voice samples.")
            return
        
        try:
            # Finalize training
            self.voice_model.train()
            # Save the model
            self.voice_model.save(VOICE_MODEL_PATH)
            QMessageBox.information(self, "Training Complete", 
                                   f"Voice model successfully trained with {self.voice_model.sample_count} samples!")
            
            # Switch to authentication UI
            self.show_authentication_ui()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to complete training: {str(e)}")

    def reset_training(self):
        """Reset the training process and delete any existing model"""
        reply = QMessageBox.question(self, "Confirm Reset", 
                                    "Are you sure you want to reset all training data? This cannot be undone.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                    
        if reply == QMessageBox.Yes:
            # Reset model
            self.voice_model = VoiceModel()
            
            # Delete saved model if it exists
            if os.path.exists(VOICE_MODEL_PATH):
                os.remove(VOICE_MODEL_PATH)
                
            # Show sample config UI
            self.show_sample_config_ui()

    def upload_voice_sample(self):
        """Upload a voice sample for authentication."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Voice Sample", "", "Audio Files (*.wav)")
        if file_path:
            self.auth_file = file_path
            self.auth_status.setText(f"Selected file: {os.path.basename(file_path)}")

    def authenticate_voice(self, purpose="access"):
        """Authenticate the user using a voice sample for a specific purpose."""
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select Voice Sample for {purpose}", "", "Audio Files (*.wav)")
        if not file_path:
            return False
            
        try:
            # Load and process the authentication sample
            auth_audio = load_audio(file_path)
            auth_embedding = encoder.embed_utterance(auth_audio)
            
            # Compare against trained model
            result = self.voice_model.authenticate(auth_embedding)
            
            if not result:
                QMessageBox.warning(self, "Authentication Failed", 
                                   f"Voice verification for {purpose} failed. Please try again with a clearer voice sample.")
            
            return result
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Authentication process failed: {str(e)}")
            return False

    def authenticate_vault(self):
        """Authenticate the user using the uploaded voice sample for vault access."""
        if not hasattr(self, 'auth_file') or self.auth_file is None:
            QMessageBox.warning(self, "Error", "Please upload a voice sample for authentication!")
            return

        try:
            # Load and process the authentication sample
            auth_audio = load_audio(self.auth_file)
            auth_embedding = encoder.embed_utterance(auth_audio)
            
            # Compare against trained model
            if self.voice_model.authenticate(auth_embedding):
                self.show_vault_ui()
            else:
                self.auth_status.setText("❌ Authentication Failed!")
                QMessageBox.warning(self, "Authentication Failed", 
                                   "Voice verification failed. Please try again with a clearer voice sample.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Authentication process failed: {str(e)}")

    def lock_vault(self):
        """Lock the vault by hiding encryption/decryption options."""
        self.show_authentication_ui()

    def encrypt_folder(self):
        """Encrypt a selected folder."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Encrypt")
        if folder_path:
            folder_name = os.path.basename(folder_path)
            
            # Create the ZIP file directly in the vault folder
            zip_path = os.path.join(VAULT_FOLDER, folder_name)
            encrypted_path = zip_path + ".enc"

            try:
                # Create ZIP archive in the vault folder
                shutil.make_archive(zip_path, 'zip', folder_path)
                
                # Encrypt the ZIP file
                key = hashlib.sha256(SECRET_KEY).digest()
                cipher = AES.new(key, AES.MODE_CBC)
                
                with open(zip_path + ".zip", "rb") as f:
                    plaintext = f.read()

                iv = cipher.iv
                ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))

                with open(encrypted_path, "wb") as f:
                    f.write(iv + ciphertext)

                # Remove the temporary ZIP file
                os.remove(zip_path + ".zip")
                
                # Ask if user wants to delete the original folder
                reply = QMessageBox.question(self, "Delete Original?", 
                                           f"Folder '{folder_name}' encrypted and stored in vault.\n\nDo you want to delete the original folder?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                           
                if reply == QMessageBox.Yes:
                    shutil.rmtree(folder_path)
                    QMessageBox.information(self, "Success", f"Original folder deleted. Encrypted data stored in vault.")
                else:
                    QMessageBox.information(self, "Success", f"Folder encrypted. Original folder preserved.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Encryption failed: {str(e)}")

    def decrypt_folder(self):
        """Decrypt a selected encrypted folder with voice authentication."""
        enc_file, _ = QFileDialog.getOpenFileName(self, "Select Encrypted Folder", VAULT_FOLDER, "Encrypted Files (*.enc)")
        if not enc_file:
            return
            
        # Get the folder name for more specific authentication message
        folder_name = os.path.basename(enc_file).replace(".enc", "")
            
        # Request voice authentication specifically for this decryption
        QMessageBox.information(self, "Voice Authentication Required", 
                              f"Please provide a voice sample to authorize decryption of '{folder_name}'.")
                              
        # Authenticate voice for decryption action
        if not self.authenticate_voice(purpose=f"decryption of '{folder_name}'"):
            return
                
        output_folder = enc_file.replace(".enc", "")

        try:
            decrypt_directory(enc_file, output_folder)
            QMessageBox.information(self, "Success", f"Folder restored successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Decryption failed: {str(e)}")


# ----------------- Audio Processing Functions -----------------
def load_audio(file):
    audio, _ = librosa.load(file, sr=16000)
    return audio


# ----------------- Folder Decryption -----------------
def decrypt_directory(encrypted_file, output_folder):
    key = hashlib.sha256(SECRET_KEY).digest()

    with open(encrypted_file, "rb") as f:
        iv = f.read(16)
        ciphertext = f.read()

    cipher = AES.new(key, AES.MODE_CBC, iv=iv)
    plaintext = unpad(cipher.decrypt(ciphertext), AES.block_size)

    zip_path = encrypted_file.replace(".enc", ".zip")
    with open(zip_path, "wb") as f:
        f.write(plaintext)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

    os.remove(zip_path)  # Remove temporary ZIP file
    # Keep the encrypted file as backup
    # os.remove(encrypted_file)  # Delete encrypted file


# ----------------- Run the GUI -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceVaultApp()
    window.show()
    sys.exit(app.exec_())