Voice-Secured Vault
Voice-Secured Vault is a Python desktop application that allows you to encrypt and store entire folders in a secure vault. Access to the vault, and the ability to decrypt files, is protected by your voice. The application uses voice biometrics to authenticate the user, ensuring that only the authorized person can access the encrypted data.

<!-- Replace with a screenshot of your application -->

Features
Voice-Powered Authentication: Uses your unique voiceprint to grant access.

Secure Folder Encryption: Encrypts entire folders using AES-256 encryption.

Easy-to-Use GUI: A simple graphical user interface built with PyQt5 for all operations.

Configurable Training: Customize the number of voice samples required for training the model for different security levels.

Secure Storage: Encrypted files are stored in a dedicated vault_storage folder.

Batch Sample Upload: Quickly train the model by uploading multiple .wav files at once.

How It Works
Voice Sampling: The first time you run the application, it prompts you to provide a number of voice samples (short .wav files of your speech).

Model Training: It uses the resemblyzer library to create a "voiceprint" or a mathematical representation (embedding) of your voice from these samples. This voice model is then saved locally in voice_model.pkl.

Authentication: To unlock the vault, you must provide a new voice sample. The application compares this sample's embedding to the trained model. If the similarity is high enough, access is granted.

Encryption: When you choose to encrypt a folder, the application first archives it into a .zip file. This ZIP file is then encrypted using AES-256. The resulting .enc file is stored in the vault_storage directory.

Decryption: To decrypt a file, you must first authenticate with your voice. Upon successful authentication, the application decrypts the .enc file back into a ZIP archive and then extracts its contents to a folder.

Requirements
The script requires the following Python libraries:

PyQt5

librosa

numpy

resemblyzer

pycryptodome

You will also need ffmpeg for resemblyzer to work correctly. Please refer to the resemblyzer documentation for installation instructions for your specific operating system.

Installation
Clone the repository:

git clone [https://github.com/your-username/voice-secured-vault.git](https://github.com/your-username/voice-secured-vault.git)
cd voice-secured-vault

Install the required Python packages:

pip install PyQt5 librosa numpy resemblyzer pycryptodome

Note: resemblyzer may require you to install PyTorch separately. Please follow the instructions on the official PyTorch website for your system.

Usage
Run the application:

python voicezip.py

First-Time Setup (Training):

On the first run, you will be asked to configure the number of voice samples for training. A higher number provides better security.

Click "Start Training".

Use the "Add Voice Samples" or "Batch Upload Samples" button to provide .wav files of your voice. The samples should be 3-5 seconds long and contain clear speech.

Once you have provided the required number of samples, the "Complete Training" button will be enabled. Click it to finalize your voice model.

Unlocking the Vault:

After the model is trained, the application will lock.

To unlock it, click "Upload Voice Sample" and select a new, fresh recording of your voice.

Click "Authenticate and Unlock Vault". If the voice matches, the vault will open.

Encrypting a Folder:

Once the vault is unlocked, click "Encrypt Folder".

Select the folder you wish to secure.

The application will encrypt it and save it in the vault_storage folder. You will be given the option to delete the original, unencrypted folder.

Decrypting a Folder:

Click "Decrypt Folder".

Select the .enc file from the vault_storage folder.

You will be prompted to provide a voice sample to authorize the decryption.

If authenticated, the folder will be decrypted and restored in the same directory.

⚠️ Security Warning
This application is intended as a practical demonstration of voice biometrics. For production-level security, you MUST change the hardcoded SECRET_KEY in the voicezip.py script.

# Secure encryption key (change this!)
SECRET_KEY = b"your-strong-password"

A hardcoded key is insecure. In a real-world application, this key should be managed securely, for example, by deriving it from a user's password using a key derivation function (like PBKDF2) and a salt.

File Structure
voicezip.py: The main Python script for the application.

voice_model.pkl: The saved, trained voice model. Do not share this file.

vault_storage/: The directory where all encrypted folders are stored.

License
This project is licensed under the MIT License. See the LICENSE file for details.
