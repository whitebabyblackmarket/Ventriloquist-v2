# Ventriloquist v2

Ventriloquist v2 is an AI-powered voice assistant that combines speech recognition, natural language processing, and text-to-speech capabilities using OpenVoice technology.

## Prerequisites

- Windows 10 or later
- Python 3.9 or later
- CUDA-capable GPU (recommended for faster processing)
- [Git](https://git-scm.com/download/win) (for cloning the repository)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ventriloquist_v2.git
   cd ventriloquist_v2
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the necessary model checkpoints:
   - Create a `checkpoints` folder in the project root
   - Download the OpenVoice base speaker and converter checkpoints
   - Place them in the `checkpoints` folder, maintaining the following structure:
     ```
     checkpoints/
     ├── base_speakers/
     │   └── EN/
     │       ├── config.json
     │       ├── checkpoint.pth
     │       ├── en_default_se.pth
     │       └── en_style_se.pth
     └── converter/
         ├── config.json
         └── checkpoint.pth
     ```

5. Set up LM Studio:
   - Download and install [LM Studio](https://lmstudio.ai/)
   - Launch LM Studio and select a suitable language model
   - Start the local server in LM Studio (usually runs on http://localhost:1234)

## Configuration

1. Update the `audio_file_pth2` variable in `talk3.py` to point to your reference audio file:
   ```python
   audio_file_pth2 = r"C:\path\to\your\reference_audio.mp3"
   ```

2. If LM Studio requires authentication, update the API key in `talk3.py`:
   ```python
   client = OpenAI(base_url="http://localhost:1234/v1", api_key="YOUR_API_KEY")
   ```

## Usage

1. Ensure your virtual environment is activated:
   ```
   .\venv\Scripts\activate
   ```

2. Run the main script:
   ```
   python talk3.py
   ```

3. The program will prompt you to speak. You can:
   - Say "exit" to end the conversation
   - Say "print info" to display the contents of vault.txt
   - Say "delete info" to clear the contents of vault.txt
   - Say "insert info" to add new information to vault.txt

4. The AI assistant will respond verbally and in text format.

## Troubleshooting

- If you encounter CUDA-related errors, ensure you have the latest NVIDIA drivers installed and that your GPU is CUDA-compatible.
- If you face issues with audio playback or recording, check your system's audio settings and ensure PyAudio is correctly installed.
- For LM Studio connection problems, verify that the local server is running and the URL/port in the script matches LM Studio's settings.

## License

[Specify your license here]

## Acknowledgments

- OpenVoice for text-to-speech technology
- Hugging Face for the Transformers library
- LM Studio for local language model hosting
