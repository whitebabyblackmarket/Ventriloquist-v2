import os
import torch
import argparse
import pyaudio
import wave
from api import BaseSpeakerTTS, ToneColorConverter
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sentence_transformers import SentenceTransformer, util
import logging
import soundfile as sf
from pydub import AudioSegment
import requests
from OpenVoice.openvoice.se_extractor import get_se
from openai import OpenAI
import json 

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='voice_assistant.log', filemode='w')

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Set up the whisper model for STT
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"

try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    logging.info(f"Using device: {device}")
    print(f"Using device: {device}")

except Exception as e:
    logging.error(f"Error initializing model or pipeline: {e}")
    print(f"Error initializing model or pipeline: {e}")

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="YOUR_API_KEY")

# Function to play audio using PyAudio
def play_audio(file_path):
    try:
        logging.info("Playing audio.")
        wf = wave.open(file_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("Audio playback completed.")
    except Exception as e:
        logging.error(f"Error during audio playback: {e}")

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    try:
        logging.info("Starting text-to-speech synthesis.")
        tts_model = en_base_speaker_tts
        source_se = en_source_default_se if style == 'default' else en_source_style_se
        target_se, audio_name = get_se(audio_file_pth, tone_color_converter, target_dir='processed')
        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')
        save_path = f'{output_dir}/output.wav'
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)
        logging.info("Audio generated successfully.")
        play_audio(save_path)
    except Exception as e:
        logging.error(f"Error during audio generation: {e}")


# Testing the transcription functionality
def test_inference(audio_file):
    try:
        result = pipe(audio_file)
        logging.info(f"Transcription result: {result}")
        print(f"Transcription result: {result}")
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        print(f"Error during inference: {e}")

# Replace 'path_to_audio_file.wav' with your actual audio file path for testing
test_inference('C:\\Users\\white\\Desktop\\ventriloquist_v2\\processed\\joa\\wavs\\joa_seg0.wav')

def get_relevant_context(user_input, vault_embeddings, vault_content, model, top_k=3):
    try:
        if vault_embeddings.nelement() == 0:
            return []

        input_embedding = model.encode([user_input])
        cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        relevant_context = [vault_content[idx].strip() for idx in top_indices]
        return relevant_context
    except Exception as e:
        logging.error(f"Error retrieving relevant context: {e}")
        return []

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name, vault_embeddings, vault_content, model):
    try:
        logging.info("Starting interaction with local LLM.")
        relevant_context = get_relevant_context(user_input, vault_embeddings, vault_content, model)
        user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input if relevant_context else user_input
        messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input_with_context}]
        
        payload = {
            "model": "cognitivecomputations/dolphin-2.9-llama3-8b-gguf",
            "messages": messages,
            "temperature": 1,
            "stream": True
        }

        with requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True) as response:
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(line)
                            if "choices" in chunk_data:
                                delta_content = chunk_data["choices"][0]["delta"].get("content", "")
                                if delta_content:
                                    print(NEON_GREEN + delta_content + RESET_COLOR, end='', flush=True)
                                    full_response += delta_content
                        except json.JSONDecodeError:
                            logging.warning(f"Failed to decode JSON: {line}")

        print()  # New line after streaming response
        logging.info(f"Generated response: {full_response}")
        return full_response
    except Exception as e:
        logging.error(f"Error during LLM interaction: {str(e)}")
        return "Sorry, I encountered an error while processing your request."

# Function to transcribe the recorded audio using whisper
def transcribe_with_whisper(audio_file):
    try:
        logging.info("Starting transcription with Whisper model.")
        result = pipe(audio_file)
        transcription = result["text"]
        logging.info(f"Transcription result: {transcription}")
        return transcription.strip()
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

# Function to record audio from the microphone and save to a file
def record_audio(file_path):
    try:
        logging.info("Starting audio recording.")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        print("Recording...")
        try:
            while True:
                data = stream.read(1024)
                frames.append(data)
        except KeyboardInterrupt:
            pass
        print("Recording stopped.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()
        logging.info("Audio recording completed.")
    except Exception as e:
        logging.error(f"Error during audio recording: {e}")

# New function to handle a conversation with a user
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot2.txt")
    # Load the sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load the initial content from the vault.txt file
    vault_content = []
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding="utf-8") as vault_file:
            vault_content = vault_file.readlines()
    # Create embeddings for the initial vault content
    vault_embeddings = model.encode(vault_content) if vault_content else []
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file
        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break
        elif user_input.lower().startswith(("print info", "Print info")):  # Print the contents of the vault.txt file
            print("Info contents:")
            if os.path.exists("vault.txt"):
                with open("vault.txt", "r", encoding="utf-8") as vault_file:
                    print(NEON_GREEN + vault_file.read() + RESET_COLOR)
            else:
                print("Info is empty.")
            continue
        elif user_input.lower().startswith(("delete info", "Delete info")):  # Delete the vault.txt file
            confirm = input("Are you sure? Say 'Yes' to confirm: ")
            if confirm.lower() == "yes":
                if os.path.exists("vault.txt"):
                    os.remove("vault.txt")
                    print("Info deleted.")
                    vault_content = []
                    vault_embeddings = []
                    vault_embeddings_tensor = torch.tensor(vault_embeddings)
                else:
                    print("Info is already empty.")
            else:
                print("Info deletion cancelled.")
            continue
        elif user_input.lower().startswith(("insert info", "Insert info")):
            print("Recording for info...")
            audio_file = "vault_recording.wav"
            record_audio(audio_file)
            vault_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)  # Clean up the temporary audio file
            with open("vault.txt", "a", encoding="utf-8") as vault_file:
                vault_file.write(vault_input + "\n")
            print("Wrote to info.")
            # Update the vault content and embeddings
            vault_content = open("vault.txt", "r", encoding="utf-8").readlines()
            vault_embeddings = model.encode(vault_content)
            vault_embeddings_tensor = torch.tensor(vault_embeddings)
            continue
        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Emma:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot", vault_embeddings_tensor, vault_content, model)
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        prompt2 = chatbot_response
        style = "default"
        audio_file_pth2 = "PATH/joa.mp3"
        process_and_play(prompt2, style, audio_file_pth2)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation