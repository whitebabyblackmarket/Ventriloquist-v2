import os
import glob
import torch
from glob import glob
import numpy as np
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"  # Update if using a different model ID

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

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

def split_audio_whisper(audio_path, target_dir='processed'):
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)
    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    target_folder = os.path.join(target_dir, audio_name)

    result = pipe(audio_path, return_timestamps=True)
    segments = result["chunks"]

    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    audio_segments = []
    for i, segment in enumerate(segments):
        start_time = max(0, segment['timestamp']['start'])
        end_time = segment['timestamp']['end']
        audio_seg = audio[int(start_time * 1000): min(max_len, int(end_time * 1000) + 80)]
        audio_segments.append((i, audio_seg))

    def export_segment(args):
        i, audio_seg = args
        fname = f"{audio_name}_seg{i}.wav"
        output_file = os.path.join(wavs_folder, fname)
        audio_seg.export(output_file, format='wav')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(export_segment, audio_segments)

    return wavs_folder

# The rest of the functions in the file remain the same as before.
