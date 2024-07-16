import os
import glob
import torch
import concurrent.futures
import numpy as np
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments

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

def split_audio_vad(audio_path, target_dir, split_seconds=10.0):
    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s, e in segments]

    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)
    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000): int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, 'wavs')
    os.makedirs(wavs_folder, exist_ok=True)

    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, 'input audio is too short'
    interval = audio_dur / num_splits

    audio_segments = []
    start_time = 0.0
    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        audio_seg = audio_active[int(start_time * 1000): int(end_time * 1000)]
        audio_segments.append((i, audio_seg))
        start_time = end_time

    def export_segment(args):
        i, audio_seg = args
        fname = f"{audio_name}_seg{i}.wav"
        output_file = os.path.join(wavs_folder, fname)
        audio_seg.export(output_file, format='wav')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(export_segment, audio_segments)

    return wavs_folder

def get_se(audio_path, vc_model, target_dir='processed', vad=True):
    device = vc_model.device

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if os.path.isfile(se_path):
        se = torch.load(se_path).to(device)
        return se, audio_name
    if os.path.isdir(audio_path):
        wavs_folder = audio_path
    elif vad:
        wavs_folder = split_audio_vad(audio_path, target_dir)
    else:
        wavs_folder = split_audio_whisper(audio_path, target_dir)

    audio_segs = glob(f'{wavs_folder}/*.wav')
    if len(audio_segs) == 0:
        raise NotImplementedError('No audio segments found!')

    return vc_model.extract_se(audio_segs, se_save_path=se_path), audio_name
