import os
import torch
import librosa

def get_se(audio_path, vc_model, target_dir='processed'):
    device = vc_model.device

    audio_name = os.path.basename(audio_path).rsplit('.', 1)[0]
    se_path = os.path.join(target_dir, audio_name, 'se.pth')

    if os.path.isfile(se_path):
        se = torch.load(se_path).to(device)
        return se, audio_name

    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=vc_model.hps.data.sampling_rate)

    # Convert to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

    # Extract speaker embedding
    se = vc_model.extract_se(audio_tensor)

    # Save the speaker embedding
    os.makedirs(os.path.dirname(se_path), exist_ok=True)
    torch.save(se.cpu(), se_path)

    return se, audio_name