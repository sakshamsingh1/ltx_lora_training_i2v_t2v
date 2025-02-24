
import torch
import librosa
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import numpy as np

MAX_WAV_VALUE = 32768.0
mel_basis = {}
hann_window = {}

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_mel_spectrogram_from_audio(audio, device="cuda"):
    audio = audio / MAX_WAV_VALUE
    audio = librosa.util.normalize(audio) * 0.95
        
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)    

    waveform = audio.to(device)
    spec = mel_spectrogram(waveform, n_fft=2048, num_mels=256, sampling_rate=16000, hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)
    return audio, spec

def normalize_spectrogram(
    spectrogram,
    max_value = 200, 
    min_value = 1e-5, 
    power = 1., 
    inverse = False
    ):   
    # Rescale to 0-1
    max_value = np.log(max_value) # 5.298317366548036
    min_value = np.log(min_value) # -11.512925464970229

    assert spectrogram.max() <= max_value and spectrogram.min() >= min_value

    data = (spectrogram - min_value) / (max_value - min_value)

    # Invert
    if inverse:
        data = 1 - data

    # Apply the power curve
    data = torch.pow(data, power)  
    
    # 1D -> 3D
    data = data.repeat(3, 1, 1)

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    return data


def denormalize_spectrogram(
    data: torch.Tensor,
    max_value = 200, 
    min_value = 1e-5, 
    power = 1, 
    inverse = False,
    ):
    
    max_value = np.log(max_value)
    min_value = np.log(min_value)

    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])

    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))
    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)
        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]

    # Reverse the power curve
    data = torch.pow(data, 1 / power)

    # Invert
    if inverse:
        data = 1 - data

    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

######################################### UTILS #########################################
import torch.nn.functional as F
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from PIL import Image

def pad_spec(spec, spec_length, pad_value=0, random_crop=True): # spec: [3, mel_dim, spec_len]
    assert spec_length % 8 == 0, "spec_length must be divisible by 8"
    if spec.shape[-1] < spec_length:
        # pad spec to spec_length
        spec = F.pad(spec, (0, spec_length - spec.shape[-1]), value=pad_value)
    else:
        # random crop
        if random_crop:
            start = random.randint(0, spec.shape[-1] - spec_length)
            spec = spec[:, :, start:start+spec_length]
        else:
            spec = spec[:, :, :spec_length]
    return spec

def normalize(images):
    if images.min() >= 0:
        return 2.0 * images - 1.0
    else:
        return images

def denormalize(images):
    """
    Denormalize an image array to [0,1].
    """
    if images.min() < 0:
        return (images / 2 + 0.5).clamp(0, 1)
    else:
        return images.clamp(0, 1)     

def torch_to_pil(image):
    """
    Convert a torch tensor to a PIL image.
    """
    if image.min() < 0:
        image = denormalize(image)

    return transforms.ToPILImage()(image.cpu().detach().squeeze())    

def image_add_color(spec_img):
    cmap = plt.get_cmap('viridis')
    cmap_r = cmap.reversed()
    image = cmap(np.array(spec_img)[:,:,0])[:, :, :3]  # 省略透明度通道
    image = (image - image.min()) / (image.max() - image.min())
    image = Image.fromarray(np.uint8(image*255))
    return image