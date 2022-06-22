from IPython.display import Audio
from IPython.utils import io
from IPython.display import display
from synthesizer.inference import Synthesizer
import soundfile as sf
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa

encoder_weights = Path("saved_models/default/encoder.pt")
vocoder_weights = Path("saved_models/default/vocoder.pt")
syn_dir = Path("saved_models/default/synthesizer.pt")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

text = "There once was a ship that put to sea The name of the ship was the Billy of Tea The winds blew up, her bow dipped down Oh blow, my bully boys, blow"

in_fpath = Path("Jim.wav")
reprocessed_wav = encoder.preprocess_wav(in_fpath)
original_wav, sampling_rate = librosa.load(in_fpath)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embed = encoder.embed_utterance(preprocessed_wav)
with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
generated_wav = vocoder.infer_waveform(specs[0])
generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
sf.write("Jim_sample.wav", generated_wav, synthesizer.sample_rate)