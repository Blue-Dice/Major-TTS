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

text = "The Solar System consists of the Sun Moon and Planets. It also consists of comets, meteoroids and asteroids. The Sun is the largest member of the Solar System. In order of distance from the Sun, the planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto; the dwarf planet. The Sun is at the centre of the Solar System and the planets, asteroids, comets and meteoroids revolve around it."

in_fpath = Path("Dhairya.wav")
reprocessed_wav = encoder.preprocess_wav(in_fpath)
original_wav, sampling_rate = librosa.load(in_fpath)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
embed = encoder.embed_utterance(preprocessed_wav)
with io.capture_output() as captured:
  specs = synthesizer.synthesize_spectrograms([text], [embed])
generated_wav = vocoder.infer_waveform(specs[0])
generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
sf.write("Dhairya_sample.wav", generated_wav, synthesizer.sample_rate)