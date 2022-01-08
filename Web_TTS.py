from flask import Flask, render_template, redirect, session, url_for, request
from flask_mysqldb import MySQL
import MySQLdb
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd

from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor

app = Flask(__name__)
app.secret_key = "12345"

app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "root"
app.config["MYSQL_DB"] = "audio"

db = MySQL(app)

tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en", name="tacotron2")

fastspeech = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech-ljspeech-en", name="fastspeech")

fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en", name="fastspeech2")

melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en", name="melgan")

melgan_stft_config = AutoConfig.from_pretrained("TensorFlowTTS/examples/melgan_stft/conf/melgan_stft.v1.yaml")
melgan_stft = TFAutoModel.from_pretrained(
    config=melgan_stft_config,
    pretrained_path="melgan.stft-2M.h5",
    name="melgan_stft"
)

mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en", name="mb_melgan")

processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")

def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
  input_ids = processor.text_to_sequence(input_text)

  # text2mel part
  if text2mel_name == "TACOTRON":
    _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        tf.convert_to_tensor([len(input_ids)], tf.int32),
        tf.convert_to_tensor([0], dtype=tf.int32)
    )
  elif text2mel_name == "FASTSPEECH":
    mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
        input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  elif text2mel_name == "FASTSPEECH2":
    mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
        tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
        speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
    )
  else:
    raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

  # vocoder part
  if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  elif vocoder_name == "MB-MELGAN":
    audio = vocoder_model(mel_outputs)[0, :, 0]
  else:
    raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

  if text2mel_name == "TACOTRON":
    return mel_outputs.numpy(), audio.numpy()
  else:
    return mel_outputs.numpy(), audio.numpy()

def visualize_mel_spectrogram(mels,namer):
  namer = 'static/Output/Graph/' + namer + '-img.png'
  mels = tf.reshape(mels, [-1, 80]).numpy()
  fig = plt.figure(figsize=(10, 8))
  ax1 = fig.add_subplot(311)
  ax1.set_title(f'Predicted Mel-after-Spectrogram')
  im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
  fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
  plt.savefig(namer, bbox_inches='tight')
  plt.close()

@app.route('/', methods=["GET", "POST"])
def profile():
    msg = ''
    cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM info')
    info_content = cursor.fetchone()
    cursor.execute("SELECT * FROM info WHERE id = (SELECT MAX(id) FROM info)")
    info_num = cursor.fetchone()
    try:
        info_num = info_num['id']
    except:
        info_num = 0
    info_num = int(info_num) + 1
    aud_info = {}
    for x in range(1,info_num):
        cursor.execute("SELECT * FROM info WHERE id = {number}".format(number=x))
        i0 = cursor.fetchone()
        i1 = i0['name']
        i4 = 'Output/Graph/' + i1 + '-img.png'
        i1 = 'Output/Audio/' + i1 + '-aud.wav'
        i2 = i0['mel']
        i3 = i0['vocoder']
        aud_info[x] = (i1, i2, i3, i4)
    if request.method == 'POST' and 'File_name' in request.form and 'Input_Field' in request.form and 'Text2Mel' in request.form and 'Vocoder' in request.form:
        Aud_name = request.form['File_name']
        Aud_text = request.form['Input_Field']
        Aud_mel = request.form['Text2Mel']
        Aud_voc = request.form['Vocoder']
        cursor = db.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM info WHERE name=%s',[Aud_name])
        file = cursor.fetchone()
        if file:
            msg = 'File name already exists!'
        elif ' ' in Aud_name:
            msg = 'File name should not have spaces!'
        else:
            if Aud_mel == 'TACOTRON':
                if Aud_voc == 'MELGAN': mels, audios = do_synthesis(Aud_text, tacotron2, melgan, Aud_mel, Aud_voc)
                elif Aud_voc == 'MELGAN-STFT': mels, audios = do_synthesis(Aud_text, tacotron2, melgan_stft, Aud_mel, Aud_voc)
                elif Aud_voc == 'MB-MELGAN': mels, audios = do_synthesis(Aud_text, tacotron2, mb_melgan, Aud_mel, Aud_voc)
            if Aud_mel == 'FASTSPEECH':
                if Aud_voc == 'MELGAN': mels, audios = do_synthesis(Aud_text, fastspeech, melgan, Aud_mel, Aud_voc)
                elif Aud_voc == 'MELGAN-STFT': mels, audios = do_synthesis(Aud_text, fastspeech, melgan_stft, Aud_mel, Aud_voc)
                elif Aud_voc == 'MB-MELGAN': mels, audios = do_synthesis(Aud_text, fastspeech, mb_melgan, Aud_mel, Aud_voc)
            if Aud_mel == 'FASTSPEECH2':
                if Aud_voc == 'MELGAN': mels, audios = do_synthesis(Aud_text, fastspeech2, melgan, Aud_mel, Aud_voc)
                elif Aud_voc == 'MELGAN-STFT': mels, audios = do_synthesis(Aud_text, fastspeech2, melgan_stft, Aud_mel, Aud_voc)
                elif Aud_voc == 'MB-MELGAN': mels, audios = do_synthesis(Aud_text, fastspeech2, mb_melgan, Aud_mel, Aud_voc)
            visualize_mel_spectrogram(mels[0],Aud_name)
            Auder = 'static/Output/Audio/' + Aud_name + '-aud.wav'
            sf.write(Auder, audios, 22050)
            cursor.execute('INSERT INTO info(name,text,mel,vocoder)VALUES(%s, %s, %s, %s)',(Aud_name,Aud_text,Aud_mel,Aud_voc))
            db.connection.commit()
            return redirect('/')
    elif request.method == 'POST':
        msg = 'Please fill out the form !'
    return render_template("profile.html", msg=msg, info_content=info_content, aud_info=aud_info)

if __name__ == '__main__':
    app.run(debug=True)
