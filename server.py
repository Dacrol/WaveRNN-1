import torch
from models.fatchord_wavernn import Model
import hparams as hp
from utils.text.symbols import symbols
from utils.paths import Paths
from models.tacotron import Tacotron
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table

from flask import Flask, request, send_file
import ptvsd
import os
import uuid


if os.environ.get('RUN_MAIN') or os.environ.get('WERKZEUG_RUN_MAIN'):
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)

# ptvsd.wait_for_attach()

app = Flask(__name__)


@app.route('/synth', methods=['POST'])
def synth():
    body = request.get_json(force=True)
    if 'text' not in body:
        return 'No text supplied'
    unbatched = False
    if 'unbatched' in body and body['unbatched'] == 'true':
        unbatched = True
    text = body['text']
    file = generate(text, unbatched)
    return send_file(file, attachment_filename=file.rsplit('/', 1)[-1])


def generate(text, unbatched):

    batched = not unbatched
    target = hp.voc_target
    overlap = hp.voc_overlap
    input_text = text
    weights_path = None
    save_attn = False

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    print('\nInitialising WaveRNN Model...\n')

    # Instantiate WaveRNN Model
    voc_model = Model(rnn_dims=hp.voc_rnn_dims,
                      fc_dims=hp.voc_fc_dims,
                      bits=hp.bits,
                      pad=hp.voc_pad,
                      upsample_factors=hp.voc_upsample_factors,
                      feat_dims=hp.num_mels,
                      compute_dims=hp.voc_compute_dims,
                      res_out_dims=hp.voc_res_out_dims,
                      res_blocks=hp.voc_res_blocks,
                      hop_length=hp.hop_length,
                      sample_rate=hp.sample_rate).cuda()

    voc_model.restore(paths.voc_latest_weights)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout).cuda()

    tts_restore_path = weights_path if weights_path else paths.tts_latest_weights
    tts_model.restore(tts_restore_path)

    if input_text:
        inputs = [text_to_sequence(input_text.strip(), hp.tts_cleaner_names)]
    else:
        with open('sentences.txt') as f:
            inputs = [text_to_sequence(
                l.strip(), hp.tts_cleaner_names) for l in f]

    voc_k = voc_model.get_step() // 1000
    tts_k = tts_model.get_step() // 1000

    simple_table([('WaveRNN', str(voc_k) + 'k'),
                  ('Tacotron', str(tts_k) + 'k'),
                  ('r', tts_model.r.item()),
                  ('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)

        os.makedirs('server_output/', exist_ok=True)
        save_path = f'server_output/{str(uuid.uuid1())}.wav'

        if save_attn:
            save_attention(attention, save_path)

        m = torch.tensor(m).unsqueeze(0)
        m = (m + 4) / 8

        voc_model.generate(m, save_path, batched,
                           hp.voc_target, hp.voc_overlap, hp.mu_law)

    print(f'\n\nDone. File: {save_path}\n')
    return save_path


if __name__ == '__main__':
    app.run(port=5700, debug=True)
