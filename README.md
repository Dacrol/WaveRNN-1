# WaveRNN Server

Web server wrapper for [WaveRNN](https://github.com/fatchord/WaveRNN) by fatchord. Start with `python server.py` and POST text to localhost:5700 to generate .wav-files.

# WaveRNN

Pytorch implementation of Deepmind's WaveRNN model from [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

# Installation

Ensure you have: 

* Python >= 3.6
* [Pytorch 1 with CUDA](https://pytorch.org/)

Then install the rest with pip:

> pip install -r requirements.txt

# How to Use

### Quick Start

If you want to use TTS functionality immediately you can simply use:

> python quick_start.py

This will generate everything in the default sentences.txt file and output to a new 'quick_start' folder where you can playback the wav files and take a look at the attention plots

You can also use that script to generate custom tts sentences and/or use '-u' to generate unbatched (better audio quality):

> python quick_start.py -u --input_text "What will happen if I run this command?'


### Training your own Models

Download the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) Dataset.

Edit **hparams.py**, point **wav_path** to your dataset and run: 

> python preprocess.py

or use preprocess.py --path to point directly to the dataset
___

Here's my recommendation on what order to run things: 

1 - Train Tacotron with:

> python train_tacotron.py

2 - You can leave that finish training or at any point you can use: 

> python train_tacotron.py --force_gta

this will force tactron to create a GTA dataset even if it hasn't finish training.

3 - Train WaveRNN with:

> python train_wavernn.py --gta

NB: You can always just run train_wavernn.py without --gta if you're not interested in TTS.

4 - Generate Sentences with both models using:

> python gen_tacotron.py

this will generate default sentences. If you want generate custom sentences you can use

> python gen_tacotron.py --input_text "this is whatever you want it to be"

And finally, you can always use --help on any of those scripts to see what options are available :)



# Samples

[Can be found here.](https://fatchord.github.io/model_outputs/)

# Pretrained Models

Currently there are two pretrained models available in the /pretrained/ folder':

Both are trained on LJSpeech

* WaveRNN trained to 800k steps (400k normal mels / 400k gta finetuned)
* Tacotron(r=1) trained to 196k steps

# Acknowledgments

* [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)
* [keithito tacotron](https://github.com/keithito/tacotron)
* Special thanks to github users [G-Wang](https://github.com/G-Wang), [geneing](https://github.com/geneing)




