# Scope
In this project, the aim has been to use an autoencoder to generate speach from text. Its been trained on my voice, using the audio clips in ./audio and the corresponding text in ./sentences.txt

# How to use
Run autoencoder.py to run the autoencoder, the audio_reader.py and text_reader.py are used as utility libraries for the autoencoder. 

# Predictions
Pred_output_\*.wav are saved outputs. 

in pred_output_100_iter_36_dataset_no_augment predictions were run on 10 audiofiles (from before more were made) with no augmentations for 100 augmentations. At this point there were a huge veriety of audio lenghts and thus the ends of the files were mostly silent. This caused the model to overfit to predict only silence. 

In pred_output_100_iter_100_dataset_using_augmentations, the outputs have been augmented with 1 input => 20 inputs. All audiofiles have also been re-recorded to fit within one second of eachother (all are between 3 and 4 secounds long). There is also significantly more audio clips, having gone from 36 to 100
