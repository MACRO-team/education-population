In order to train a new model you should use the following command:
```
python train.py --datapath <path-to-preprocessed-file>
```
<path-to-preprocessed-file> is a path to your preprocessed articles file that is created by the 'header_extract.py' script in the preprocessing_article folder. An example for this kind of file is 'dataset_master_with_sections_and_sentences.json'.

After you have the trained model, you should use the following command to test the model on your test set:
```
python test.py --datapath <path-to-preprocessed-file> --checkpoint <path-to-trained-model>
```