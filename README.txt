Jointly Learning Latent Categorizations of Interview Prompts to Predict Depression in Screening Interview Transcripts
Proceedings of ACL 2020

Alex Rinaldi - Department of Computer Science, UC Santa Cruz - arinaldi@ucsc.edu
Jean E. Fox Tree - Department of Psychology, UC Santa Cruz - foxtree@ucsc.edu
Snigdha Chaturvedi - Department of Computer Science, University of North Carolina at Chapel Hill - snigdha@cs.unc.edu

This repository contains code to reproduce results from our ACL 2020 paper. A Python 3.6.7 environment with the following packages is required:

-mxnet 1.4.1
-gluonnlp 0.6.0
-scikit-learn 0.20.1
-scipy 1.1.0
-numpy 1.14.6
-tensorflow 1.13.1

Here is a description of pertinent files and folders:
	
-models - Class files for proposed and baseline models
-train_export.py  - Train and output development and test -predictions for any of our models (instructions below)
-data_provider.py - Parses and preprocesses conversation transcripts into a turn structure
-features_provider.py - Extracts averaged glove embedding features for each prompt/response
evaluate_metrics.py - Evaluate the F1+, F1-, and accuracy scores for a trained model's development or test predictions (instructions below)
-evaluate_significance.py - Evaluate two models' performances for statistical significance (instructions below).

Data

Access to the DAIC dataset must be requested individually. Once the data has been obtained, the csv files for transcripts should be placed in the Data directory, keeping the default filenames (<participant_number>_TRANSCRIPT.csv). Additionally, include the following files:
-full_test_split.csv
-test_split_Depression_AVEC2017.csv
-train_split_Depression_AVEC2017.csv
-dev_split_Depression_AVEC2017.csv

The already-existing file participant_indices.txt contains the pool of all participant IDs used to generate the splits.

Train/test/development splits are chosen by a seeded random number generator. The file reported_splits.txt contains the splits used to report results for our paper.

Training
To reproduce results for our models, first train the model using one of the following commands:

1. PO 
python train_export.py --experiment_prefix=glove_reprod_prompt_baseline --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=2000 --num_tests=10 --evaluate_every=100 --model=prompt_baseline --prompt_latent_type__dropout_keep=0.9 --learning_rate=0.003 --save_final_epoch=True

2. RO
python train_export.py --experiment_prefix=glove_reprod_response_baseline --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=2000 --num_tests=10 --evaluate_every=100 --model=response_baseline --prompt_latent_type__dropout_keep=0.9 --learning_rate=0.003 --save_final_epoch=True

3. PR
python train_export.py --experiment_prefix=glove_reprod_promptresponse_baseline --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=2000 --num_tests=10 --evaluate_every=100 --model=promptresponse_baseline --prompt_latent_type__dropout_keep=0.9 --learning_rate=0.003 --save_final_epoch=True

4. JLPC
python train_export.py --experiment_prefix=reprod_glove_latenttype_k11entropy --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=1700 --num_tests=10 --evaluate_every=100 --model=prompt_latent_type_latent_entropy --prompt_latent_type__num_channels=11 --prompt_latent_type__dropout_keep=0.9 --entropy_coefficient=0.1 --learning_rate=0.0005 --save_final_epoch=True

5. JLPCPre
python train_export.py --experiment_prefix=glove_latenttype_deepconcatk11entropy --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=1400 --num_tests=10 --evaluate_every=100 --model=prompt_latent_type_concat_latent_entropy --prompt_latent_type__num_channels=11 --prompt_latent_type__dropout_keep=0.9 --entropy_coefficient=1 --learning_rate=0.001 --save_final_epoch=True

6. JLPCPost
python train_export.py --experiment_prefix=glove_latenttype_concatk11entropy --train_size=0.7  --dev_size=0.2 --batch_size=10 --num_epochs=1300 --num_tests=10 --evaluate_every=100 --model=prompt_latent_type_concatv2_latent_entropy --prompt_latent_type__num_channels=11 --prompt_latent_type__dropout_keep=0.9 --entropy_coefficient=0.1 --learning_rate=0.0005 --save_final_epoch=True

Each model will output the following line when training is finished:
"Wrote everything to <path>/runs/<experiment_prefix>/<results_set_number>"

Evaluation

To evaluate the results of a trained model, run one of the following commands:

1. Dev set: F1 +/-, Accuracy
python evaluate_metrics.py runs/<experiment_prefix>/<results_set_number> 10  dev_predictions.csv

2. Test set: F1 +/- Accuracy
python evaluate_metrics.py runs/<experiment_prefix>/<results_set_number> 10  test_predictions.csv

The outputs of these two commands are formated as:
F1 pos: <mean> (<stdev>) : <median>
F1 neg: <mean> (<stdev>) : <median>
Accuracy: <mean> (<stdev>) : <median>

3. Test set statistical significance
python evaluate_metrics.py runs/<baseline_experiment_prefix>/<results_set_number> runs/<model_experiment_prefix>/<results_set_number> 10  

The output for this command follows the format provided by the scipy package's ttest_ind functionality.



