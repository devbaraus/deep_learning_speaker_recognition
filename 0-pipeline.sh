source ./venv/Scripts/activate

python 1-merge_audios.py
python 2-process_mfcc.py
python 3-run_perceptron-svm.py