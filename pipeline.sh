# python3 merge_audios.py -l portuguese
# python3 merge_audios.py -l english

# python3 process.py -l portuguese -r psf,melbanks
# python3 process.py -l english -r psf,melbanks
# python3 process.py -l english -r psf,melbanks -s 65 -p 109
# python3 process.py -l english -r psf,melbanks -s 80 -p 80

# python3 perceptron.py -l portuguese -r psf
# python3 perceptron.py -l portuguese -r melbanks
# python3 perceptron.py -l portuguese -r mixed

# python3 perceptron.py -l english -r psf -p 109 -s 65
# python3 perceptron.py -l english -r melbanks -p 109 -s 65
# python3 perceptron.py -l english -r mixed -p 109 -s 65

# python3 perceptron.py -l english -r psf -s 80 -p 80
# python3 perceptron.py -l english -r melbanks -s 80 -p 80
# python3 perceptron.py -l english -r mixed -s 80 -p 80

# python3 perceptron.py -l mixed -r psf -p 109 -s 65
# python3 perceptron.py -l mixed -r melbanks -p 109 -s 65
# python3 perceptron.py -l mixed -r mixed -p 109 -s 65

# python3 perceptron.py -l english -r psf
# python3 perceptron.py -l english -r melbanks
# python3 perceptron.py -l english -r mixed

# python3 perceptron.py -l mixed -r psf 
# python3 perceptron.py -l mixed -r melbanks 
# python3 perceptron.py -l mixed -r mixed 

# python3 svm.py -l portuguese -r psf
# python3 svm.py -l portuguese -r melbanks
# python3 svm.py -l portuguese -r mixed

# python3 svm.py -l english -r psf
# python3 svm.py -l english -r melbanks
# python3 svm.py -l english -r mixed

python3 svm.py -l mixed -r psf
python3 svm.py -l mixed -r melbanks
python3 svm.py -l mixed -r mixed