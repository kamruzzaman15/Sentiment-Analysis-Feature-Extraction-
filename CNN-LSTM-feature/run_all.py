import subprocess

# List of Python files to run
files = ['CNN-LSTM-feature-svm-pro.py', 'CNN-LSTM-feature-vot-pro.py', 'CNN-LSTM-feature-boost-pro.py', 'CNN-LSTM-feature-svm-res.py','CNN-LSTM-feature-vot-res.py','CNN-LSTM-feature-boost-res.py','CNN-LSTM-feature-svm-IMDB.py','CNN-LSTM-feature-vot-IMDB.py','CNN-LSTM-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-CNN-LSTM.txt', 'w') as f:
    # Loop through each file
    for file in files:
        try:
            # Run the Python file and capture the output
            output = subprocess.check_output(['python3', file], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # Handle any subprocess errors
            output = e.output
        # Write the output to the results file
        f.write(f'Results for {file}:\n\n{output.decode()}\n\n')

