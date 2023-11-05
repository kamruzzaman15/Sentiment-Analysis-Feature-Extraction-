import subprocess

# List of Python files to run
files = ['LSTM-feature-svm-pro.py', 'LSTM-feature-vot-pro.py', 'LSTM-feature-boost-pro.py', 'LSTM-feature-svm-res.py','LSTM-feature-vot-res.py','LSTM-feature-boost-res.py','LSTM-feature-svm-IMDB.py','LSTM-feature-vot-IMDB.py','LSTM-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-LSTM-feature.txt', 'w') as f:
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

