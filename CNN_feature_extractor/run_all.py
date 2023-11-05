import subprocess

# List of Python files to run
files = ['CNN-feature-svm-pro.py', 'CNN-feature-vot-pro.py', 'CNN-feature-boost-pro.py', 'CNN-feature-svm-res.py','CNN-feature-vot-res.py','CNN-feature-boost-res.py','CNN-feature-svm-IMDB.py','CNN-feature-vot-IMDB.py','CNN-feature-boost-IMDB.py']

# Open a file to store the results
with open('results-CNN-feature.txt', 'w') as f:
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

