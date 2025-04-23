import subprocess

# Path to the widgets.py script
script_path = "Web_Scrap/widgets.py"
input_argument = "amazon womens hoddie"  # The input that determines domain and product

try:
    # Run the widgets.py script with the input argument
    subprocess.run(["python", script_path, input_argument], check=True)
    print(f"Successfully ran widgets.py with input: {input_argument}")
except subprocess.CalledProcessError as e:
    print(f"Error running widgets.py: {e}")