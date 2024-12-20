import json
import tensorflow as tf  
import os


def get_model_file(directory):
    # Search for the .tflite or .onnx file in the directory
    for file in os.listdir(directory):
        print(file)
        if file.endswith(".tflite") or file.endswith(".onnx"):
            return os.path.join(directory, file)
    raise FileNotFoundError("No .tflite or .onnx model file found in the directory.")

def load_model_output_shapes(model_file):
    print(f"Loading model from: {model_file}")
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    
    # Get output details (shapes of all output tensors)
    output_details = interpreter.get_output_details()
    print(f"Model output details: {output_details}")
    
    # Collect all output shapes
    output_shapes = [tuple(detail['shape']) for detail in output_details]
    print(f"Model output shapes: {output_shapes}")
    
    return output_shapes

def load_json_shapes(json_file):
    print(f"Loading JSON data from: {json_file}")
    # Load the input/output shapes stored in a JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"JSON data loaded: {data}")
    
    return data

def check_output_shape(model_file, json_file):
    print("Checking output shapes...")
    # Get the model output shapes
    model_output_shapes = load_model_output_shapes(model_file)
    
    # Load the JSON data (input/output shapes)
    json_data = load_json_shapes(json_file)
    
    # Check if any of the model's output shapes match any entry in the JSON data
    for entry in json_data['shapes']:
        for json_output_shape in entry['output_shapes']:
            print(f"Comparing model shape {model_output_shapes} with JSON shape {json_output_shape}")
            if tuple(json_output_shape) in model_output_shapes:
                print("Match found!")
                return True
    
    print("No matching shapes found.")
    return False

# Main script
json_file = 'shapes_data.json'  # Update this with your actual JSON file path

def checker(path):
    model_file = path  # Directly use the specified file path
    print(f"Checking model file: {model_file}")
    try:
        result = check_output_shape(model_file, json_file)
        print(f"Result: {result}")  # Prints True or False
        return result
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False

checker('deeplabv3_257_mv_gpu.tflite')