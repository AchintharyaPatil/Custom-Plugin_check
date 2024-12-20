import json
import os
import tensorflow as tf
import onnxruntime as ort

class TensorConfig:
    """Class to mimic GstTensorsConfig in Python."""
    def __init__(self, num_tensors, tensor_info):
        self.num_tensors = num_tensors
        self.tensor_info = tensor_info

def get_model_file(directory):
    # Search for the .tflite or .onnx file in the directory
    for file in os.listdir(directory):
        if file.endswith(".tflite") or file.endswith(".onnx"):
            return os.path.join(directory, file)
    raise FileNotFoundError("No .tflite or .onnx model file found in the directory.")

def load_model_output_shapes(model_file):
    print(f"Loading model from: {model_file}")
    if model_file.endswith('.tflite'):
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_file)
        interpreter.allocate_tensors()
        
        # Get output details (shapes of all output tensors)
        output_details = interpreter.get_output_details()
        
        output_shapes = [tuple(detail['shape']) for detail in output_details]
        
        return output_details, output_shapes
    
    elif model_file.endswith('.onnx'):
        # Load the ONNX model
        session = ort.InferenceSession(model_file)
        
        # Get output details (shapes of all output tensors)
        output_details = [
            {
                'name': output.name,
                'index': idx,
                'shape': session.get_outputs()[idx].shape,
                'dtype': session.get_outputs()[idx].type,
            }
            for idx, output in enumerate(session.get_outputs())
        ]
        
        return output_details
    
    else:
        raise ValueError("Unsupported model format. Please provide a .tflite or .onnx file.")

def load_json_shapes(json_file):
    # Load the input/output shapes stored in a JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    
    return data

def check_output_shape(model_file, json_file):
    # Get the model output shapes
    output_details, output_shapes = load_model_output_shapes(model_file)
    
    # Load the JSON data (input/output shapes)
    json_data = load_json_shapes(json_file)
    
    # Check if any of the model's output shapes match any entry in the JSON data
    for entry in json_data['shapes']:
        for json_output_shape in entry['output_shapes']:
            if tuple(json_output_shape) in output_shapes:
                return True
    
    print("No matching shapes found.")
    return False

def check_type(config):
    if not config.tensor_info:
        print("No tensor information available.")
        return False
    first_tensor_type = config.tensor_info[0]['type']
    for tensor_info in config.tensor_info:
        if tensor_info['type'] != first_tensor_type:
            print("Tensor types are inconsistent.")
            return False
    return True

def check_sanity(config, max_labels=None):
    """Try all modes and return True if any mode passes the sanity check."""
    modes = ["TFLITE_DEEPLAB", "SNPE_DEEPLAB", "SNPE_DEPTH"]
    
    for mode in modes:
        if mode == "TFLITE_DEEPLAB":
            if (
                config.tensor_info[0]['type'] == "FLOAT32" and
                max_labels is not None and  # Ensure max_labels is not None
                config.tensor_info[0]['dimension'][0] == max_labels + 1
            ):
                return True
        elif mode == "SNPE_DEEPLAB":
            if config.tensor_info[0]['type'] == "FLOAT32":
                return True
        elif mode == "SNPE_DEPTH":
            if (
                config.tensor_info[0]['type'] == "FLOAT32" and
                config.tensor_info[0]['dimension'][0] == 1
            ):
                return True
    
    print("No valid mode found.")
    return False

def run_all_checks(model_file, json_file, max_labels=None):
    """Run all tensor checks and return a single True or False."""
    # Load the model output shapes using the model_file
    output_details, output_shapes = load_model_output_shapes(model_file)
    
    # Create a TensorConfig object from the output shapes
    config = TensorConfig(
        num_tensors=len(output_details),
        tensor_info=[
            {
                "name": shape_info['name'],
                "index": shape_info['index'],
                "type": shape_info['dtype'].__name__.upper() if hasattr(shape_info['dtype'], '__name__') else str(shape_info['dtype']),
                "dimension": shape_info['shape'],
            }
            for shape_info in output_details
        ]
    )
    
    # Run checks
    check_type_result = check_type(config)
    check_output_shape_result = check_output_shape(model_file, json_file)
    
        # Skip sanity check for ONNX models
    if model_file.endswith('.onnx'):
        check_sanity_result = True
    else:
        check_sanity_result = check_sanity(config, max_labels)
        
    
    
    # Return True if all checks pass, otherwise False
    return check_type_result and check_sanity_result and check_output_shape_result

def checker(directory, json_file):
    try:
        # Use get_model_file to find the model file
        model_file = get_model_file(directory)
        result = run_all_checks(model_file, json_file, max_labels=None)
        print(f"\nResult: {result}")  # Prints True or False
        return result
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return False

# Example usage
checker('C:/Users/achin/Desktop/customPluginCheck', 'shapes_data.json')