import tensorflow as tf
import os
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
        
        return output_details
    
    elif model_file.endswith('.onnx'):

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

# def check_tensors(config, limit):
#     """Check if the tensor configuration meets the specified criteria."""
#     if config is None:
#         print("Configuration is invalid.")
#     elif config.num_tensors < limit:
#         print("Insufficient tensors.")
#         return False

#     if config.num_tensors > limit:
#         print(
#             f"Warning: BoundingBox accepts {limit} or fewer tensors. "
#             f"Supplied: {config.num_tensors}. Bandwidth might be wasted."
#         )

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

def run_all_checks(model_file, max_labels=None):
    """Run all tensor checks and return a single True or False."""
    # Load the model output shapes using the model_file
    output_shapes = load_model_output_shapes(model_file)
    
    # Create a TensorConfig object from the output shapes
    config = TensorConfig(
        num_tensors=len(output_shapes),
        tensor_info=[
            {
                "name": shape_info['name'],
                "index": shape_info['index'],
                "type": shape_info['dtype'].__name__.upper() if hasattr(shape_info['dtype'], '__name__') else str(shape_info['dtype']),
                "dimension": shape_info['shape'],
            }
            for shape_info in output_shapes
        ]
    )
    
    # Run checks
    check_type_result = check_type(config)
    
    # Skip sanity check for ONNX models
    if model_file.endswith('.onnx'):
        check_sanity_result = True
    else:
        check_sanity_result = check_sanity(config, max_labels)
    
    
    return check_type_result and check_sanity_result


def checker(directory):
    try:
        model_file = get_model_file(directory)
        result = run_all_checks(model_file, max_labels=None)
        print(f"\nResult: {result}")  # Prints True or False
        return result
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return False

checker('C:/Users/achin/Desktop/customPluginCheck')