class TensorConfig:
    """Class to mimic GstTensorsConfig in Python."""
    def __init__(self, num_tensors, tensor_info):
        self.num_tensors = num_tensors
        self.tensor_info = tensor_info  # List of dicts, each representing a tensor's properties


def check_tensors(config, limit):
    """Check if the tensor configuration meets the specified criteria."""
    if config is None or config.num_tensors < limit:
        print("Configuration is invalid or insufficient tensors.")
        return False

    if config.num_tensors > limit:
        print(
            f"Warning: BoundingBox accepts {limit} or fewer tensors. "
            f"Supplied: {config.num_tensors}. Bandwidth might be wasted."
        )

    # Check if all tensor types are the same
    first_tensor_type = config.tensor_info[0]['type']
    for i in range(1, config.num_tensors):
        if config.tensor_info[i]['type'] != first_tensor_type:
            print("Tensor types are inconsistent.")
            return False

    return True


def check_label_props(label_path, labels, total_labels):
    """Check if label properties are valid."""
    if not label_path or not labels or total_labels <= 0:
        print("Label properties are invalid.")
        return False
    return True


def check_sanity(mode, config, max_labels=None):
    """Sanity checks based on the mode."""
    if mode == "TFLITE_DEEPLAB":
        return (
            config.tensor_info[0]['type'] == "FLOAT32" and
            config.tensor_info[0]['dimension'][0] == max_labels + 1
        )
    elif mode == "SNPE_DEEPLAB":
        return config.tensor_info[0]['type'] == "FLOAT32"
    elif mode == "SNPE_DEPTH":
        return (
            config.tensor_info[0]['type'] == "FLOAT32" and
            config.tensor_info[0]['dimension'][0] == 1
        )
    else:
        print("Invalid mode.")
        return False


def _check_tensors(config):
    """Ensure all tensors in the configuration have the same type."""
    if config is None or not config.tensor_info:
        print("Configuration is invalid.")
        return False

    first_tensor_type = config.tensor_info[0]['type']
    for tensor in config.tensor_info:
        if tensor['type'] != first_tensor_type:
            print("Tensor types are inconsistent.")
            return False

    return True


# Example usage
if __name__ == "__main__":
    # Example tensor configuration
    config = TensorConfig(
        num_tensors=3,
        tensor_info=[
            {"type": "FLOAT32", "dimension": [4]},
            {"type": "FLOAT32", "dimension": [4]},
            {"type": "FLOAT32", "dimension": [4]},
        ],
    )

    # Perform checks
    print("Check Tensors:", check_tensors(config, limit=4))
    print("Check Label Props:", check_label_props("path/to/labels.txt", ["label1"], 5))
    print("Check Sanity (TFLITE_DEEPLAB):", check_sanity("TFLITE_DEEPLAB", config, max_labels=3))
    print("Check All Tensor Types:", _check_tensors(config))
