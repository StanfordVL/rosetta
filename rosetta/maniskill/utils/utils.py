import inspect
import os
import time
import os
import time
import uuid
import inspect
import ast
import astor
import json
def get_avail_save_path(save_path,default_name="0"):
    date_time = time.strftime("%Y%m%d%H%M%S")
    unique_id = str(uuid.uuid4())  # Generate a unique identifier
    # use simpler unique_id
    unique_id=unique_id.split("-")[0]
    index=0
    while os.path.exists(os.path.join(save_path, f"{date_time}_{unique_id}_{index}")):
        index+=1
    available_save_path = os.path.join(save_path, f"{date_time}_{unique_id}_{index}_{default_name}")
    
    # Create the directory if it doesn't exist
    os.makedirs(available_save_path, exist_ok=True)
    
    return available_save_path

def get_task_env_source(env, base_class):
    while not isinstance(env, base_class):
        if hasattr(env, 'env'):
            env = env.env
        else:
            source=f"Reached an object of type {type(env)} which does not have an 'env' attribute."
            return source
    source = inspect.getsource(env.__class__)
    return source

def get_modified_source(source_code, reward_json):
    """Get modified source code by replacing functions from reward_json.
    
    Args:
        source_code (str): Original source code of the environment class
        reward_json (dict): Dictionary containing the new function implementations
        
    Returns:
        str: Modified source code with replaced functions
    """
    # Remove environment ID if present
    reward_dict = reward_json.copy()
    if "Environment ID" in reward_dict:
        del reward_dict["Environment ID"]
    
    # Split source into lines
    source_lines = source_code.split('\n')
    final_source = []
    skip_old_function = False
    
    for line in source_lines:
        # Check if line starts a function that needs to be replaced
        if any(line.strip().startswith(f"def {func_name}") for func_name in reward_dict.keys()):
            skip_old_function = True
            # Get the function name
            func_name = next(name for name in reward_dict.keys() 
                           if line.strip().startswith(f"def {name}"))
            # Add the new implementation with proper indentation
            new_func_lines = reward_dict[func_name].strip().split('\n')
            # Add proper indentation
            indented_source = '\n'.join('    ' + line for line in new_func_lines)
            final_source.append(indented_source)
            final_source.append('')  # Add blank line after function
            continue
            
        if skip_old_function:
            if line.strip() and not line.startswith(' '):  # Check if we're out of the function definition
                skip_old_function = False
            else:
                continue
                
        final_source.append(line)
    
    return '\n'.join(final_source)

def get_updated_class_source(cls, function_dict):
    # Step 1: Get the original class source code
    original_source = inspect.getsource(cls)
    
    # Step 2: Parse the original source code
    parsed_source = ast.parse(original_source)
    
    # Step 3: Find the class definition
    class_def = None
    for node in parsed_source.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            class_def = node
            break

    if class_def is None:
        raise ValueError(f"Class definition for {cls.__name__} not found in the source code.")

    # Step 4: Remove methods that are being replaced
    methods_to_replace = function_dict.keys()
    new_body = []
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name in methods_to_replace:
            continue  # Skip methods to be replaced
        new_body.append(node)
    class_def.body = new_body

    # Step 5: Add new methods
    for func_name, func_content in function_dict.items():
        if isinstance(func_content, str):
            # Parse the function code
            func_ast = ast.parse(func_content).body[0]
        elif callable(func_content):
            # Get the source code of the function
            try:
                func_source = inspect.getsource(func_content)
                func_ast = ast.parse(func_source).body[0]
            except (TypeError, OSError):
                # Cannot get source code
                continue
        else:
            continue  # Unsupported type

        class_def.body.append(func_ast)

    # Step 6: Convert the AST back to source code
    updated_source = astor.to_source(parsed_source)
    return updated_source


def load_default_config(config_path: str) -> dict:
    """
    Load default configuration from a JSON file.
    
    Args:
        config_path (str): Path to the default config JSON file
        
    Returns:
        dict: Configuration parameters from the JSON file
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Default config file not found at: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in default config file: {config_path}")

def update_config_from_dict(args, config_dict: dict):
    """
    Update configuration parameters from a dictionary.
    
    Args:
        args (MAPLEConfig): Original configuration object
        config_dict (dict): Dictionary containing new parameter values
        
    Returns:
        MAPLEConfig: Updated configuration object
    """
    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Parameter '{key}' in default config not found in Config")
    return args