from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, TypeVar, Type
import yaml
import logging
from typing import get_type_hints

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base configuration class that implements CLI > YAML > defaults precedence."""
    config_yaml: Optional[Path] = None
    
    def load_yaml_config(self) -> Dict:
        if self.config_yaml is None:
            print("No config_yaml specified")
            return {}
            
        yaml_path = Path(self.config_yaml)
        if not yaml_path.exists():
            print(f"Config path {yaml_path} does not exist")
            return {}
            
        print(f"Loading config from {yaml_path}")
        with open(yaml_path, 'r') as f:
            config=yaml.safe_load(f)
        
        for key, value in config.items():
            if hasattr(self, key):
                print(f"Setting {key}={value} from YAML")
                setattr(self, key, value)
    

def generate_hash_uid(input_string: str, length: int = 16) -> str:
    """
    Generate a UID by hashing an input string to a specified length.
    
    Args:
        input_string (str): The string to hash
        length (int): Desired length of the output hash (default: 16)
    
    Returns:
        str: A hexadecimal hash string of the specified length
    """
    import hashlib
    
    # Generate SHA-256 hash of the input string
    hash_obj = hashlib.sha256(input_string.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Return the first 'length' characters of the hash
    return hash_hex[:length]

def gen_uid_by_timestamp(length:int=16)->str:
    """
    Generate a UID by hashing the current timestamp.
    
    Args:
        length (int): Desired length of the output hash (default: 16)
    
    Returns:
        str: A hexadecimal hash string of the specified length
    """
    import time
    return generate_hash_uid(str(time.time()), length)