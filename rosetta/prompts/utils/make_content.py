'''
Utils for making FM user messages
'''
import ast 
import astor
import inspect
import os
import re

from rosetta.prompts.prompt_message import PromptMessage
from rosetta.prompts.utils.constants import *


def get_prompt_content(prompt_name):
    with open(os.path.join(PROMPT_DIR, prompt_name) + ".txt") as f:
        prompt = f.read() 
    return prompt


def prep_env_code(raw_env_code, simulator="maniskill", act_space="actprim", use_prior_reward=False): 
    """
    Remove unwanted functions from ManiSkill environment code
    """
    if simulator == "maniskill":
        return prep_maniskill_env_code(raw_env_code, act_space, use_prior_reward)
    else:
        raise NotImplementedError(f"Unsupported simulator {simulator} and action space {act_space}")


def prep_maniskill_env_code(raw_env_code, act_space, use_prior_reward):
    if act_space == "actprim":
        remove_funcs = MANISKILL_ACTPRIM_REMOVAL_FUNCTIONS
    elif act_space == "contcontrol":
        remove_funcs = MANISKILL_CONTCONTROL_REMOVAL_FUNCTIONS
    else:
        raise ValueError(f"Invalid act space {act_space}")
    if not use_prior_reward:
        remove_funcs.union(set(FUNCS_TO_OVERWRITE[act_space]))
    
    class CustomTransformer(ast.NodeTransformer):
        def visit_Import(self, node):
            return None  # Remove all import statements

        def visit_ImportFrom(self, node):
            return None  # Remove all 'from module import ...' statements

        def visit_ClassDef(self, node):
            # Check and remove specific decorator
            node.decorator_list = []
            # Traverse the methods in the class and remove specific ones
            node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name in remove_funcs)]
            return node

    tree = ast.parse(raw_env_code)
    transformer = CustomTransformer()
    new_tree = transformer.visit(tree)
    return astor.to_source(new_tree)


def get_method_code(cls):
    """
    Get the source code of all methods defined in the given class.
    
    Args:
        cls: The class object whose methods you want to inspect
        
    Returns:
        dict: A dictionary mapping method names to their source code
    """
    method_codes = {}
    
    # Get all methods defined in the class
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip methods that are inherited from parent classes
        if method.__qualname__.startswith(cls.__name__):
            try:
                # Get the source code of the method
                source = inspect.getsource(method)
                method_codes[name] = source
            except (IOError, TypeError):
                method_codes[name] = "Could not retrieve source code"
    
    return method_codes


def override_code_message(role, func_dict):
    new_code_text = "\n\n".join([f"## {name}\n\n```python\n{function}\n```" for name, function in func_dict.items()])
    content = f"Here's new code that integrates the preference into the reward calculation:\n\n{new_code_text}\n\nThis should help the robot incorporate the preference."
    return PromptMessage(role=role, content=content)


def replace_methods(class_code, method_replacements):
    lines = class_code.split('\n')
    new_lines = []
    skip_mode = False
    method_indent = ''
    method_indent_level = 0
    i = 0
    n = len(lines)

    # Assuming top-level method indentation is a known constant:
    TOP_LEVEL_INDENT_LEVEL = 4  # Adjust if needed

    while i < n:
        line = lines[i]

        if not skip_mode:
            # Look for a potential top-level method definition
            method_def_match = re.match(r'^(\s*)def\s+(\w+)\s*\(.*\):', line)
            if method_def_match:
                method_indent = method_def_match.group(1)
                method_indent_level = len(method_indent)
                method_name = method_def_match.group(2)

                # Only trigger replacement if this is a top-level method
                if (method_indent_level in [TOP_LEVEL_INDENT_LEVEL, 0]) and (method_name in method_replacements):
                    # Begin skip mode and replace the method
                    skip_mode = True
                    new_method_code = method_replacements[method_name]
                    new_method_lines = new_method_code.split('\n')

                    # Insert the replacement method code with the same indentation
                    for new_line in new_method_lines:
                        new_lines.append(method_indent + new_line)
                    i += 1
                    continue
                else:
                    # Not replacing this method (either it's not top-level or not in replacements)
                    new_lines.append(line)
                    i += 1
            else:
                # Normal line, just copy it
                new_lines.append(line)
                i += 1
        else:
            # We're currently skipping lines from an old method definition
            next_line = lines[i]
            next_line_indent_match = re.match(r'^(\s*)', next_line)
            next_line_indent = next_line_indent_match.group(1)
            next_line_indent_level = len(next_line_indent)

            # We continue to skip as long as we're within or deeper than the method block
            # Ending conditions:
            # - Encounter a line that is dedented (less or equal indentation) and not blank
            # If blank lines or docstrings (which might be at same indentation) appear, just skip them.
            if next_line_indent_level <= method_indent_level and next_line.strip():
                # Found a dedented, non-blank line that signifies the end of the method block
                skip_mode = False
                # Do NOT increment i here, so we re-evaluate this line in non-skip mode
                continue
            else:
                # Still inside the method (or encountering blank line/docstring at method level)
                i += 1
                continue

    return '\n'.join(new_lines)