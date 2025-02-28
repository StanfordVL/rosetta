import inspect
import json
from openai import OpenAI
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

from rosetta_prompts.prompt_designs import DESIGN_NAME_TO_FUNC
from rosetta_prompts.utils import prep_env_code, replace_methods, setup_backup_files, make_readable_output, make_readable_funcs


def generate_reward(
    client: OpenAI,
    exp_id: str,
    human_input: str,
    env_id: str,
    act_space: str,
    task_description: str,
    demo_dir: str,
    prev_funcs: Optional[Dict[str, str]] = None,
    prompt_design: str = "rosetta_lh",
    simulator: str = "maniskill",
    temperature: float = 0.2,
    max_completion_tokens: int = 4096,
    top_p: float = 1.0,
    frequency_penalty: float = 1.0,
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, str], int]:
    """
    Generate reward functions based on human feedback using LLM.

    Args:
        client (OpenAI): Initialized OpenAI client for API calls
        exp_id (str): experiment ID for parent of these generations
        human_input (str): Natural language feedback describing desired reward behavior
        env_id (str): Identifier for the environment
        act_space (str): Action space type. Options:
            - "contcontrol": Continuous control space
            - "actprim": Action primitive space
        task_description (str): Description of goal in existing demo, i.e. prior to human_input.
        demo_dir (str): Path to existing demo's data.
        prev_funcs (Optional[Dict[str, str]]): Previous reward functions to build upon
        prompt_design (str): Strategy for prompting the LLM. Default matches continuous control
        simulator (str): Simulation environment (currently only supports "maniskill")
        temperature (float): Controls randomness in LLM output. Lower values make output more 
            deterministic (range: 0.0-2.0)
        max_completion_tokens (int): Maximum length of LLM response
        top_p (float): Nucleus sampling parameter. Lower values make output more focused 
            (range: 0.0-1.0)
        frequency_penalty (float): Penalizes token frequency to reduce repetition
            (range: 0.0-2.0)
        **kwargs: Additional arguments passed to the prompt execution function

    Returns:
        Tuple containing:
        - message_history: List of message exchanges with LLM
        - generated_functions: Dictionary mapping function names to implementations
        - num_stages: Number of generation stages completed

    Raises:
        ValueError: If incompatible prompt_design and act_space combinations are used
    """
    # Set up backup files
    hist_fp, debug_fp, func_fp, interm_func_fp = setup_backup_files(exp_id)
    hist_f = open(hist_fp, "w") 
    debug_f = open(debug_fp, "w") 

    # Set up LLM parameters
    params = {
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty
    }
    
    # Get prompt execution function and content version
    prompt_exec, content_version = DESIGN_NAME_TO_FUNC[prompt_design]

    # Initialize histories
    message_history = []
    debug_history = []

    # Generate reward functions
    grounding_comps, final_funcs, all_funcs, num_stages = prompt_exec(
        human_input=human_input,
        env_id=env_id,
        prev_funcs=prev_funcs,
        act_space=act_space,
        task_description=task_description,
        demo_dir=demo_dir,
        client=client,
        content_version=content_version,
        hist=message_history,
        hist_f=hist_f,
        debug_hist=debug_history,
        debug_f=debug_f,
        params=params,
        **kwargs
    )

    # Final close and backup
    if hist_f is not None: 
        hist_f.close() 
        make_readable_output(hist_fp)
    if debug_f is not None: 
        debug_f.close() 
        make_readable_output(debug_fp)
    with open(func_fp, "w") as func_f:
        json.dump(final_funcs, func_f, indent=4)
    make_readable_funcs(func_fp)
    interm_func_path = Path(interm_func_fp)
    for i in range(len(all_funcs)):
        interm_func_fn = interm_func_path.parent / (interm_func_path.stem + "v" + str(i) + interm_func_path.suffix)
        with open(interm_func_fn, "w") as interm_func_f:
            json.dump(all_funcs[i], interm_func_f, indent=4)
        make_readable_funcs(interm_func_fn)

    return {
        "history": message_history,
        "grounding": grounding_comps,
        "functions": all_funcs,
        "stages": num_stages,
        "backup_folder": str(hist_fp.parent)
    }