'''
Utils for making backups and readable markdowns of all generations.
'''
import json
import os
from pathlib import Path
import re

from .constants import *


def save_hist_to_json(hist, f):
    hist_dicts = [msg if isinstance(msg, dict) else msg.to_dict() for msg in hist]
    f.seek(0)
    f.truncate()
    json.dump(hist_dicts, f, indent=4)
    f.flush()


def default_save_msg_hist(msg, hist, hist_fobj):
    """
    Do default save of message to history:
    - Add message to history 
    - If the history file object is not None, save history to history file object

    Not for use in cases where history needs to be manipulated in a way other 
        than just appending the message. E.g. after an error correction, we typically
        don't want to just append the latest message on. So don't use this in that case.
    """
    hist.append(msg)
    if hist_fobj is not None: 
        save_hist_to_json(hist, hist_fobj)


def make_readable_output(fp):
    with open(fp, "r") as f:
        data = json.load(f)

    path, fn = os.path.split(fp)
    base, __ = os.path.splitext(fn)
    with open(os.path.join(path, f"readable_{base}.md"), "w") as f:
        for index, item in enumerate(data):
            role = item.get("role", "No role specified")
            content = item.get("content", "No content specified")
            f.write(f"## Entry {index}\n")
            f.write(f"**Role:** {role}\n\n")
            f.write(content + "\n\n")


def make_readable_funcs(fp, name="funcs"):
    with open(fp, "r") as f:
        funcs = json.load(f)
    
    fp = Path(fp)
    save_path = fp.parent / f"readable_{fp.stem}.md"
    with open(save_path, "w") as f:
        func_list = sorted(funcs.items(), key=lambda n: n[0])
        for name, func in func_list:
        # for name, func in sorted(funcs.items(), key=lambda n: n[0]):
            f.write(f"## `{name}`\n")
            f.write("```python\n")
            f.write(func)
            f.write("\n")
            f.write("```")
            f.write("\n\n")


def make_json_funcs_from_readable(fp):
    fp = Path(fp)
    with open(fp, "r") as f:
        readable_funcs = f.read()
    functions = {}
    # Split the content on '```python\n'
    parts = readable_funcs.split('```python\n')
    for part in parts[1:]:  # Skip the first part before the first code block
        # Split on '```' to get the code block and discard the rest
        code_block = part.split('```', 1)[0]
        code_block = code_block.strip()
        # Get the first line of the code block (function declaration)
        first_line = code_block.split('\n', 1)[0].strip()
        # Extract the function name from the first line
        match = re.match(r'def\s+([^\s(]+)', first_line)
        if match:
            function_name = match.group(1)
            # Map the function name to the full code block
            functions[function_name] = code_block
        else:
            # If the function name can't be extracted, skip this code block
            functions[None] = code_block
    
    with open(fp.parent / (fp.stem[9:] + ".json"), "w") as f:
        json.dump(functions, f, indent=4)



# backup files

def get_next_id(exp_id, gentype="reward"): 
    outdir = BACKUP_DIR
    os.makedirs(outdir, exist_ok=True)
    existing_exps = os.listdir(outdir)
    exp_name_matches = [d for d in existing_exps if exp_id in d]
    if exp_name_matches: 
        max_i = max(int(d[len(exp_id) + 4:]) for d in exp_name_matches)
    else:
        max_i = -1 
    return max_i + 1 


def setup_backup_files(exp_name):
    outdir = BACKUP_DIR
    gen_id = get_next_id(exp_name)
    exp_name = exp_name + f"_gen{gen_id}"
    os.makedirs(outdir / exp_name, exist_ok=True)

    hist_file = outdir / exp_name / "history.json"
    debug_hist_file = outdir / exp_name / "debug_history.json"
    func_file = outdir / exp_name / "funcs.json"
    interm_func_file = outdir / exp_name / "funcs_.json"
    
    return hist_file, debug_hist_file, func_file, interm_func_file