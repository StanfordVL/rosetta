'''
Utils for extracting necessary content (plan, code, num stages) from FM assistant messages
    and collecting/updating wherever they are stored.
Includes util for building long-horizon function from generations, which are not the function's
    final form.
'''
import json
import re
import redbaron
import textwrap


def extract_functions_and_names(code):
    # Parse code with RedBaron to preserve formatting, comments, etc.
    try:
        red = redbaron.RedBaron(code)
    except Exception as e:
        red = None

    functions = []

    # Helper to extract function code using redbaron
    def get_function_code_redbaron(func_node):
        return func_node.dumps()

    if red:
        # Traverse through the nodes in RedBaron
        for node in red:
            if node.type == "def":
                # Top level function (directly under the module)
                functions.append((node.name, get_function_code_redbaron(node)))
            elif node.type == "class":
                # Methods within top-level class
                for class_child in node.value:
                    if class_child.type == "def":
                        functions.append((class_child.name, get_function_code_redbaron(class_child)))

    else:
        def process_function(lines, start_index, functions):
            n = len(lines)
            i = start_index
            line = lines[i]
            stripped_line = line.lstrip()
            indent = len(line) - len(stripped_line)
            func_decl_lines = [line]
            open_parens = stripped_line.count('(') - stripped_line.count(')')
            open_brackets = stripped_line.count('[') - stripped_line.count(']')
            open_braces = stripped_line.count('{') - stripped_line.count('}')

            # Collect function declaration lines (handles multi-line declarations)
            while (not stripped_line.endswith(':') or open_parens > 0 or open_brackets > 0 or open_braces > 0) and i + 1 < n:
                i += 1
                line = lines[i]
                stripped_line = line.lstrip()
                func_decl_lines.append(line)
                open_parens += stripped_line.count('(') - stripped_line.count(')')
                open_brackets += stripped_line.count('[') - stripped_line.count(']')
                open_braces += stripped_line.count('{') - stripped_line.count('}')

            # Extract the function name using regex
            func_decl = '\n'.join(func_decl_lines)
            match = re.match(r'\s*def\s+([^\s(]+)', func_decl)
            if match:
                func_name = match.group(1)
            else:
                # Skip if function name cannot be extracted
                return None, None, i + 1

            # Collect function body lines
            i += 1
            func_body_lines = []
            while i < n:
                line = lines[i]
                stripped_line = line.lstrip()
                current_indent = len(line) - len(stripped_line)
                if len(stripped_line) == 0:
                    # Include blank lines
                    func_body_lines.append(line)
                    i += 1
                elif current_indent > indent:
                    # Line is part of the function body
                    if stripped_line.startswith('def '):
                        # Handle nested function
                        nested_func_name, nested_func_code, i = process_function(lines, i, functions)
                        if nested_func_name:
                            functions[nested_func_name] = nested_func_code
                    else:
                        func_body_lines.append(line)
                        i += 1
                else:
                    # Indentation decreased; function body ends
                    break

            # Combine declaration and body
            function_code = '\n'.join(func_decl_lines + func_body_lines)
            return func_name, function_code, i

        functions = []
        lines = code.split('\n')
        n = len(lines)
        i = 0

        while i < n:
            line = lines[i]
            stripped_line = line.lstrip()
            if stripped_line.startswith('def '):
                func_name, func_code, i = process_function(lines, i, functions)
                if func_name:
                    functions.append((func_name, func_code))
            else:
                i += 1

    # Make any hand additions
    updated_functions = []
    for name, func_code in functions:
        updated_functions.append((name, func_code))

    return updated_functions


def extract_functions_from_content(content):
    left_extracteds = content.split("```python")[1:]
    extracteds = []
    for left_extracted in left_extracteds:
        extracted = left_extracted.split("```")[0]
        extracteds.append(extracted)

    functions = {}
    for code in extracteds:
        extracted_functions = extract_functions_and_names(code)
        for name, extracted_function in extracted_functions:
            functions[name] = extracted_function

    for name in functions:
        if name == "skill_reward":
            functions[name] = functions[name] # WHAT??

    return functions


def extract_plan(message):
    content = message.message["content"]
    plan = content.split("# Plan")[-1].split("## End Of Plan")[0]
    return plan


def extract_target_actions(message):
    """
    Extract the 'Stage template' values from a markdown-formatted text,
    focusing specifically on a continuous block of stages.

    Args:
        markdown_text (str): The markdown text containing stage descriptions

    Returns:
        list: A list of stage template values
    """
    content = message.message["content"]
    # content = message["content"]    # TODO comment after debugging

    # Find stage number and stage action lines
    stage_block_pattern = r"(?m)(^### Stage \d+: .*$)\n(^- Stage template: \".*\"$)"
    stage_blocks = re.findall(stage_block_pattern, content)

    if not stage_blocks:
        return []

    action_pattern = r'- Stage template: "(.*?)"'
    action_names = [
        re.search(action_pattern, stage_block[1]).group(1) for stage_block in stage_blocks
    ]
    def action_map(action):
        if action == "pick up": return 0
        elif action == "place": return 1
        elif action == "push": return 2
        else: raise ValueError(f"Invalid action {action}")
    actions = [action_map(action_name) for action_name in action_names]

    return actions


def extract_num_stages(interim_funcs):
    stage_funcs = [
        func_name for func_name in interim_funcs.keys()
        if func_name.startswith('compute_target_position_stage') or
           func_name.startswith('compute_target_pos_reward_stage')
    ]
    stages = sorted(
        set(int(func_name.split('stage')[-1]) for func_name in stage_funcs)
    )
    if stages != list(range(len(stages))):
        raise ValueError("Stage indices must be 0-indexed and consecutive")

    return len(stages)



def update_latest_funcs(message, func_dict):
    content = message.message["content"]
    new_func_dict = extract_functions_from_content(content)
    updated_func_dict = func_dict.copy()
    updated_func_dict.update(new_func_dict)
    return updated_func_dict


def update_latest_interim_funcs_actprim(message, interim_func_dict, target_actions, debug_mode=False):
    if debug_mode:
        content = message["content"]
    else:
        content = message.message["content"]
    new_interim_funcs = extract_functions_from_content(content)
    updated_interim_funcs = interim_func_dict.copy()
    updated_interim_funcs.update(new_interim_funcs)
    check_if_interim_funcs_complete(updated_interim_funcs)
    return updated_interim_funcs


def build_actprim_funcs(interim_funcs, target_actions):
    """
    Builds final-format actprim funcs from intermediate format that came from generation

    :param target_actions {List[int]}: ordered list of target actions for each stage, so that
                                       target_actions[i] contains the target_action for stage i
    :param interim_funcs {Dict[str, str]}: dict of functions that contains some subset of
                                       - evaluate
                                       - compute_target_position_stagei for every stage
                                       - compute_target_pos_reward_stagei for every stage
                                       Newly generated versions.
    :returns {Dict[str, str]}: dict of functions that contains
                               - evaluate
                               - skill_reward
    """
    # Validate stage indices
    stage_funcs = [
        func_name for func_name in interim_funcs.keys()
        if func_name.startswith('compute_target_position_stage') or
           func_name.startswith('compute_target_pos_reward_stage')
    ]
    stages = sorted(
        set(int(func_name.split('stage')[-1]) for func_name in stage_funcs)
    )

    if stages != list(range(len(stages))):
        raise ValueError("Stage indices must be 0-indexed and consecutive")

    # Generate stage reward function
    def generate_stage_reward_func(stage):
        return textwrap.indent(f'''
def stage_{stage}_reward():
    # Target position computation
{textwrap.indent(
    get_func_inner_content(
        interim_funcs[f'compute_target_position_stage{stage}']
), "    ")}
    target_action = {target_actions[stage]}
    if current_selected_action == target_action:
        # Target position reward computation
{textwrap.indent(
    get_func_inner_content(
        interim_funcs[f'compute_target_pos_reward_stage{stage}']
    ), "        "
)}
        # scale reward to positive
        reward_components["afford"] = (1 + reward) * 5.0

    if cur_info["stage{stage}_success"]:
        reward_components["success"] = 10.0

    return reward_components''',
"    ")

    # Generate full reward computation function
    reward_stages = "\n".join(
        # textwrap.indent(generate_stage_reward_func(stage), "    ")
        generate_stage_reward_func(stage)
        for stage in stages
    )

    stage_selection_logic = "\n\n"
    for stage in stages:
        stage_selection_logic += f"    if self.cur_stage == {stage}:\n"
        stage_selection_logic += f"        reward = stage_{stage}_reward()\n"

    stage_selection_logic += "\n\n"

    for stage in stages:
        stage_selection_logic += f"    if (self.cur_stage == {stage}) and cur_info[\"stage{stage}_success\"]:\n"
        stage_selection_logic += f"        self.cur_stage = {stage + 1}\n"



    skill_reward_str = textwrap.dedent(f'''
def skill_reward(self, prev_info, cur_info, action, **kwargs):
    """
    Compute reward based on previous and current stage information.

    Args:
        prev_info (dict): Previous stage information
        cur_info (dict): Current stage information
        action (np.array): Action vector

    Returns:
        dict: Reward components
    """
    reward_components = dict((k, 0.0) for k in self.reward_components)
    current_selected_action = np.argmax(action[:len(self.task_skill_indices.keys())])

    if current_selected_action in [0, 1]:
        current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
        current_selected_pos_2 = None
    else:
        current_selected_pos_1 = action[len(self.task_skill_indices.keys()):len(self.task_skill_indices.keys())+3]
        current_selected_pos_2 = action[len(self.task_skill_indices.keys())+3:len(self.task_skill_indices.keys())+6]

{reward_stages}

    # Stage selection logic
{stage_selection_logic}

    return reward
''')

    eval_func_str = \
"""def evaluate(self):
    info = self._get_obs_info()
"""
    for stage in stages:
        eval_func_str += textwrap.indent(interim_funcs[f'stage{stage}_success'], "    ") + "\n\n"


    for stage in stages:
        eval_func_str += f"    info[\"stage{stage}_success\"] = stage{stage}_success(info)" + "\n"

    eval_func_str += \
f"""
    info["success"] = torch.tensor(False)
    if self.cur_stage=={stages[-1]}:
        info["success"] = torch.tensor(info["stage{stages[-1]}_success"])

    return info
"""

    return {
        "skill_reward": skill_reward_str,
        "evaluate": eval_func_str
    }


def get_func_inner_content(func_str):
    lines = func_str.splitlines()

    # Remove the function definition lines (handles multi-line 'def' statements)
    def_lines_end = 0
    in_def = False
    paren_balance = 0
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('def'):
            in_def = True
        if in_def:
            paren_balance += line.count('(') - line.count(')')
            if stripped_line.endswith(':') and paren_balance == 0:
                def_lines_end = i + 1  # Move past the function header
                break
    else:
        # No function definition found
        return ''

    # Remove the function header lines
    lines = lines[def_lines_end:]

    # Remove the docstring if present
    def is_docstring_start(line):
        stripped = line.strip()
        return (stripped.startswith('"""') or stripped.startswith("'''"))

    def remove_docstring(lines):
        if not lines:
            return lines
        if is_docstring_start(lines[0]):
            quote = lines[0].strip()[:3]
            # Single-line docstring
            if lines[0].strip().endswith(quote) and len(lines[0].strip()) > 6:
                return lines[1:]
            # Multi-line docstring
            for i in range(1, len(lines)):
                if lines[i].strip().endswith(quote):
                    return lines[i+1:]
            return []  # Docstring not properly closed
        return lines

    lines = remove_docstring(lines)

    # Remove any lines that are return statements
    def remove_return_statements(lines):
        return [line for line in lines if not line.lstrip().startswith('return')]

    lines = remove_return_statements(lines)

    # Dedent the remaining lines
    def dedent_lines(lines):
        import sys
        min_indent = sys.maxsize
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:
                leading_spaces = len(line) - len(line.lstrip())
                min_indent = min(min_indent, leading_spaces)
        if min_indent == sys.maxsize:
            min_indent = 0
        return [line[min_indent:] for line in lines]

    dedented_lines = dedent_lines(lines)

    # Join the lines back into a single string
    return '\n'.join(dedented_lines)


def check_if_interim_funcs_complete(interim_funcs_dict):

    stages = sorted(
        set(int(func_name.split('stage')[-1].split("_")[0]) for func_name in interim_funcs_dict if func_name not in ["evaluate", "skill_reward"])
    )
    print("Stages:", stages)
    print("Interim funcs dict keys")
    from pprint import pprint
    pprint(list(interim_funcs_dict.keys()))
    assert stages == list(range(len(stages))), "stage indices must be 0-indexed and consecutive"

    for stage_i in stages:
        assert f"stage{stage_i}_success" in interim_funcs_dict, f"Stage {stage_i} doesn't have stage{stage_i}_success"
        assert f"compute_target_position_stage{stage_i}" in interim_funcs_dict, f"Stage {stage_i} doesn't have compute_target_position_stage{stage_i}"
        assert f"compute_target_pos_reward_stage{stage_i}" in interim_funcs_dict, f"Stage {stage_i} doesn't have compute_target_pos_reward_stage{stage_i}"


def get_intermediate_funcs_from_hist(hist, end_msg_ind, save_fn):
    '''
    Get the funcs finalized at an intermediate point in a history and save to specified filename
        in same directory as original history
    '''
    func_dict = {}
    for msg_ind in range(end_msg_ind + 1):
        msg = hist[msg_ind]
        if msg["role"] != "assistant": continue
        new_func_dict = extract_functions_from_content(msg["content"])
        func_dict.update(new_func_dict)
    with open(save_fn, "w") as f:
        json.dump(func_dict, f, indent=4)