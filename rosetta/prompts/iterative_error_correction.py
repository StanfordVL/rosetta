from feedback_to_reward.maniskill.pipelines.error_code_long_horizon import direct_evaluation_run_code as error_test_longhorizon
from feedback_to_reward.maniskill.pipelines.error_code_short_horizon import direct_evaluation_run_code as error_test_shorthorizon

from feedback_to_reward_prompts.prompt_message import PromptMessage
from feedback_to_reward_prompts.utils import *


def get_error_data(
    func_dict,
    env_id,
    functions_to_overwrite,
    act_space,
    stages=None,
    target_actions=None
):
    if act_space == "actprim":
        res = error_test_longhorizon(
            env_id=env_id,
            func_dict=func_dict,
            functions_to_overwrite=functions_to_overwrite,
            stages=stages,
            target_actions=target_actions
        )
    elif act_space == "contcontrol":
        res = error_test_shorthorizon(
            env_id=env_id,
            func_dict=func_dict,
            functions_to_overwrite=functions_to_overwrite
        )
    return res


def get_error_line(
    error_trace,
    func_dict
):

    line_info_pattern = re.compile(r'File ".*?", line (\d+), in (\w+)')
    matches = line_info_pattern.findall(error_trace)
    print("error line matches:")
    print(matches)
    print()
    error_line_num, error_func_name = None, None

    for line_num, func_name in reversed(matches):
        if func_name in func_dict:
            error_line_num = int(line_num)
            error_func_name = func_name
            break

    error_code = func_dict[error_func_name]

    class CodeVisitor(ast.NodeVisitor):
        def __init__(self):
            super().__init__()
            self.relevant_nodes = []

        def visit(self, node):
            if hasattr(node, "lineno"):
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", start_line)

                if start_line <= error_line_num <= end_line:
                    self.relevant_nodes.append(node)

            super().visit(node)


    def extract_code_from_node(node, lines):
        start_line = node.lineno - 1        # Convert 1-indexed to 0-indexed
        end_line = getattr(node, "end_lineno", node.lineno)
        return "".join(lines[start_line:end_line])

    tree = ast.parse(error_code)
    visitor = CodeVisitor()
    visitor.visit(tree)

    relevant_node = min(visitor.relevant_nodes, key=lambda n: (n.end_lineno - n.lineno), default=None)

    if relevant_node:
        lines = error_code.splitlines(keepends=True)
        return extract_code_from_node(relevant_node, lines)
    else:
        return None


def o1mini_error_loop(
    client,
    params,
    num_tries,
    latest_funcs,
    env_id,
    funcs_to_overwrite,
    act_space,
    env_code,
    hist,
    debug_hist,
    debug_f,
    stages=None,
    target_actions=None
):
    for __ in range(num_tries):
        print()
        print("Starting error test in loop")
        print()
        error_data = get_error_data(latest_funcs, env_id, funcs_to_overwrite, act_space, stages=stages, target_actions=target_actions)

        if error_data["success"]:
            print("Success")
            break

        # If necessary, create error correction user message
        print()
        print("error_message")
        print(error_data["error_message"])
        print()
        error_line = get_error_line(error_data["error_message"], latest_funcs)

        env_code = replace_methods(env_code, latest_funcs)

        user_errorcorr_msg = PromptMessage(role="user", content=get_prompt_content(f"errorcorr/{act_space}/o1mini"))
        if act_space == "contcontrol":
            documentation = get_prompt_content(f"documentation/{act_space}")
            user_errorcorr_msg.fill_dynamic_fields({
                "error_trace": error_data["error_message"],
                "error_line": error_line,
                "generated_env_code": env_code,
                "documentation": documentation
            })
        elif act_space == "actprim":
            user_errorcorr_msg.fill_dynamic_fields({
                "error_trace": error_data["error_message"],
                "error_line": error_line,
                "generated_env_code": env_code
            })

        hist.append(user_errorcorr_msg)
        default_save_msg_hist(user_errorcorr_msg, debug_hist, debug_f)

        # build history with just user message for o1mini
        error_hist = [user_errorcorr_msg]

        # run api on error_hist (just user message)
        asst_errcorr_msg = query_until_complete(client, error_hist, "o1-mini", params)
        default_save_msg_hist(asst_errcorr_msg, debug_hist, debug_f)

        latest_funcs = update_latest_funcs(asst_errcorr_msg, latest_funcs)

    else:
        latest_funcs = {}

    return latest_funcs


def o1mini_error_loop_eureka(
    client,
    params,
    num_tries,
    latest_funcs,
    env_id,
    funcs_to_overwrite,
    act_space,
    env_code,
    hist,
    debug_hist,
    debug_f,
    stages=None,
    target_actions=None
):
    for __ in range(num_tries):
        print()
        print("Starting error test in loop")
        print()
        error_data = get_error_data(latest_funcs, env_id, funcs_to_overwrite, act_space, stages=stages, target_actions=target_actions)

        if error_data["success"]:
            print("Success")
            break

        # If necessary, create error correction user message
        print()
        print("error_message")
        print(error_data["error_message"])
        print()
        error_line = get_error_line(error_data["error_message"], latest_funcs)

        env_code = replace_methods(env_code, latest_funcs)
        documentation = get_prompt_content(f"documentation/{act_space}")

        user_errorcorr_msg = PromptMessage(role="user", content=get_prompt_content(f"errorcorr/{act_space}/o1minieureka"))
        user_errorcorr_msg.fill_dynamic_fields({
            "error_trace": error_data["error_message"],
            "error_line": error_line,
            "generated_env_code": env_code,
            "documentation": documentation
        })
        hist.append(user_errorcorr_msg)
        default_save_msg_hist(user_errorcorr_msg, debug_hist, debug_f)

        # build history with just user message for o1mini
        error_hist = [user_errorcorr_msg]

        # run api on error_hist (just user message)
        asst_errcorr_msg = query_until_complete(client, error_hist, "o1-mini", params)
        default_save_msg_hist(asst_errcorr_msg, debug_hist, debug_f)

        latest_funcs = update_latest_funcs(asst_errcorr_msg, latest_funcs)

    else:
        latest_funcs = {}

    return latest_funcs


def o1mini_error_loop_ro(
    client,
    params,
    num_tries,
    latest_funcs,
    env_id,
    funcs_to_overwrite,
    act_space,
    env_code,
    hist,
    debug_hist,
    debug_f,
    stages=None,
    target_actions=None
):
    for __ in range(num_tries):
        print()
        print("Starting error test in loop")
        print()
        error_data = get_error_data(latest_funcs, env_id, funcs_to_overwrite, act_space, stages=stages, target_actions=target_actions)

        if error_data["success"]:
            print("Success")
            break

        # If necessary, create error correction user message
        print()
        print("error_message")
        print(error_data["error_message"])
        print()
        error_line = get_error_line(error_data["error_message"], latest_funcs)

        env_code = replace_methods(env_code, latest_funcs)
        documentation = get_prompt_content(f"documentation/{act_space}")

        user_errorcorr_msg = PromptMessage(role="user", content=get_prompt_content(f"errorcorr/{act_space}/o1miniro"))
        user_errorcorr_msg.fill_dynamic_fields({
            "error_trace": error_data["error_message"],
            "error_line": error_line,
            "generated_env_code": env_code,
            "documentation": documentation
        })
        hist.append(user_errorcorr_msg)
        default_save_msg_hist(user_errorcorr_msg, debug_hist, debug_f)

        # build history with just user message for o1mini
        error_hist = [user_errorcorr_msg]

        # run api on error_hist (just user message)
        asst_errcorr_msg = query_until_complete(client, error_hist, "o1-mini", params)
        default_save_msg_hist(asst_errcorr_msg, debug_hist, debug_f)

        latest_funcs = update_latest_funcs(asst_errcorr_msg, latest_funcs)

    else:
        latest_funcs = {}

    return latest_funcs
