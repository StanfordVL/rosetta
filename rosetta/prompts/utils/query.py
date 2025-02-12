import openai 
from feedback_to_reward_prompts.prompt_message import PromptMessage


def query_until_complete(
    client, 
    history, 
    model,
    params,
):
    """
    This function queries the OpenAI GPT-4 API repeatedly until the full response is retrieved, 
    handling responses that exceed the token limit by requesting continuation tokens.

    :param client: The client.
    :param history: The input message history.
    :param model: The model to be used for the API call.
    :param max_completion_tokens: The maximum number of tokens to be generated in each API call.
    :param temperature: Controls the randomness of the output.
    :return: Message history with new generation.
    """
    messages = [prompt_message.message for prompt_message in history]

    if model in ["o1-preview", "o1-mini"]:
        full_response = ""
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        full_response += response.choices[0].message.content
        messages.append(response.choices[0].message)

        while response.choices[0].finish_reason == "length":
            response = openai.Completion.create(
                model=model,
                messages=messages,
            )
            messages[-1]["content"] += response.choices[0].message.content
    elif "4o" in model:
        response_format = params.get("response_format", None)
        full_response = ""
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params["temperature"],
            response_format={"type": response_format if response_format is not None else "text"}
        )
        full_response += response.choices[0].message.content
        messages.append(response.choices[0].message)

        while response.choices[0].finish_reason == "length":
            response = openai.Completion.create(
                model=model,
                messages=messages,
                temperature=params["temperature"],
                response_format={"type": response_format if response_format is not None else "text"}
            )
            messages[-1]["content"] += response.choices[0].message.content
    else:   # Can be enabled once max_completion_tokens and temperature are both available. Will need to adapt for json_output.
        full_response = ""
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=params["max_completion_tokens"],
            temperature=params["temperature"]
        )
        full_response += response.choices[0].message.content
        messages.append(response.choices[0].message)

        while response.choices[0].finish_reason == "length":
            response = openai.Completion.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=params["temperature"],
                stop=None  # You can customize this if you have specific endpoint tokens.
            )
            messages[-1]["content"] += response.choices[0].message.content

    return PromptMessage(
        role=messages[-1].role,
        content=messages[-1].content
    )

