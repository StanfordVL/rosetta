import copy
from string import Template


class PromptMessage(object):
    
    def __init__(self, role="assistant", content=""):
        """
        Iniitalizes the PromptBlock with a core prompt text

        Args:  
            content (str): The base text of the prompt that can contain `format`able placeholders for dynamic content.
            role (str): The role of the message (e.g., 'system', 'user', 'assistant').
        """
        self.content = content
        self.role = role 
        self.message = {
            "content": content,
            "role": self.role
        }

    def fill_dynamic_fields(self, dynamic_parts):
        """
        Prepares the full prompt by incorporating multiple dynamic contents into the core text.

        Args:
            dynamic_parts (dict): A dictionary containing dynamic contents to insert into the core text.

        Returns:
            dict: A dictionary representing the complete message with 'content' and 'role'.
        """
        self.content = Template(self.content).substitute(dynamic_parts)
        # self.content = Template(self.content).safe_substitute(dynamic_parts)      # TODO uncomment when done debugging to get error-safe code even when fields are missed
        self.message["content"] = self.content
    
    def to_dict(self):
        return copy.deepcopy(self.message)