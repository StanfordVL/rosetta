from abc import ABC, abstractmethod


class Task(ABC):
    description = ""
    setup_description=""

    @abstractmethod
    def state_str(self, state, precision=2):
        pass

    @abstractmethod
    def action_str(self, state, precision=2):
        pass
