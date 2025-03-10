from collections import deque, defaultdict


class ContextManager:
    def __init__(self, max_memory=10):
        self.memory = defaultdict(lambda: deque(maxlen=max_memory))

    def __getitem__(self, key):
        return self.memory[key]

    def add_to_memory(self, user_input, slot, model_response):
        self.memory[slot].append({"user": user_input, "assistant": model_response})

    def get_context(self, slot):
        return "\n".join(
            [
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in self.memory[slot]
            ]
        )
