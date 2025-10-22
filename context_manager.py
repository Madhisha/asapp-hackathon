# context_manager.py
class ContextManager:
    """Keep last n turns for multi-turn conversation."""
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.history = []

    def add_turn(self, user_text, bot_text):
        self.history.append((user_text, bot_text))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        context_str = ""
        for user, bot in self.history:
            context_str += f"User: {user}\nBot: {bot}\n"
        return context_str
