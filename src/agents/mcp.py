"""
Model Context Protocol (MCP)
Implements persistent, context-aware memory for agent interactions.
"""

class ModelContext:
    def __init__(self):
        self.context = {}

    def update_context(self, agent_name, info):
        self.context.setdefault(agent_name, []).append(info)

    def get_context(self, agent_name):
        return self.context.get(agent_name, [])
