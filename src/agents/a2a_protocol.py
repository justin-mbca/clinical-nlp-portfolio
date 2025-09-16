"""
Agent-to-Agent Protocols (A2A)
Defines base classes and message passing for agent collaboration.
"""

class Agent:
    def __init__(self, name):
        self.name = name
        self.memory = []

    def send_message(self, recipient, message):
        recipient.receive_message(self, message)

    def receive_message(self, sender, message):
        self.memory.append((sender.name, message))
        print(f"{self.name} received from {sender.name}: {message}")

# Example agents
class ClaimsAgent(Agent):
    pass

class EligibilityAgent(Agent):
    pass

class ProviderMatchAgent(Agent):
    pass
