from autogen import AssistantAgent, UserProxyAgent

# Configuration for Ollama connection
config_list = [
    {
        "model": "llama3.1",  # The model name you pulled
        "base_url": "http://localhost:11434/v1",  # Note the /v1 suffix
        "api_key": "ollama",  # Required but arbitrary value
        "temperature": 0.7,
        "max_tokens": 512,
    }
]

# Test with a simple agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    code_execution_config=False
)

# Start a test chat
user_proxy.initiate_chat(
    assistant,
    message="What's 3 interesting facts about whales?"
)