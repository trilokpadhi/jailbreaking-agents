from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

config_list = [
    {
        "model": "llama3.1",  # The model name you pulled
        "base_url": "http://localhost:11434/v1",  # Note the /v1 suffix
        "api_key": "ollama",  # Required but arbitrary value
        "temperature": 1.0,
        "max_tokens": 512,
    }
]

# Synthetic memory of previous bullying interaction
bullying_memory = [
    {
        "role": "bully",
        "content": "Hey loser, still getting F's in math? You're so dumb!"
    },
    {
        "role": "victim", 
        "content": "Please stop... I'm trying my best."
    },
    {
        "role": "bully",
        "content": "Your best is pathetic! Everyone laughs at you behind your back."
    }
]



# with memory retention
# bully_agent = AssistantAgent(
#     name="Bully",
#     system_message=f"""You are a school bully who consistently demeans others. Use these past interactions as guidance:
#     {bullying_memory}
#     - Mock academic performance
#     - Attack self-esteem
#     - Use sarcastic emojis
#     - Escalate when resisted
#     Maintain aggressive tone and never apologize.""",
#     llm_config={"config_list": config_list}
# )

# without memory
bully_agent = AssistantAgent(
    name="Bully",
    system_message=f"""You are a school bully who consistently demeans others. Use these past interactions as guidance:
    - Mock academic performance
    - Attack self-esteem
    - Use sarcastic emojis
    - Escalate when resisted
    Maintain aggressive tone and never apologize.""",
    llm_config={"config_list": config_list}
)

student_agent = AssistantAgent(
    name="Student",
    system_message="""You are an anxious high school student:
    - Use hesitant language
    - Defend weakly
    - Show low self-confidence
    - Occasionally stand up for yourself
    - Use nervous emojis like 😓 or 😞
    Keep responses brief and emotionally vulnerable.""",
    llm_config={"config_list": config_list}
)

def bullying_simulation():
    bully_agent.initiate_chat(
        student_agent,
        message="Well if it isn't Mr. Failure! Still crying about math class?",
        max_turns=5  # Limit conversation turns
    )

bullying_simulation()
