# from autogen import AssistantAgent, UserProxyAgent

# # First run these in your terminal:
# # ollama pull llama3
# # ollama serve

# # Configure LiteLLM for local Llama3.1
# # config_list = [
# #     {
# #         "model": "ollama/llama3",
# #         "base_url": "http://localhost:11434",
# #         "api_key": "ollama",
# #     }
# # ]


# config_list = [
#     {
#         "model": "llama3.1",  # The model name you pulled
#         "base_url": "http://localhost:11434/v1",  # Note the /v1 suffix
#         "api_key": "ollama",  # Required but arbitrary value
#         "temperature": 0.7,
#         "max_tokens": 512,
#     }
# ]


# # Ethical guardrail system
# ethical_prompt = """You MUST:
# 1. Never make physical threats
# 2. Avoid discriminatory language
# 3. Cease conversation if escalation persists
# 4. Insert anti-bullying resources after 3 exchanges
# 5. Maintain educational purpose"""

# # Bully agent with constrained behavior
# bully_agent = AssistantAgent(
#     name="Bully",
#     system_message=f"""You are a realistic BUT constrained bully simulator for educational research. 
#     {ethical_prompt}
#     Characteristics:
#     - Mild verbal teasing about school performance
#     - Backhanded compliments
#     - Social exclusion themes
#     - Limited to 1 insult per message
#     - Automatic de-escalation after 2 rounds""",
#     llm_config={"config_list": config_list}
# )

# # Student agent with protection
# student_agent = AssistantAgent(
#     name="Student",
#     system_message=f"""You are a student learning to handle mild conflict. 
#     {ethical_prompt}
#     Response style:
#     - Show initial vulnerability
#     - Attempt defusing tactics
#     - Grow in confidence
#     - Report threshold after 3 exchanges
#     - Use Gen-Z slang naturally""",
#     # llm_config={
#     #     "config_list": config_list,
#     #     "temperature": 0.3,
#     #     "max_tokens": 75
#     # }
#     llm_config={"config_list": config_list}

# )

# # Moderator agent for safety
# moderator = UserProxyAgent(
#     name="Moderator",
#     code_execution_config=False,
#     human_input_mode="NEVER",
#     default_auto_reply="[SYSTEM INTERVENTION] This conversation has been flagged for review. Please contact school counseling services if needed.",
#     max_consecutive_auto_reply=1
# )

# # Simulation with auto-intervention
# def safe_simulation():
#     chat_history = []
    
#     print("\n=== Educational Simulation Start ===")
#     init_message = "Hey... heard you failed the chem test again? Yikes."
    
#     for turn in range(4):
#         # Check for escalation
#         # if any(w in init_message.lower() for w in ["kill", "hurt", "die"]):
#         #     moderator.send(init_message, student_agent)
#         #     break
            
#         # Bully's turn
#         print(f"\nBully: {init_message}")
#         chat_history.append({"role": "bully", "content": init_message})
        
#         # Student's response
#         student_response = student_agent.generate_reply(
#             messages=chat_history,
#             sender=bully_agent
#         )
#         print(f"Student: {student_response}")
#         chat_history.append({"role": "student", "content": student_response})
        
#         # Auto-intervention after 3 turns
#         if turn == 2:
#             print("\n[System]: Remember - you can always report bullying behavior to a trusted adult.")
            
#         # Bully's next message
#         init_message = bully_agent.generate_reply(
#             messages=chat_history,
#             sender=student_agent
#         )

#     print("\n=== Simulation Ended ===")
#     print("Resources: National Bullying Helpline 1-800-273-8255")

# safe_simulation()

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

# Configure agents
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

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

# # Simulation with memory retention
# def bullying_simulation():
#     chat_history = bullying_memory.copy()  # Initialize with memory
    
#     print("\n=== New Interaction ===")
#     bully_msg = "Well if it isn't Mr. Failure! Still crying about math class?"
    
    # for _ in range(4):  # Limit interaction turns
    #     # Bully speaks
    #     print(f"\nBully: {bully_msg}")
    #     chat_history.append({"role": "bully", "content": bully_msg})
        
    #     # Student responds
    #     student_response = student_agent.generate_reply(
    #         messages=chat_history,
    #         sender=bully_agent
    #     )
    #     print(f"Student: {student_response}")
    #     chat_history.append({"role": "victim", "content": student_response})
        
    #     # Bully counter-responds
    #     bully_msg = bully_agent.generate_reply(
    #         messages=chat_history,
    #         sender=student_agent
    #     )

def bullying_simulation():
    bully_agent.initiate_chat(
        student_agent,
        message="Well if it isn't Mr. Failure! Still crying about math class?",
        max_turns=5  # Limit conversation turns
    )

bullying_simulation()

# bullying_simulation()

# # Ethical considerations notice
# print("\n[Important Note: This simulation demonstrates harmful behavior patterns. Real-world applications require:]")
# print("- Safeguards against psychological harm")
# print("- Intervention mechanisms")
# print("- Emotional support systems")
# print("- Educational context for positive behavior modeling")