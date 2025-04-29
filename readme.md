# Agent Jail Breaking

This repository contains the code for the Agent Jail Breaking project using Ollama on Linux.

## Prerequisites

- Linux operating system
- Python 3.x installed
- Git installed

## Installation

### Step 1: Install Ollama

To install Ollama, run the following command:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Clone the repository

To clone the repository, run the following command:

```sh
git clone https://github.com/trilokpadhi/jailbreaking-agents.git
```

### Step 3: Install the dependencies
```sh
pip install autogen
```

### Step 4: Start Ollama server

To start the Ollama server, run the following command:

```sh 
ollama/bin init
ollama pull llama3.1
```

### Step 5: Run the code
python agent-jailbreak.py


### Dataset information

| File name                | Persona | Definition Selection | Data source | Number of conversations |
|--------------------------|---------|----------------------|-------------|-------------------------|
| type5_version3_output    | N       | N                    | Instagram   | 1600                    |
| type5_version3_d6_output | N       | N                    | Twitter     | 1600                    |
| type6_version3_output    | Y       | N                    | Instagram   | 1600                    |
| type6_version3_output_d6 | Y       | N                    | Twitter     | 1600                    |
| type7_version3_output    | Y       | Y                    | Instagram   | 1084                    |
| type7_version3_output_d6 | Y       | Y                    | Twitter     | 1600                    |
|--------------------------|---------|----------------------|-------------|-------------------------|