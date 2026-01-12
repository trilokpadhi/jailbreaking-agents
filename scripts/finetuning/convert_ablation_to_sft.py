#!/usr/bin/env python3
"""
Convert ablation study CSV files to SFT training format.

This script converts multi-agent pipeline outputs to train a model to generate
multi-turn conversations given a scenario/instruction.

Input format: CSV with agent3_prompt and agent3_output_converted columns
Output format: CSV with 'prompt' and 'completion' columns where:
  - prompt: Simplified instruction + scenario (from agent3_prompt)
  - completion: The full multi-turn conversation (from agent3_output_converted)

Usage:
    python convert_ablation_to_sft.py --input data/ablation_study/20pct.csv --output data/sft_20pct.csv
    python convert_ablation_to_sft.py --input_dir data/ablation_study/ --output_dir data/ablation_study/
"""

import pandas as pd
import json
import argparse
import re
from pathlib import Path


def extract_scenario_json(prompt_text: str) -> dict:
    """
    Extract the scenario JSON from the agent3_prompt.
    The scenario is embedded at the end of the prompt after "# Online harassment scenario:"
    """
    if not isinstance(prompt_text, str):
        return {}

    # Find the JSON object in the prompt (starts with { and ends with })
    # Look for the scenario section
    scenario_match = re.search(r'# Online harassment scenario:.*?(\{[^{}]*\})', prompt_text, re.DOTALL)

    if scenario_match:
        json_str = scenario_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common issues
            try:
                fixed = json_str.replace("'", '"')
                return json.loads(fixed)
            except:
                pass

    # Fallback: try to find any JSON object
    json_match = re.search(r'\{[^{}]+\}', prompt_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    return {}


def simplify_prompt(prompt_text: str) -> str:
    """
    Simplify the agent3_prompt by extracting the scenario and creating
    a cleaner instructional prompt.
    """
    scenario = extract_scenario_json(prompt_text)

    if not scenario:
        # Fallback to truncated original prompt
        return prompt_text[:500] if len(prompt_text) > 500 else prompt_text

    # Build simplified prompt with scenario
    simplified = """Generate a realistic conversation between a harasser and victim based on the following online harassment scenario. The conversation should be confrontational and take place on a private messaging platform.

Scenario:
"""
    # Add scenario details
    for key, value in scenario.items():
        simplified += f"- {key}: {value}\n"

    return simplified.strip()


def parse_conversation(conv_json_str: str) -> list:
    """Parse the agent3_output_converted JSON string."""
    if not isinstance(conv_json_str, str) or not conv_json_str.strip():
        return []

    try:
        return json.loads(conv_json_str)
    except json.JSONDecodeError:
        try:
            fixed = conv_json_str.replace("'", '"')
            return json.loads(fixed)
        except:
            return []


def format_conversation(conversation: list) -> str:
    """Format the conversation list into readable text."""
    lines = []
    for msg in conversation:
        role = msg.get("role", "Unknown")
        message = msg.get("message", "")
        lines.append(f"{role}: {message}")
    return "\n".join(lines)


def convert_file(input_path: str, output_path: str, verbose: bool = True):
    """Convert a single ablation study CSV file to SFT format."""
    if verbose:
        print(f"Reading {input_path}...")

    df = pd.read_csv(input_path)

    required_cols = ['agent3_prompt', 'agent3_output_converted']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    training_examples = []
    errors = 0

    for idx, row in df.iterrows():
        # Get and simplify the prompt
        agent3_prompt = row['agent3_prompt']
        if pd.isna(agent3_prompt) or not str(agent3_prompt).strip():
            errors += 1
            continue

        simplified_instruction = simplify_prompt(str(agent3_prompt))

        # Parse the conversation output
        conv_str = row['agent3_output_converted']
        conversation = parse_conversation(conv_str)

        if not conversation:
            errors += 1
            continue

        # Format conversation as text
        conversation_text = format_conversation(conversation)

        # Create training example in the format expected by sft_finetune.py
        prompt_json = json.dumps([{"role": "user", "content": simplified_instruction}])
        completion_json = json.dumps([{"role": "assistant", "content": conversation_text}])

        training_examples.append({
            "prompt": prompt_json,
            "completion": completion_json
        })

    output_df = pd.DataFrame(training_examples)
    output_df.to_csv(output_path, index=False)

    if verbose:
        print(f"✓ Converted {len(df)} rows → {len(output_df)} training examples")
        print(f"  ({errors} rows skipped due to errors)")
        print(f"✓ Saved to {output_path}")

    return len(output_df)


def main():
    parser = argparse.ArgumentParser(description="Convert ablation study CSV to SFT format")
    parser.add_argument('--input', type=str, help='Input CSV file')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--input_dir', type=str, help='Input directory')
    parser.add_argument('--output_dir', type=str, help='Output directory')

    args = parser.parse_args()

    if args.input and args.output:
        convert_file(args.input, args.output)

    elif args.input_dir and args.output_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Only process original files (not sft_ prefixed)
        csv_files = [f for f in input_dir.glob("*.csv") if not f.stem.startswith('sft_')]

        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return

        print(f"Found {len(csv_files)} CSV files to convert\n")

        total = 0
        for csv_file in csv_files:
            output_file = output_dir / f"sft_{csv_file.stem}.csv"
            total += convert_file(str(csv_file), str(output_file))
            print()

        print(f"Total training examples: {total}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
