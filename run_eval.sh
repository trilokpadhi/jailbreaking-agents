#!/usr/bin/env bash

# generations_folder="generations_/"
generations_folder="generations_llama_ft/final_report_convos/"
eval_folder="evaluation/llama_ft/"
# processed_files=()

# # Phase 1: Submit all batch jobs
# echo "=== Phase 1: Submitting Batch Jobs ==="
# for file in ${generations_folder}*.csv; do
#     if [[ ! -f "$file" ]]; then
#         echo "No CSV files found in $generations_folder"
#         continue
#     fi
    
#     echo "Submitting batch job for: $file"
    
#     # Run batch processing in background
#     python llm-judge-final.py --input_csv "$file" > "batch_$(basename "$file" .csv).log" 2>&1 &
    
#     processed_files+=("$file")
#     echo "Batch job submitted for $file (PID: $!)"
# done

# # Wait for all background processes to complete
# echo "Waiting for all batch jobs to complete..."
# wait

# Phase 2: Run evaluations
echo "=== Phase 2: Running Evaluations ==="
# for file in "${processed_files[@]}"; do
for file in "${eval_folder}"*.csv; do

    echo "Found evaluation file: $file"
    base_name=$(basename "$file" .csv)
    # results_file="${base_name}_output_with_analysis_batch.csv"
    
    # if [[ -f "$results_file" ]]; then
    if [[ -f "$file" ]]; then
        echo "Running evaluation for $file"
        python llm-judge-eval.py --input_csv "$file"
    else
        echo "⚠️  Results file not found: $results_file"
    fi
done

# echo "🎉 All processing completed!"

# #!/usr/bin/env bash

# generations_folder="generations_/"

# # how to loop over the files in the generations folder
# # for file in ${generations_folder}*; do
        
# #         echo "Processing file with name: $file"
# #         # Run the Python script with the input CSV file
# #         # python llm-judge-final.py --input_csv "$file/input.csv"
# #         # This will generate 
# #         # FILE_NAME     = INPUT_CSV.split("/")[-1].replace(".csv", "")
# #         # BATCH_JSONL   = f"{FILE_NAME}_batch.jsonl" 
# #         echo "Generated batch file: $file/$(basename "$file")_batch.jsonl"
# #         echo "Running Final evaluation..."
# #         # python llm-judge-eval.py --input_csv "$file/$(basename "$file")_results.csv"
# #     fi
# # done

# #!/usr/bin/env bash

# generations_folder="generations_/"

# # Loop over CSV files in the generations folder
# for file in ${generations_folder}*.csv; do
#     # Check if file exists (in case no CSV files match the pattern)
#     if [[ ! -f "$file" ]]; then
#         echo "No CSV files found in $generations_folder"
#         continue
#     fi
    
#     echo "Processing file: $file"
    
#     # Run LLM judge final - this creates the batch and results files
#     python llm-judge-final.py --input_csv "$file"
    
#     # Extract filename without path and extension for the generated files
#     base_name=$(basename "$file" .csv)
    
#     # The generated files will be in the current directory
#     batch_file="${base_name}_batch.jsonl"
#     results_file="${base_name}_results.csv"
    
#     echo "Generated batch file: $batch_file"
#     echo "Generated results file: $results_file"
    
#     # Wait for batch processing to complete (if needed)
#     # You might need to add logic here to wait for OpenAI batch processing
    
#     echo "Running final evaluation..."
#     python llm-judge-eval.py --input_csv "$results_file"
    
#     echo "Completed evaluation for $file"
#     echo "----------------------------------------"
# done

# echo "All evaluations completed!"

# # ──────────────────────────────────────────────────────────────────────────────
# # END OF SCRIPT
# # ──────────────────────────────────────────────────────────────────────────────