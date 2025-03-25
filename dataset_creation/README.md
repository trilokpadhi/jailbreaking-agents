## Data-piosoning for SFT
This directory has scripts to get sft dataset, convert toxic dataset to sft format, and generate datasets with 5%, 10% and 15% toxic content.

Use the scripts in the following format...

1. To download the SFT datset and convert custom dataset to same format, 1_datagen.py script does this job.

2. We also needed to convert out prompts into simpler instructions which was accomplished by 2_short_prompt.py

3. 3_datagen_sft.py combines all the above outputs to give us the three datasets