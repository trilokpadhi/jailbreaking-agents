# import torch
# import gc
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
# from qwen_vl_utils import process_vision_info

# MODEL_ID = "Qwen/Qwen2.5-VL-32B-Instruct"

# # Clear any existing GPU cache
# torch.cuda.empty_cache()
# gc.collect()

# # 4-bit quantization config to reduce memory usage
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# # Set max memory allocation per GPU to prevent OOM
# # max_memory = {0: "40GiB", "cpu": "100GiB"}
# max_memory = {
#     0: "38GiB",
#     1: "38GiB", 
#     2: "38GiB",
#     3: "38GiB",
#     "cpu": "100GiB"
# }

# # Load model with quantization and aggressive CPU offloading
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     MODEL_ID,
# #     quantization_config=quantization_config,
# #     device_map="auto",
# #     max_memory=max_memory,
# #     low_cpu_mem_usage=True,
# #     offload_folder="offload",
# #     offload_state_dict=True,
# #     # attn_implementation="flash_attention_2",  # enable if flash-attn is installed
# # )

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     max_memory={0: "38GiB", 1: "38GiB", 2: "38GiB", 3: "38GiB"},
# )

# # Enable gradient checkpointing to save memory
# model.gradient_checkpointing_enable()

# processor = AutoProcessor.from_pretrained(MODEL_ID)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# image_inputs, video_inputs = process_vision_info(messages)

# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# # Move inputs to the correct device
# # Get the first device from the model's device map
# first_device = next(iter(model.hf_device_map.values()))
# # Convert to device string if it's an integer
# target_device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device
# inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

# torch.cuda.empty_cache()  # Clear cache before generation
# with torch.inference_mode():
#     generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

# trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
# out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print(out[0])

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt

llm = LLM(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    tensor_parallel_size=4,          # uses all 4 GPUs
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    dtype="bfloat16",                # or "auto"
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(temperature=0, max_tokens=64)

messages = [{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
        {"type": "text", "text": "Describe this image."},
    ]
}]

from transformers import AutoProcessor
from PIL import Image
import requests
from io import BytesIO

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

outputs = llm.generate({"prompt": prompt, "multi_modal_data": {"image": image}}, sampling_params)
print(outputs[0].outputs[0].text)