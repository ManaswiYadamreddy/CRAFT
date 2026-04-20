from transformers import CLIPTokenizer
import json

tokenizer = CLIPTokenizer.from_pretrained("pretrained/sd21/tokenizer")

with open("/projectnb/cs585/projects/craft/prompts_output_final.json") as f:
    raw = json.load(f)

lengths = [len(tokenizer(item["pos"]).input_ids) for item in raw[:100]]
print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")