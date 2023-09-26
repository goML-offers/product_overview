import torch
import transformers
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Blip2Processor, Blip2ForConditionalGeneration
import json
import re
import textwrap
import requests
from PIL import Image



def prod_overview(url): #amazon s3 url. The location of the image
  device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

  processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl",cache_dir="/tmp/")
  B_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",torch_dtype=torch.float32, device_map="auto",cache_dir="/tmp/")

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token='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',cache_dir="/tmp/")
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto',torch_dtype=torch.float32, use_auth_token="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",cache_dir="/tmp/")

  B_INST, E_INST = "[INST]", "[/INST]"
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
  DEFAULT_SYSTEM_PROMPT = """\
  The structure of the output should look like this only, for example:
  {
    "Product_Overview": "some text",
    "Estimated_Price": "$x - $y",
    "Product_Description”:”some text"
  }
  this is just an example for the products. Give me output in these way. Do not generate any additional dialogues
  Your answers should not include any harmful, racist, sexist, toxic, dangerous content. Please ensure that your responses are socially unbiased, informative and positive.
  Always generate a Product_Overview, Estimated_Price, Product_Description which is provided by the user. Provide your answer in json format"""

  SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

  def get_prompt(instruction):
      prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
      return prompt_template

  def cut_off_text(text, prompt):
      cutoff_phrase = prompt
      index = text.find(cutoff_phrase)
      if index != -1:
          return text[:index]
      else:
          return text

  def remove_substring(string, substring):
      return string.replace(substring, "")



  def generate(text):
      prompt = get_prompt(text)
      with torch.autocast('cpu', dtype=torch.bfloat16):
          inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
          outputs = model.generate(**inputs,
                                  max_new_tokens=512,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.eos_token_id,
                                  )
          final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
          final_outputs = cut_off_text(final_outputs, '</s>')
          final_outputs = remove_substring(final_outputs, prompt)

      return final_outputs#, outputs

  def parse_text(text):
          wrapped_text = textwrap.fill(text, width=100)
          print(wrapped_text +'\n\n')
          return wrapped_text


  img_url = url
  raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
  question = "in 5 words tell me , what is in the image?"
  inputs = processor(raw_image, question, return_tensors="pt").to("cpu")
  out = B_model.generate(**inputs)
  prompt = processor.decode(out[0], skip_special_tokens=True)
  print("======")
  print(prompt)
  print("======")
  generated_text = generate(prompt)
  text=re.sub(r'^.*?{', '{', generated_text)
  return text


def est_price(input_string):
  pattern = r"Estimated Price: \$(\d+)"
  match = re.search(pattern, input_string)
  return match.group()


# from PIL import Image
# import urllib.request
# urllib.request.urlretrieve("https://gomloffers.s3.amazonaws.com/product_overview/1429474.jpeg")
# img = Image.open("1429474.jpeg")
# img.show()
