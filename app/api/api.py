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

  processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
  B_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl",torch_dtype=torch.float16, device_map="auto")

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token='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')
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto',torch_dtype=torch.float16, use_auth_token="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")

  B_INST, E_INST = "[INST]", "[/INST]"
  B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
  DEFAULT_SYSTEM_PROMPT = """\
  The structure of the output should look like this only, for example:

  overview: 'Here's a product description for the Apple iPhone 11
  Pro 256GB Space Grey: The Apple iPhone 11 Pro is the latest flagship device from Apple, featuring a
  powerful A13 Bionic chip, a stunning 6.1-inch Super Retina HD display, and an impressive quad-camera
  system. With a sleek and durable design, this phone is sure to turn heads. The A13 Bionic chip
  provides lightning-fast performance and efficient battery life, allowing you to multitask with ease
  and enjoy your favorite apps and games without worrying about running out of juice. The quad-camera
  system includes a wide-angle lens, a telephoto lens, and a macro lens, giving you more creative
  options when capturing photos and videos. The iPhone 11 Pro also features a high-quality audio
  experience, with improved speakers and a new spatial audio feature that immerses you in your
  favorite music, movies, and games. With Apple's advanced Face ID technology, you can unlock your
  phone with just a glance, and the phone's long-lasting battery ensures that you can use it all day
  without needing to recharge. In terms of design, the iPhone 11 Pro features a sleek and durable
  stainless steel and glass construction, available in three gorgeous colors: Space Grey, Gold, and
  Silver. The phone's IP68 rating means it can withstand being submerged in water up to 4 meters for
  up to 30 minutes, making it perfect for those who love to take their phone with them wherever they
  go.I hope this product description helps! Let me know if you have
  any other questions.', 'estimated_price': '$324'

  this is just an example for the products. Give me output in these way. Do not generate any additional dialogues

  Your answers should not include any harmful, racist, sexist, toxic, dangerous content. Please ensure that your responses are socially unbiased, informative and positive.

  Always generate a product description which is provided by the user. The description should be approximately for about 300 words. Also provide estimated price of the product"""

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
      with torch.autocast('cuda', dtype=torch.bfloat16):
          inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
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
  question = "what is in the image?"
  inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
  out = B_model.generate(**inputs)
  prompt = processor.decode(out[0], skip_special_tokens=True)
  generated_text = generate(prompt)

  return parse_text(generated_text)


def est_price(input_string):
  pattern = r"Estimated Price: \$(\d+)"
  match = re.search(pattern, input_string)
  return match.group()


# from PIL import Image
# import urllib.request
# urllib.request.urlretrieve("https://gomloffers.s3.amazonaws.com/product_overview/1429474.jpeg")
# img = Image.open("1429474.jpeg")
# img.show()