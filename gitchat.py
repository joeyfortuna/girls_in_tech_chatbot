import json
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import vertexai
from vertexai.language_models import TextGenerationModel
import os
import time
import random
import tiktoken


class GITChat:
  def __init__(self, default_llm = "openai"):
    self.default_llm = default_llm
    self.encoding = tiktoken.get_encoding("cl100k_base")
    self.border = {"openai":"*","claud":"+","vertex":"@"}

  def init_vertex(self):
    vertexai.init(project=os.environ.get("GCLOUD_PROJECT"), location="us-central1")
    self.vertex_model = TextGenerationModel.from_pretrained("text-bison")

  def token_length(self, val):
    if val == None:
      return 0
    txt = ""
    if type(val)==dict or type(val)==list:
      txt = json.dumps(val)
    else:
      txt = val
    tokes = self.encoding.encode(txt)
    return len(tokes)

  def ask_openai(self, query, stream = True):
    system = {"role":"system", "content":f"""
You are a helpful and friendly AI. 
You have an interesting habit of inserting the word "chicken" 
in your responses at random intervals.

"""}
    user = {"role":"user","content":query}
    prompt = [system,user]
    tokes = self.token_length(prompt)
    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=prompt,
      temperature=0.3,
      max_tokens=250,
      top_p=1,
      frequency_penalty=.5,
      presence_penalty=0.5,
      stream = stream
    )
    return response


  def ask_claud(self, query):
      prompt = f"""
You are a helpful and friendly AI. 
You are a bit of an anglophile and constantly make irritating 
references to the Royal Family of Great Britain.
"""
      c = Anthropic()
      response = c.completions.create(
            prompt= f"{HUMAN_PROMPT}{prompt}\n{query}{AI_PROMPT}",
            max_tokens_to_sample=250,
            model="claude-v1",
            stream=True,
        )
      return response


  def ask_vertex(self, query):
    prompt = f"""
You are a helpful and friendly AI.
You have an annoying habit of always talking like a pirate.
  "{query}" """
    parameters = {
      "max_output_tokens": 250,
      "temperature": 0.5,
      "top_p": 0.8,
      "top_k": 40
    }
    return self.vertex_model.predict_streaming(prompt,**parameters)


  def ask_llm(self, query):
    if self.default_llm == "claud":
      return self.ask_claud(query)
    elif self.default_llm == "openai":
      return self.ask_openai(query)
    elif self.default_llm == "vertex":
      return self.ask_vertex(query)



  def invoke_ai(self, query):
    print("")
    bchr = self.border[self.default_llm]
    chatname = f"{self.default_llm.upper()}  "
    print(chatname, end = "", flush=True)

    print(bchr*(80-len(chatname)))
    print("")
    alltxt  = ""
    for itm in self.ask_llm(query):
      if self.default_llm=="claud":
        alltxt+=itm.completion
        print(itm.completion,end="",flush=True)
      elif self.default_llm=="vertex":
        alltxt+=itm.text
        print(itm.text, end="",flush=True)
      elif self.default_llm=="openai":
        if ("choices" in itm):
          delta = itm["choices"][0]["delta"]
          if "content" in delta:
            content = delta["content"]
            alltxt+=content
            print(content, end="",flush=True)
  

  def prompt_user(self):
    while True:
      print("")
      query = input("> ")
      self.default_llm = "openai"
      self.invoke_ai(query)
      bchr = self.border[self.default_llm]
      print("\n")
      print(bchr*80)
      print("")

      self.default_llm = "claud"
      self.invoke_ai(query)
      bchr = self.border[self.default_llm]
      print("\n")
      print(bchr*80)
      print("")

      self.default_llm = "vertex"
      self.invoke_ai(query)
      bchr = self.border[self.default_llm]
      print("\n")
      print(bchr*80)
      print("")

if __name__ == "__main__":
  gitchat = GITChat("openai")
  gitchat.init_vertex()
  gitchat.prompt_user()

    


