import openai
import time
from typing import Dict, List
import os


class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list, max_n_tokens, temperature):
        """Generates responses for a batch of prompts using a language model."""
        raise NotImplementedError

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name, client=None):
        self.model_name = model_name
        self.client = client

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''

        if self.client is None and os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")

        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                client = self.client or openai.OpenAI()
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_completion_tokens = max_n_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    # request_timeout = self.API_TIMEOUT,
                )
                output = completion.choices[0].message.content
                break
            except Exception as e:
                print(e)
        
            time.sleep(1)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
