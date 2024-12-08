import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from utils.idselector import VIDEO_ID
from utils.railway_dictionary import RAILWAY_IDS

class RailwaysAnnouncementPreprocessor:
   def __init__(self, dictionary=RAILWAY_IDS):
      self.multi_word_list = [key for key, value in dictionary.items() if len(key.split()) > 1]
      self.stop_words = [
         "a", "an", "the","by", 
         "is", "am", "are", "was", "were", "be", "been", "being", 
         "do", "does", "did", "has", "have", "had", "will", "shall", 
         "would", "should", "can", "could", "may", "might", "must",
         "and", "or", "nor", "so", "for", "yet", 
         "oh", "uh", "um", "ah", "wow", 
      ]
      self.number_words = {
      "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
      "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
      "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
      "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
      "eighteen": 18, "nineteen": 19, "twenty": 20
      }
      # self.llm = ChatOpenAI(model='gpt-4o')
      self.llm = ChatGroq(model='llama-3.3-70b-versatile')
    
   def preprocess(self, sentence):
      sentence = " ".join([word.lower() for word in sentence.split() if word.lower() not in self.stop_words])
      prompt_template = PromptTemplate.from_template("""
         You are a highly advanced text preprocessing assistant for Indian Sign Language (ISL) translation. Your task is to process railway announcements into context-aware tokens suitable for ISL video generation. Follow these steps precisely and **STRICTLY ADHERE** to the instructions.

         NOTE: 
         1. Stop words have already been removed from the sentence. **Do NOT remove stop words again.**
         2. **Do NOT ignore or skip any words under any circumstances.**

         PRELIMINARY STEP: 
         - All words in the sentence are to be converted to lowercase, even if it's a place, city, state, or any proper noun.

         ### Inputs:
         - Sentence: {sentence}
         - List: {list}

         ### Steps:

         1. **Lemmatize Tokens**:
            - Perform **lemmatization using part-of-speech tagging (POS)**.
            - Ensure all tokens are lemmatized to their root forms based on their context and part of speech. 
            - Examples:
            - "leaving/left" → "leave"
            - "platforms" → "platform"
            - "assisting" → "assist"
            - "requested" → "request"
            - "departing/departed" → "depart"
            - "flies" → "fly" (if referring to the verb).
            - **DO NOT SKIP lemmatization for any words. Each word MUST be processed.**

         2. **Context-Based Tokenization**:
            - Treat multi-word phrases from the list as single tokens.
            - Rules:
            - Multi-word phrases in the list (e.g., "how many") should NOT be tokenized into separate words ("how" and "many").
            - If no multi-word match is found, proceed with normal tokenization.
            - They must not be capitalized or altered in any way.

         3. **Named Entity Recognition (NER)**:
            - Extract:
            - Train Numbers (e.g., "train no. 1675" → `train`, `number`, `1`, `6`, `7`, `5`).
            - Platform Names (e.g., "platform 9B" → `platform`, `9`, `B`).
            - Times (e.g., "12:45" → `1`, `2`, `4`, `5`).
            - Station Names (e.g., "Andhra Pradesh", "New Delhi", "Tamil Nadu").
            - They must not be capitalized or altered in any way.

         4. **Normalize Time and Numbers**:
            - Break time into individual digits.
            - Replace abbreviations with their expanded forms:
            - Example: "no." → "number", "P/F" → "platform".
            - Alphabets which are part of the token should be retained as they are.
            - Example: "9 B" → `9`, `B`.

         5. **Combine Tokens**:
            - Retain the sequential order of the sentence while ensuring the output contains ISL-friendly tokens.

         ### Example Inputs and Outputs:
         Input:
         "Attention all, train no. 1675, Rajdhani from platform 9B is leaving from Andhra Pradesh at 12:45."

         Output:
         ['attention', 'all', 'train', 'number', '1', '6', '7', '5', 'rajdhani', 'from', 'platform', '9', 'B', 'leave', 'from', 'andhra pradesh', 'at', '1', '2', '4', '5']
         
         Input:
         "Attention all, train no. 1675, Rajdhani from platform nine B is leaving from Madhya Pradesh at 12:45."

         Output:
         ['attention', 'all', 'train', 'number', '1', '6', '7', '5', 'rajdhani', 'from', 'platform', '9', 'B', 'leave', 'from', 'madhya pradesh', 'at', '1', '2', '4', '5']

         ### Important:
         - **NO PREAMBLE OR EXPLANATIONS** in your response.
         - Only return the processed tokens in a **list format**.
         - You are not to return code.
         - Any deviation will result in severe penalties.

         Now process the sentence according to these instructions.
      """)
      
      prompt = prompt_template.invoke({
         'sentence': sentence,
         'list': str(self.multi_word_list),
      })
      print("Prompt",prompt,end="\n\n")
      
      response = self.llm.invoke(prompt)
      keys = eval(response.content)
      final_keys = []
      for key in keys:
        if key in RAILWAY_IDS.keys():
            final_keys.append(key)
        elif key in self.number_words.keys():
            final_keys.append(str(self.number_words[key]))
        elif key == ' ':
            continue
        else:
            for letter in key:
               final_keys.append(letter.upper())
      return final_keys

if __name__ == "__main__":
   load_dotenv()
   preprocessor = RailwaysAnnouncementPreprocessor()
   sentence = "Attention all, train vande bharat from platform 9B is leaving from Andhra Pradesh."
   print(preprocessor.preprocess(sentence))