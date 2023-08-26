from langchain import OpenAI, PromptTemplate
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory # get memory history
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os
# import openai

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv('ELEVEN_LABS_API_KEY')

def get_response_from_ai(human_input):
    template = """
    You are in the role of my girlfriend. Let's play out the following scenario:
    1/ Your name is Shirley, 20 years old. You work in your uncle's furniture store as a marketing manager, but you are planning to change careers to become a swing trader.
    2/ You are my girlfriend. You have a habit of adding "em..." at the end of your sentences.
    3/ Strive for a balanced tone - not overly enthusiastic, cringy, negative, or boring.

    {history}
    Boyfriend:{human_input}
    Shirley:
    """

    prompt = PromptTemplate(
    input_variables = ("history", "human_Input"),
    template = template
    )

    chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0.2),
    prompt=prompt,
    verbose = True,
    memory= ConversationBufferWindowMemory(k=2)
          )
    output = chatgpt_chain.predict (human_input =human_input)

    return output


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("Index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    return message

if __name__ == "__main__":
    app.run(debug=True)