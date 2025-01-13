
import os
from dotenv import load_dotenv, find_dotenv
import json
import asyncio
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from tools import add_contact_to_csv, get_order_status, predict_intent, save_to_json
from pathlib import Path
from flask import Flask, request, jsonify, render_template

_ = load_dotenv(find_dotenv()) 
api_key  = os.environ['OPENAI_API_KEY']   # Get the OpenAI API key from the environment
app = Flask(__name__)
# Create an instance of the ChatOpenAI class
llm = ChatOpenAI(temperature=0.3, 
                streaming=True, 
                openai_api_key=api_key) 

# Setting the system prompt
prompt = hub.pull("hwchase17/openai-tools-agent")
system_prompt = Path("system_prompt.txt").read_text().strip()
prompt.messages[0].prompt.template = system_prompt


# tools for the llm - the llm can use them to do more complex actions
tools = [add_contact_to_csv, get_order_status]

# agent wrapper for the llm - the agent is the one who will use the tools
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

# Create an empty list to store the conversation messages
messages = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    data = request.json
    user_input = data.get('message')
    chat_history = data.get('history', "")
    
    if user_input.lower() == 'exit':
        save_to_json(messages, "dialogue4.json")          # Rename to the specified dialogue file name to export the conversation
        return jsonify({"response": "Goodbye!", "history": chat_history})
    
    messages.append({"role": "user", "content": user_input, "intent": predict_intent(user_input)})
    response = ""
    
    async for event in agent_executor.astream_events(
            {"input": user_input, "chat_history": messages}, version="v1"
        ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                response += content
                print(content, end="", flush=True)
    
    messages.append({"role": "assistant", "content": response})
    return jsonify({"response": response.strip(), "history": chat_history})

if __name__ == '__main__':
    app.run(debug=True)