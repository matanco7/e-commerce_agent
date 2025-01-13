
import json
from langchain_core.tools import tool
import pandas as pd

CONTACTS_FILE = 'files/contacts.csv'
ORDERS_FILE = 'files/orders.csv'

# ------------ START OF HELPER FUNCTIONS ------------

# Helper functions for evaluation and implementation of the agent.

# Function that takes a user message and return predicted intent.
def predict_intent(message):

    if "hi" in message.lower():
        return "greeting"
    elif "status" in message.lower():
        return "order_status"
    elif "return" in message.lower():
        return "return_policy"
    elif "human" in message.lower():
        return "human_representative"
    else:
        return "unknown"

# Function to save conversation to a JSON file.
def save_to_json(list_of_dicts, json_filename):
    with open(json_filename, mode='w') as file:
        json.dump(list_of_dicts, file, indent=4)

# Function to test the agent's intent recognition
def evaluate_intent_recognition(conversation):
    results = []
    for message in conversation:
        if message["role"] == "user":
            predicted_intent = predict_intent(message["content"])
            actual_intent = message["intent"]
            results.append((message["content"], actual_intent, predicted_intent))
    return results

# ------------ START OF TOOLS (Funcion calling) ------------


# Helper tools for the LLM.
# every tool has a decorator @tool and after it a declartion
# the tool must have a description this is useful for the llm to know when he should call this function

@tool
async def add_contact_to_csv(full_name: str, email: str, phone: str) -> str:
    """ Use this tool when user gives you his full name, email and phone. You need to store this information in a CSV file."""
    # Read existing data from the CSV file
    df = pd.read_csv(CONTACTS_FILE)
    
    # Create a new DataFrame with the provided details
    new_row = pd.DataFrame({'Full Name': [full_name], 'Email': [email], 'Phone': [phone]})
    
    # Concatenate the new row to the DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the DataFrame back to the CSV file
    df.to_csv(CONTACTS_FILE, index=False)
    return f"Ok {full_name}, we will be in touch"

@tool
async def get_order_status(order_id: str) -> str:
    """Use this tool when the user want to know the status of an order from the orders CSV file """
    # Read the existing data from the CSV file
    df = pd.read_csv(ORDERS_FILE)
    
    # Find the row with the given order ID
    order = df[df['Order ID'] == order_id]
    
    if not order.empty:
        # Return the status of the found order
        status = order.iloc[0]['Status']
        return f"Order {order_id} has status '{status}'."
    else:
        return f"Order {order_id} not found."

