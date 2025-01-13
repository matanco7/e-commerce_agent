
from sklearn.metrics import classification_report
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from tools import evaluate_intent_recognition


with open("dialogue3.json", "r") as file:        # Rename to the specified dialogue file
    conversation = json.load(file)

# Accuracy metric via intent recognition
results = evaluate_intent_recognition(conversation)

# Prepare lists of actual and predicted intents
actual_intents = [actual for _, actual, _ in results]
predicted_intents = [predicted for _, _, predicted in results]

report = classification_report(actual_intents, predicted_intents, target_names=['greeting', 'order_status', 'return_policy', 'human_representative', 'unknown'])
print(report)

# BLEU score metric
reference_responses = [
    "Hello Matan! How can I help you today?",  # Reference for greeting
    "The status of order ORD0010 is 'Delivered'. Do you have any other questions?",  # Reference for order status
    "Refunds are processed to the original payment method. Credit card payments will be refunded to the card, and cash/check payments will be refunded in cash.",  # Reference for return policy
    "Please provide me with your full name, email, and phone number so I can connect you with a human representative.", # Reference for human representative
    "Thank you, Israel Israeli. A human representative will contact you shortly. If you have any other questions, please let me know."    # Reference for human representative
]

candidates = [
    {
        "user_message": "hi im matan",
        "assistant_response": "Hello Matan! How can I assist you today?",
    },
    {
        "user_message": "what is the status of ORD0010?",
        "assistant_response": "Order ORD0010 has the status 'Delivered'. Is there anything else you would like to know or inquire about?",
    },
    {
        "user_message": "I want to return it, how will i receive my refund?",
        "assistant_response": "Refunds will be issued to the original form of payment. If you paid by credit card, the refund will be credited to your card. If you paid by cash or check, you will receive a cash refund. Is there anything else you would like to know or do regarding your order return?",
    },
    {
        "user_message": "I want to speak with human representative",
        "assistant_response": "I will need some information to connect you with a human representative. Please provide me with your full name, email, and phone number.",
    },
    {
        "user_message": "israel israeli israel90@gmail.com 052099911124",
        "assistant_response": "Israel Israeli, you will be contacted shortly by a human representative. If you have any other questions or concerns, feel free to let me know.",
    },
]



# Calculate BLEU score for each assistant response
references = [[ref.split() for ref in refs] for refs in reference_responses]
candidates = [cand["assistant_response"].split() for cand in candidates]

smoothing_function = SmoothingFunction().method1  # To avoid zero BLEU scores for short sentences
bleu_scores = []
for i in range(len(candidates)):
    score = sentence_bleu(reference_responses[i], candidates[i], smoothing_function=smoothing_function)
    bleu_scores.append(score)

# Print BLEU scores
for i, score in enumerate(bleu_scores):
    print(f"Candidate {i+1}: BLEU score = {score:.4f}")

# Users satisfaction scores (USS) metrics are on the report.