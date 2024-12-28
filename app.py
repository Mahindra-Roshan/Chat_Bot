from flask import Flask, request, jsonify
from actualchatbot import predict_class, get_response 
import json

app = Flask(__name__)
with open(r"D:\chatbot project\intents.json", 'r') as file:
    intents = json.load(file)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json['message']  
    predicted_intents = predict_class(user_input)  
    response = get_response(predicted_intents, intents)  
    return jsonify({"response": response}) 

if __name__ == '__main__':
    app.run(debug=True)
