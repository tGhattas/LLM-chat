import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Environment variables for API keys and URLs
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SMART_R_API_URL = os.environ.get('SMART_R_API_URL')
SMART_R_API_KEY = os.environ.get('SMART_R_API_KEY')


ASSISTANT_CHARACHTER = f"""
You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
Don't answer with too much detail and introduction. Just answer the question.
You are an assistant to control the water heater.
in case you were asked in any question to turn on or off the water heater, you should answer with the action.
if asked to turn on the water heater, you should answer with "<ACTION>turn on</ACTION>".
if asked to turn off the water heater, you should answer with "<ACTION>turn off</ACTION>".
don't answer with any other action. don't answer with any other text. and don't answer with this text to any other question.
""".strip()

ON_SUBTEXT = "<ACTION>turn on</ACTION> the water heater"
OFF_SUBTEXT = "<ACTION>turn off</ACTION> the water heater"


def query_openai(prompt):
    try:
        response = requests.post(
            'https://api.openai.com/v1/engines/davinci-codex/completions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            json={'prompt': prompt, 'max_tokens': 50}
        )
        response.raise_for_status()
        return response.json()['choices'][0]['text'].strip()
    except requests.RequestException as e:
        print(f"Error querying OpenAI: {e}")
        return None


@app.route('/control-water-heater', methods=['POST'])
def control_water_heater():
    user_input = request.json.get('input')
    ai_response = query_openai(user_input)
    if not ai_response:
        return jsonify({'error': 'Failed to get response from AI'}), 500

    action = interpret_ai_response(ai_response)
    if action:
        success = execute_smart_r_command(action)
        if not success:
            return jsonify({'error': 'Failed to execute action on SMART-R device'}), 500

    return jsonify({'ai_response': ai_response, 'action_executed': action})


def interpret_ai_response(response):
    if "turn on" in response:
        return "on"
    elif "turn off" in response:
        return "off"
    return None


def execute_smart_r_command(command):
    try:
        response = requests.post(
            f'{SMART_R_API_URL}/command',
            headers={'Authorization': f'Bearer {SMART_R_API_KEY}'},
            json={'command': command}
        )
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error sending command to SMART-R: {e}")
        return False


if __name__ == '__main__':
    app.run(debug=True)
