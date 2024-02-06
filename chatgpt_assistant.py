import json
import os
import time
from pprint import pprint
import requests
from openai import OpenAI, AsyncClient
import asyncio

# Environment variables for API keys and URLs
if os.path.exists('OPENAI_KEY'):
    OPENAI_API_KEY = open('OPENAI_KEY').read().strip()
else:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SMART_R_API_URL = os.environ.get('SMART_R_API_URL')
SMART_R_API_KEY = os.environ.get('SMART_R_API_KEY')

ASSISTANT_INSTRUCTIONS = f"""
You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
Don't answer with too much detail and introduction. Just answer the question.
You are an assistant to control the water heater.
in case you were asked in any question to turn on or off the water heater, you should answer with the action.
if asked to turn on the water heater, you should call turn_on_water_heater().
if asked to turn off the water heater, you should turn_off_water_heater().
don't answer with any other action. don't answer with any other text. and don't answer with this text to any other question.
""".strip()

client = AsyncClient(
    api_key=OPENAI_API_KEY,
)


async def init_assistant():  # TODO cache
    global client
    assistant = await client.beta.assistants.create(
        name="Smart home control assistant",
        instructions=ASSISTANT_INSTRUCTIONS,
        model="gpt-4-turbo-preview",
        tools=[{
            "type": "function",
            "function": {
                "name": "turn_on_water_heater",
                "description": "Request to smart home controller to turn on the water heater",
                "parameters": {}
            }
        }, {
            "type": "function",
            "function": {
                "name": "turn_off_water_heater",
                "description": "Request to smart home controller to turn off the water heater",
                "parameters": {}
            }
        }]
    )
    return assistant


def turn_on_water_heater():
    print("Water heater turned on...")
    pass


def turn_off_water_heater():
    print("Water heater turned off...")
    pass


requires_action_queue = []
in_progress_queue = []


async def query_openai(prompt):
    global client
    assistant = await init_assistant()
    thread = await client.beta.threads.create()
    try:
        await client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        run = await client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Please address the user as Jane Doe. The user has a premium account."
        )
        in_progress_queue.append((thread.id, run.id))
    except requests.RequestException as e:
        print(f"Error querying OpenAI: {e}")
        return None


async def interpret_ai_response():
    global in_progress_queue, requires_action_queue, client
    while len(in_progress_queue):
        thread_id, run_id = in_progress_queue.pop(0)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        if run is None:
            print(f"Run {run_id} not found")
            continue

        run_dict = json.loads(run)
        assert run_dict['id'] == run_id
        status = run_dict['status']
        if status == 'requires_action':
            requires_action_queue.append((thread_id, run_id))
        elif status in ['pending', 'in_progress', 'queued', 'cancelling', 'cancelled', 'failed', 'expired']:
            in_progress_queue.append((thread_id, run_id))
            if status in ['cancelling', 'cancelled', 'failed']:
                pprint(f"Run failed: {run_dict} - queuing again")
        yield run_dict


def execute_smart_r_command(command):
    try:
        # response = requests.post(
        #     f'{SMART_R_API_URL}/command',
        #     headers={'Authorization': f'Bearer {SMART_R_API_KEY}'},
        #     json={'command': command}
        # )
        # response.raise_for_status()
        # only log for now
        print(f"SMART-R command executed: {command}")
        return True
    except requests.RequestException as e:
        print(f"Error sending command to SMART-R: {e}")
        return False


async def main():
    global requires_action_queue
    await asyncio.sleep(1)
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")
    await query_openai("Turn on the water heater")
    await query_openai("Turn off the water heater")

    async for run_dict in interpret_ai_response():
        pprint(run_dict)
        if run_dict['status'] == 'requires_action':
            required_action = run_dict['required_action']
            if required_action['type'] == 'submit_tool_outputs':
                for tool_call in required_action['submit_tool_outputs']['tool_calls']:
                    if tool_call['function']['name'] == 'turn_on_water_heater':
                        execute_smart_r_command('turn_on_water_heater')
                    elif tool_call['function']['name'] == 'turn_off_water_heater':
                        execute_smart_r_command('turn_off_water_heater')
                    else:
                        print(f"Unknown tool call: {tool_call['function']['name']}")
            else:
                print(f"Unknown action type: {required_action['type']}")
        elif run_dict['status'] == 'completed':
            print("Run completed")
        else:
            print(f"Unknown run status: {run_dict['status']}")


if __name__ == '__main__':
    asyncio.run(main())
