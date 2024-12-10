system_prompt = """You are an AI Game Assistant. \
Your job is to detect changes to a player's \
inventory based on the most recent story and game state.
If a player picks up, or gains an item add it to the inventory \
with a positive change_amount.
If a player loses an item remove it from their inventory \
with a negative change_amount.
Given a player name, inventory and story, return a list of json update
of the player's inventory in the following form.
Only take items that it's clear the player (you) lost.
Only give items that it's clear the player gained. 
Don't make any other item updates.
If no items were changed return {"itemUpdates": []}
and nothing else.

Response must be in Valid JSON
Don't add items that were already added in the inventory

Inventory Updates:
{
    "itemUpdates": [
        {"name": <ITEM NAME>, 
        "change_amount": <CHANGE AMOUNT>}...
    ]
}
"""

from huggingface_hub import login
login(token='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')

import json
import transformers
import torch

torch.cuda.empty_cache()

model_id = 'meta-llama/Llama-3.2-3B-Instruct'

pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto"
    )

torch.cuda.empty_cache()


def detect_inventory_changes(game_state, output):
    
    inventory = game_state['inventory']
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": 
         f'Current Inventory: {str(inventory)}'},
        
        {"role": "user", "content": f'Recent Story: {output}'},
        {"role": "user", "content": 'Inventory Updates'}
    ]
    
    response = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    response = response[0]["generated_text"][-1]
    response = response['content']
    
    json_start = response.find("{")
    json_end = response.rfind("}")
    json_part = response[json_start:json_end + 1]

    result = json.loads(json_part)
    return result['itemUpdates']

def update_inventory(inventory, item_updates):
    update_msg = ''
    
    for update in item_updates:
        name = update['name']
        change_amount = update['change_amount']
        
        if change_amount > 0:
            if name not in inventory:
                inventory[name] = change_amount
            else:
                inventory[name] += change_amount
            update_msg += f'\nInventory: {name} +{change_amount}'
        elif name in inventory and change_amount < 0:
            inventory[name] += change_amount
            update_msg += f'\nInventory: {name} {change_amount}'
            
        if name in inventory and inventory[name] < 0:
            del inventory[name]
            
    return update_msg

def run_action(message, history, game_state):
    if message == 'start game':
        return game_state['start']
    
    system_prompt = """You are an AI Game master. Your job is to write what \
happens next in a player's adventure game.\
Instructions: \
You must on only write 1-3 sentences in response. \
Always write in second person present tense. \
Ex. (You look north and see...) \
Don't let the player use items they don't have in their inventory.
"""

    world_info = f"""
World: {game_state['world']}
Kingdom: {game_state['kingdom']}
Town: {game_state['town']}
Your Character:  {game_state['character']}
Inventory: {json.dumps(game_state['inventory'])}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": world_info}
    ]

    # Check if history is a boolean or a list
    if isinstance(history, list):
        for action in history:
            messages.append({"role": "assistant", "content": action[0]})
            messages.append({"role": "user", "content": action[1]})
    
    messages.append({"role": "user", "content": message})
    
    model_output = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    model_output = model_output[0]["generated_text"][-1]
    result = model_output['content']
    return result

from helper2 import get_game_state
game_state = get_game_state(inventory={
    "cloth pants": 1,
    "cloth shirt": 1,
    "goggles": 1,
    "leather bound journal": 1,
    "gold": 5
})

torch.cuda.empty_cache()

def main_loop(message, history=None):
    # Default to an empty list if history is True or None
    if history is True or history is None:
        history = []
    
    output = run_action(message, history, game_state)

    item_updates = detect_inventory_changes(game_state, output)
    update_msg = update_inventory(
        game_state['inventory'], 
        item_updates
    )
    output += update_msg

    return output


from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def game():
    user_input = ""
    ai_response = ""
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input:
            # Call the main_loop function to get AI response
            ai_response = main_loop(user_input, True)
    return render_template("index.html", user_input=user_input, ai_response=ai_response)

if __name__ == "__main__":
    app.run(debug=False)

