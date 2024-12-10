import json
import transformers
import torch

from huggingface_hub import login
login(token='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')

model_id = 'meta-llama/Llama-3.2-3B-Instruct'

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map= 0,
)

def load_world(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_game_state(inventory={}):
    world = load_world(r"C:\Users\arnav\OneDrive\Desktop\ai_powered_dm\shared_data\Kyrethys_as_in_video.json")
    kingdom = world['kingdoms']['Aerthys']
    town = kingdom['towns']["Brindlemark"]
    character = town['npcs']['Kaelin Darkshadow']
    start = world['start']

    game_state = {
        "world": world['description'],
        "kingdom": kingdom['description'],
        "town": town['description'],
        "character": character['description'],
        "start": start,
        "inventory": inventory
    }

    return game_state

def run_action(message, history, game_state):
    
    if(message == 'start game'):
        return game_state['start']
        
    system_prompt = """You are an AI Game master. Your job is to write what \
happens next in a player's adventure game.\
Instructions: \
You must on only write 1-3 sentences in response. \
Always write in second person present tense. \
Ex. (You look north and see...)"""
    
    world_info = f"""
World: {game_state['world']}
Kingdom: {game_state['kingdom']}
Town: {game_state['town']}
Your Character:  {game_state['character']}"""


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": world_info}
    ]

    for action in history:
        messages.append({"role": "assistant", "content": action[0]})
        messages.append({"role": "user", "content": action[1]})
           
    messages.append({"role": "user", "content": message})
    
    model_output = pipeline(
        messages,
        max_new_tokens=256,
    )
    
    result = model_output[0]["generated_text"][-1]
    result = result['content']
    return result



