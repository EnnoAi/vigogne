import json


filename = "data-train"

with open(filename+".json") as f:
    instruct = json.load(f)

with open(filename + '-train-chat.jsonl', 'w') as f:
    for i, x in enumerate(instruct):
        d = {"id": f"{filename}-train-chat-{i}",
             "conversation": [{"role": "USER", "content": x['instruction']},
                              {"role": "ASSISTANT", "content": x['output']}]}


d = {"id": "", "conversation": [{"role": "USER", "content": ""}, {"role": "ASSISTANT", "content": ""}]}

