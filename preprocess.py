import json
file = open("data/file.json1", "rb")

lines = file.readlines()

print("Keys in data = {}".format(json.loads(lines[0]).keys()))

for line in lines:
    text = json.loads(line)["text"]
    labels = json.loads(line)["text"]
