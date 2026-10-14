# %%
import json

json_file = 'test.json'
# %%
with open(json_file, "r") as test_json:
    json_dataset = json.load(test_json)
# %%
print(json_dataset["assets"]["d37f8621bf6f327da7a213ec653ed59a"]["asset"]["name"])

# %%
for i in json_dataset["assets"]:
    print(json_dataset["assets"][i]['asset']["name"])
    width = json_dataset["assets"][i]["asset"]['size']['width']
    height = json_dataset["assets"][i]["asset"]['size']['height']
    print(f"width : {width}  height : {height}")
# %%
json_dataset["assets"]["d37f8621bf6f327da7a213ec653ed59a"]["asset"]["name"]
# %%
json_dataset["assets"]["d37f8621bf6f327da7a213ec653ed59a"]['regions'][0]

# %%
json_dataset["assets"]["d37f8621bf6f327da7a213ec653ed59a"]["asset"]['size']['height']