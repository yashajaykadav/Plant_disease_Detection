import json

# Load disease information from JSON
with open('Information.json', 'r') as file:
    desInfo = json.load(file)

def getInfo(disease_name):
    # Check if the plant and disease names are valid
    try:
        disease_info = desInfo[disease_name]
        description = disease_info.get('Description', "No Information Available")
        solution = disease_info.get('Solution', "No Solution Available")
        return {"Description": description, "Solution": solution}
    except KeyError:
        return {"Description": "Plant type or disease name not found.", "Solution": "No Solution Available."}
