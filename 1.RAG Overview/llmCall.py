from utils import (
    generate_with_single_input, 
    generate_with_multiple_input,
    get_proxy_url,
    get_proxy_headers,
    get_together_key
)

# Example call
output = generate_with_single_input(
    prompt="What is the capital of France?"
)

print("Role:", output['role'])
print("Content:", output['content'])

# Example call
messages = [
    {'role': 'user', 'content': 'Hello, who won the FIFA world cup in 2018?'},
    {'role': 'assistant', 'content': 'France won the 2018 FIFA World Cup.'},
    {'role': 'user', 'content': 'Who was the captain?'}
]

output = generate_with_multiple_input(
    messages=messages,
    max_tokens=100
)

print("Role:", output['role'])
print("Content:", output['content'])

from openai import OpenAI, DefaultHttpxClient
import httpx

base_url = get_proxy_url() # If using together endpoint, add it here https://api.together.xyz/

# Custom transport to bypass SSL verification. This is only needed if using our proxy. Otherwise you can ignore it.
transport = httpx.HTTPTransport(local_address="0.0.0.0", verify=False)

# Create a DefaultHttpxClient instance with the custom transport
http_client = DefaultHttpxClient(transport=transport, headers=get_proxy_headers())

client = OpenAI(
    api_key = get_together_key(), # Set any as our proxy does not use it. Set the together api key if using the together endpoint.
    base_url=base_url, 
    http_client=http_client, # ssl bypass to make it work via proxy calls, remove it if running with together.ai endpoint 
)

messages = [
    {'role': 'user', 'content': 'Hello, who won the FIFA world cup in 2018?'},
    {'role': 'assistant', 'content': 'France won the 2018 FIFA World Cup.'},
    {'role': 'user', 'content': 'Who was the captain?'}
]

response = client.chat.completions.create(messages = messages, model ="Qwen/Qwen3.5-9B", extra_body={
        "reasoning": False
    })

print(response)

print(response.choices[0].message.content)

house_data = [
    {
        "address": "123 Maple Street",
        "city": "Springfield",
        "state": "IL",
        "zip": "62701",
        "bedrooms": 3,
        "bathrooms": 2,
        "square_feet": 1500,
        "price": 230000,
        "year_built": 1998
    },
    {
        "address": "456 Elm Avenue",
        "city": "Shelbyville",
        "state": "TN",
        "zip": "37160",
        "bedrooms": 4,
        "bathrooms": 3,
        "square_feet": 2500,
        "price": 320000,
        "year_built": 2005
    }
]

# First, let's create a layout for the houses

def house_info_layout(houses):
    # Create an empty string
    layout = ''
    # Iterate over the houses
    for house in houses:
        # For each house, append the information to the string using f-strings
        # The following way using brackets is a good way to make the code readable as in each line you can start a new f-string that will appended to the previous one
        layout += (f"House located at {house['address']}, {house['city']}, {house['state']} {house['zip']} with "
            f"{house['bedrooms']} bedrooms, {house['bathrooms']} bathrooms, "
            f"{house['square_feet']} sq ft area, priced at ${house['price']}, "
            f"built in {house['year_built']}.\n") # Don't forget the new line character at the end!
    return layout

# Check the layout
print(house_info_layout(house_data))

def generate_prompt(query, houses):
    # The code made above is modular enough to accept any list of houses, so you could also choose a subset of the dataset.
    # This might be useful in a more complex context where you want to give only some information to the LLM and not the entire data
    houses_layout = house_info_layout(houses)
    # Create a hard-coded prompt. You can use three double quotes (") in this cases, so you don't need to worry too much about using single or double quotes and breaking the code
    PROMPT = f"""
Use the following houses information to answer users queries.
{houses_layout}
Query: {query}    
             """
    return PROMPT

print(generate_prompt("What is the most expensive house?", houses = house_data))

# Now we can call the LLM with the generated prompt
query = "What is the most expensive house? And the bigger one?"
# Asking without the augmented prompt, let's pass the role as user
query_without_house_info = generate_with_single_input(prompt = query, role = 'user')
# With house info, given the prompt structuer, let's pass the role as assistant
enhanced_query = generate_prompt(query, houses = house_data)
query_with_house_info = generate_with_single_input(prompt = enhanced_query, role = 'assistant')

# Without house info
print(query_without_house_info['content'])

# With house info
print(query_with_house_info['content'])