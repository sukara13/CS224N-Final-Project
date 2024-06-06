import os
from together import Together
from string import Template
import pandas as pd

"""
 CAR                      car acceptability
   . PRICE                  overall price
   . . buying               buying price
   . . maint                price of the maintenance
   . TECH                   technical characteristics
   . . COMFORT              comfort
   . . . doors              number of doors
   . . . persons            capacity in terms of persons to carry
   . . . lug_boot           the size of luggage boot
   . . safety               estimated safety of the car

Number of Instances: 1728

Number of Attributes: 6

Attribute Values:
   buying       vhigh, high, med, low
   maint        vhigh, high, med, low
   doors        2, 3, 4, 5more
   persons      2, 4, more
   lug_boot     small, med, big
   safety       low, med, high

Class Distribution:
   class      N          N[%]
   -----------------------------
   unacc     1210     (70.023 %) 
   acc        384     (22.222 %) 
   good        69     ( 3.993 %) 
   vgood       65     ( 3.762 %) 
"""

price_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
doors_dict = {'2': 'two', '3': 'three', '4': 'four', '5more': 'five or more'}
persons_dict = {'2': 'two', '4': 'four', 'more': 'more than four'}
lug_boot_dict = {'big': 'big', 'med': 'medium', 'small': 'small'}
safety_dict = {'high': 'high', 'med': 'medium', 'low': 'low'}
class_dict = {'unacc': 'unacceptable', 'acc': 'acceptable', 'good': 'good', 'vgood': 'very good'}

template_car = Template('The Buying price is ${buying}. ' \
                        'The Maintenance costs are ${maint}. ' \
                        'The Doors are ${doors}. ' \
                        'The Persons are ${persons}. ' \
                        'The Trunk size is ${lug_boot}. ' \
                        'The Safety score is ${safety}. ')

"""
template_car = Template('The purchase price is ${buying}. ' \
               'The maintenance cost is ${maint}. ' \
               'The number of doors is ${doors}. ' \
               'The number of people to carry is ${persons}. ' \
               'The trunk size is ${lug_boot}. ' \
               'The safety score is ${safety}. ')
"""

template_car_json = Template('\
{\
  "Buying Price": "${buying}",\
  "Maintenance Cost": "${maint}",\
  "Doors": "${doors}",\
  "Persons": "${persons}",\
  "Trunk Size": "${lug_boot}",\
  "Safety Score": "${safety}"\
}')

template_car_xml = Template('\
<Car>\
  <BuyingPrice>${buying}</BuyingPrice>\
  <MaintenanceCost>${maint}</MaintenanceCost>\
  <Doors>${doors}</Doors>\
  <Persons>${persons}</Persons>\
  <TrunkSize>${lug_boot}</TrunkSize>\
  <SafetyScore>${safety}</SafetyScore>\
</Car>')

template_car_yaml = Template('\
Car:\n\
  BuyingPrice: ${buying}\n\
  MaintenanceCost: ${maint}\n\
  Doors: ${doors}\n\
  Persons: ${persons}\n\
  TrunkSize: ${lug_boot}\n\
  SafetyScore: ${safety}')

"""
Cars:
  - BuyingPrice: ${buying1}
    MaintenanceCost: ${maint1}
    Doors: ${doors1}
    Persons: ${persons1}
    TrunkSize: ${lug_boot1}
    SafetyScore: ${safety1}
  - BuyingPrice: ${buying2}
    MaintenanceCost: ${maint2}
    Doors: ${doors2}
    Persons: ${persons2}
    TrunkSize: ${lug_boot2}
    SafetyScore: ${safety2}
"""

template_car_html_table = Template('\
<table>\
  <thead>\
    <tr>\
      <th>Buying Price</th>\
      <th>Maintenance Cost</th>\
      <th>Doors</th>\
      <th>Persons</th>\
      <th>Trunk Size</th>\
      <th>Safety Score</th>\
    </tr>\
  </thead>\
  <tbody>\
    <tr>\
      <td>${buying}</td>\
      <td>${maint}</td>\
      <td>${doors}</td>\
      <td>${persons}</td>\
      <td>${lug_boot}</td>\
      <td>${safety}</td>\
    </tr>\
  </tbody>\
</table>')

template_car_html_table_2 = Template('<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>Buying Price</th>\n      <th>Maintenance Cost</th>\n      <th>Doors</th>\n      <th>Persons</th>\n      <th>Trunk Size</th>\n      <th>Safety Score</th>\n      <th>Recommendation</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>${buying}</td>\n      <td>${maint}</td>\n      <td>${doors}</td>\n      <td>${persons}</td>\n      <td>${lug_boot}</td>\n      <td>${safety}</td>\n      <td></td>\n    </tr>\n    </tbody>\n</table>')
car_html_table_one_shot = Template('<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th>Buying Price</th>\n      <th>Maintenance Cost</th>\n      <th>Doors</th>\n      <th>Persons</th>\n      <th>Trunk Size</th>\n      <th>Safety Score</th>\n      <th>Recommendation</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <td>very high</td>\n      <td>very high</td>\n      <td>five or more</td>\n      <td>more than four</td>\n      <td>big</td>\n      <td>high</td>\n      <td>unacceptable</td>\n    </tr>\n    </tbody>\n</table>')

template_car_html_div = Template('\
<div>\
    <div>\
        <div>Buying Price</div>\
        <div>Maintenance Cost</div>\
        <div>Doors</div>\
        <div>Persons</div>\
        <div>Trunk Size</div>\
        <div>Safety Score</div>\
    </div>\
    <div>\
      <div>${buying}</div>\
      <div>${maint}</div>\
      <div>${doors}</div>\
      <div>${persons}</div>\
      <div>${lug_boot}</div>\
      <div>${safety}</div>\
    </div>\
</div>')

df = pd.read_csv('/Users/yilmazkara/Documents/CS224/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

# Define a function to apply to each row
def format_row(row):
    # Replace the values using the provided dictionaries
    formatted_values = {
        'buying': price_dict[row['buying']],
        'maint': price_dict[row['maint']],
        'doors': doors_dict[row['doors']],
        'persons': persons_dict[row['persons']],
        'lug_boot': lug_boot_dict[row['lug_boot']],
        'safety': safety_dict[row['safety']]
    }
    # Use the template to substitute the values
    #return template_car.substitute(formatted_values)
    #return template_car_html_table.substitute(formatted_values)
    #return template_car_html_div.substitute(formatted_values)
    #return template_car_json.substitute(formatted_values)
    #return template_car_xml.substitute(formatted_values)
    return template_car_yaml.substitute(formatted_values)

# Apply the function to each row to add a new column as description
df['description'] = df.apply(format_row, axis=1)

"""
# Apply the dictionary mappings directly to the DataFrame
df['buying'] = df['buying'].map(price_dict)
df['maint'] = df['maint'].map(price_dict)
df['doors'] = df['doors'].map(doors_dict)
df['persons'] = df['persons'].map(persons_dict)
df['lug_boot'] = df['lug_boot'].map(lug_boot_dict)
df['safety'] = df['safety'].map(safety_dict)

# Shuffle the DataFrame and reset the index
shuffled_df = df.sample(frac=1, random_state=1)

# Copy the DataFrame for temporary manipulation
temp_df = shuffled_df.copy()
#temp_df = df.copy()

# Apply masking
temp_df['class'] = ''

# Change column names
new_columns = ['Buying Price', 'Maintenance Cost', 'Doors', 'Persons', 'Trunk Size', 'Safety Score', 'Recommendation']
#new_columns = ['Purchase Price', 'Maintenance Cost', 'Number of Doors', 'Number of People to Carry', 'Trunk Size', 'Safety Score', 'Recommendation']
temp_df.columns = new_columns

# Set batch size
batch_size = 12

# Get number of rows
num_rows = len(df)
"""

# Create the Together AI client
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# Set counters
count = 0
success = 0

price_dict = {'vhigh': 'very high', 'high': 'high', 'med': 'medium', 'low': 'low'}
doors_dict = {'2': 'two', '3': 'three', '4': 'four', '5more': 'five or more'}
persons_dict = {'2': 'two', '4': 'four', 'more': 'more than four'}
lug_boot_dict = {'big': 'big', 'med': 'medium', 'small': 'small'}
safety_dict = {'high': 'high', 'med': 'medium', 'low': 'low'}
prompt_attrib = "Every car has these attributes: BuyingPrice, MaintenanceCost, Doors, Persons, TrunkSize, SafetyScore. \
                BuyingPrice can be set to 'very high', 'high', 'medium', or 'low'. \
                MaintenanceCost can be set to 'very high', 'high', 'medium', or 'low'. \
                Doors can be set to 'two', 'three', 'four', or 'five or more'. \
                Persons can be set to 'two', 'four', or 'more than four'. \
                TrunkSize which can be set to 'big', 'medium', or 'small'. \
                SafetyScore can be set to 'high', 'medium', or 'low'."
one_shot = "Car:\
                BuyingPrice: very high\
                MaintenanceCost: very high\
                Doors: five or more\
                Persons: more than four\
                TrunkSize: big\
                SafetyScore: high\
                Recommendation: unacceptable"
two_shot = "Car:\
                BuyingPrice: very high\
                MaintenanceCost: very high\
                Doors: five or more\
                Persons: more than four\
                TrunkSize: big\
                SafetyScore: high\
                Recommendation: unacceptable\
            Car:\
                BuyingPrice: very high\
                MaintenanceCost: medium\
                Doors: two\
                Persons: four\
                TrunkSize: small\
                SafetyScore: high\
                Recommendation: acceptable"

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-yaml-two-shot.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            #top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "system", "content": two_shot},
                      {"role": "user", "content": "Here are the attributes of a car in YAML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-yaml-prompt-user.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": prompt_attrib},
                      {"role": "user", "content": "Here are the attributes of a car in YAML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-yaml-prompt-system.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "system", "content": prompt_attrib},
                      {"role": "user", "content": "Here are the attributes of a car in YAML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-yaml-shuffle.csv', 'w') as file:
    # Shuffle the DataFrame and reset the index
    df = df.sample(frac=1, random_state=1)

    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here are the attributes of a car in YAML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-yaml.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here are the attributes of a car in YAML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-xml.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here are the attributes of a car in XML format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-json.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here are the attributes of a car in JSON format: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-div.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here is a nested HTML div containing the attributes of a car with the column header in the first child div and the values in the second child div: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-batch-table-' + str(batch_size) + '.csv', 'w') as file:
    while count < num_rows:
        # Check remaining rows
        if batch_size > num_rows - count:
            batch_size = num_rows - count

        # Get the next batch
        batch = temp_df.iloc[count:count+batch_size]

        # Convert the selected rows to an HTML table
        html_table = batch.to_html(index=False)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here is an HTML table containing the attributes of a car in each row: " + html_table + "For each row, summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning. Return the same HTML table by filling in the Recommendation column for each row."}]
        )
        llm_eval = response.choices[0].message.content.lower().replace('"', '').replace("here is the updated html table with my recommendations:\n\n", '')
        dfs = pd.read_html(llm_eval)
        html_df = dfs[0]

        if len(html_df) != batch_size:
            print(batch_size, len(html_df))
            break

        for i in range(batch_size):
            ground_truth = class_dict[shuffled_df.iloc[count + i]['class']]
            rec = html_df.iloc[i]['recommendation']

            file.write(ground_truth + ',' + rec + '\n')
            if rec == ground_truth:
                success += 1
        count += batch_size
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-batch-list-' + str(batch_size) + '.csv', 'w') as file:
    while count < num_rows:
        # Check remaining rows
        if batch_size > num_rows - count:
            batch_size = num_rows - count

        # Get the next batch
        batch = temp_df.iloc[count:count+batch_size]

        # Convert the selected rows to an HTML table
        html_table = batch.to_html(index=False)

        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      #{"role": "system", "content": two_shot},
                      #{"role": "system", "content": prompt_attrib},
                      {"role": "user", "content": "Here is an HTML table containing the attributes of a car in each row: " + html_table + "For each row, summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning. Return a comma delimited list of recommendations, one for each row. Make sure you return " + str(batch_size) + " items in that list."}]
        )
        llm_eval = response.choices[0].message.content.lower().replace('"', '').replace("here is the list of recommendations:\n\n", '')
        rec_list = llm_eval.split(', ')
        if len(rec_list) != batch_size:
            print(batch_size, len(rec_list))
            break

        for i in range(batch_size):
            ground_truth = class_dict[shuffled_df.iloc[count + i]['class']]
            file.write(ground_truth + ',' + rec_list[i] + '\n')
            if rec_list[i] == ground_truth:
                success += 1
        count += batch_size
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-table-shuffle.csv', 'w') as file:
    # Shuffle the DataFrame and reset the index
    shuffled_df = df.sample(frac=1, random_state=1)

    for index, row in shuffled_df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here is an HTML table containing the attributes of a car in each row: " + row['description'] +  "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-table.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here is an HTML table containing the attributes of a car in each row: " + row['description'] +  "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

"""
with open('/Users/yilmazkara/Documents/CS224/llm-res-Llama-3-70b-chat-hf-html-tabllm.csv', 'w') as file:
    for index, row in df.iterrows():
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            top_k=4,
            messages=[{"role": "system", "content": "You are a helpful AI assistant to evaluate cars based on several attribues and make a recommendation."},
                      {"role": "user", "content": "Here are the attributes of a car: " + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."}]
        )

        llm_eval = response.choices[0].message.content.lower().replace('"', '')
        ground_truth = class_dict[row['class']]
        if llm_eval == ground_truth:
            success += 1
        count += 1
        file.write(ground_truth + ',' + llm_eval + '\n')
"""

print(count)
print(success)

"""
zero-shot
llm-res-Llama-3-8b-chat-hf-tabllm: 991/1728 = 57% (1 hr, 4 cents)
llm-res-Llama-3-70b-chat-hf-tabllm: 1177/1728 = 68% (49 min, 18 cents)
llm-res-Llama-3-70b-chat-hf-html-table: 1166/1728 = 67% (46 min, 34 cents)
llm-res-Llama-3-70b-chat-hf-html-table-shuffle: 1147/1728 = 66% (48 min, 19 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-1-noshuffle: 806/1728 = 47% (48 min, 44 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-1: 1129/1728 = 65% (48 min, 44 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-2: 909/1728 = 53% (23 min, 27 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-8: 975/1728 = 56% (5 min, 15 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-10: 967/1728 = 56% (4 min, 14 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-12: 902/1728 = 52% (3 min, 13 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-table-10: 893/1728 = 52% (26 min, 25 cents with a batch size of 10)
llm-res-Llama-3-70b-chat-hf-html-div: 824/1728 = 48% (48 min, 35 cents)
llm-res-Llama-3-70b-chat-hf-json: 1142/1728 = 66% (48 min, 21 cents)
llm-res-Llama-3-70b-chat-hf-xml: 1046/1728 = 61% (46 min, 23 cents)
llm-res-Llama-3-70b-chat-hf-yaml: 1181/1728 = 68% (48 min, 18 cents)
llm-res-Llama-3-70b-chat-hf-yaml-shuffle: 1151/1728 = 67% (48 min, 19 cents)

prompt engineering
llm-res-Llama-3-70b-chat-hf-yaml-prompt-system: 878/1728 = 51% (47 min, 48 cents)
llm-res-Llama-3-70b-chat-hf-yaml-prompt-user: 872/1728 = 50% (47 min, 48 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-8-prompt-system: 961/1728 = %56 (5 min, 17 cents)
llm-res-Llama-3-70b-chat-hf-yaml-one-shot: 1174/1728 = 68% (49 min, 25 cents)
llm-res-Llama-3-70b-chat-hf-yaml-two-shot: 1130/1728 = 68% (47 min, 31 cents)
llm-res-Llama-3-70b-chat-hf-html-batch-list-4-two-shot: 1076/1728 = 62% (12 min, 22 cents)

few-shot/in-context-learning/ICL (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
fine tuning
"""