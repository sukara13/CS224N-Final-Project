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

"""
template_car_text = Template('The Buying price is ${buying}. ' \
                        'The Maintenance Costs are ${maint}. ' \
                        'The Doors are ${doors}. ' \
                        'The Persons are ${persons}. ' \
                        'The Trunk size is ${lug_boot}. ' \
                        'The Safety score is ${safety}. ')
template_car_text = Template('Buying Price: ${buying}, ' \
                        'Maintenance Cost: ${maint}, ' \
                        'Doors: ${doors}, ' \
                        'Persons: ${persons}, ' \
                        'Trunk Size: ${lug_boot}, ' \
                        'Safety Score: ${safety}. ')
"""

template_car_text = Template('\
Buying Price is ${buying}, \
Maintenance Cost is ${maint}, \
Doors are ${doors}, \
Persons are ${persons}, \
Trunk Size is ${lug_boot}, \
Safety Score is ${safety}. ')

gemma_user_prompt_text = 'You are an expert car evaluator. Here are the attributes of a car in text format: \
Buying Price is ${buying}, \
Maintenance Cost is ${maint}, \
Doors are ${doors}, \
Persons are ${persons}, \
Trunk Size is ${lug_boot}, \
Safety Score is ${safety}. \
Give your recommendation to buy this car as \'unacceptable\', \'acceptable\', \'good\', or \'very good\' without providing any explanation.'

template_car_qa_text = Template(gemma_user_prompt_text + '\t' + '${class}')

template_car_csv = Template('Buying Price, Maintenance Cost, Doors, Persons, Trunk Size, Safety Score\n\
${buying}, ${maint},  ${doors}, ${persons}, ${lug_boot}, ${safety}\n')

template_car_table = Template('\
<table>\n\
  <thead>\n\
    <tr>\n\
      <th>Buying Price</th>\n\
      <th>Maintenance Cost</th>\n\
      <th>Doors</th>\n\
      <th>Persons</th>\n\
      <th>Trunk Size</th>\n\
      <th>Safety Score</th>\n\
    </tr>\n\
  </thead>\n\
  <tbody>\n\
    <tr>\n\
      <td>${buying}</td>\n\
      <td>${maint}</td>\n\
      <td>${doors}</td>\n\
      <td>${persons}</td>\n\
      <td>${lug_boot}</td>\n\
      <td>${safety}</td>\n\
    </tr>\n\
  </tbody>\n\
</table>\n')

gemma_user_prompt_table = 'You are an expert car evaluator. Here are the attributes of a car in table format:|\
<table>|\
  <thead>|\
    <tr>|\
      <th>Buying Price</th>|\
      <th>Maintenance Cost</th>|\
      <th>Doors</th>|\
      <th>Persons</th>|\
      <th>Trunk Size</th>|\
      <th>Safety Score</th>|\
    </tr>|\
  </thead>|\
  <tbody>|\
    <tr>|\
      <td>${buying}</td>|\
      <td>${maint}</td>|\
      <td>${doors}</td>|\
      <td>${persons}</td>|\
      <td>${lug_boot}</td>|\
      <td>${safety}</td>|\
    </tr>|\
  </tbody>|\
</table>|\
Give your recommendation to buy this car as \'unacceptable\', \'acceptable\', \'good\', or \'very good\' without providing any explanation.'

template_car_qa_table = Template(gemma_user_prompt_table + '\t' + '${class}')

template_car_div = Template('\
<div>\n\
    <p>Buying Price: ${buying}</p>\n\
    <p>Maintenance Cost: ${maint}</p>\n\
    <p>Doors: ${doors}</p>\n\
    <p>Persons: ${persons}</p>\n\
    <p>Trunk Size: ${lug_boot}</p>\n\
    <p>Safety Score: ${safety}</p>\n\
</div>\n')

template_car_json = Template('\
{\n\
  "Buying Price": "${buying}",\n\
  "Maintenance Cost": "${maint}",\n\
  "Doors": "${doors}",\n\
  "Persons": "${persons}",\n\
  "Trunk Size": "${lug_boot}",\n\
  "Safety Score": "${safety}"\n\
}\n')

template_car_xml = Template('\
<Car>\n\
  <BuyingPrice>${buying}</BuyingPrice>\n\
  <MaintenanceCost>${maint}</MaintenanceCost>\n\
  <Doors>${doors}</Doors>\n\
  <Persons>${persons}</Persons>\n\
  <TrunkSize>${lug_boot}</TrunkSize>\n\
  <SafetyScore>${safety}</SafetyScore>\n\
</Car>\n')

template_car_yaml = Template('\
Car:\n\
  BuyingPrice: ${buying}\n\
  MaintenanceCost: ${maint}\n\
  Doors: ${doors}\n\
  Persons: ${persons}\n\
  TrunkSize: ${lug_boot}\n\
  SafetyScore: ${safety}\n')

# Define a function to apply to each row
def format_row(format, row):
    # Replace the values using the provided dictionaries
    formatted_values = {
        'buying': price_dict[row['buying']],
        'maint': price_dict[row['maint']],
        'doors': doors_dict[row['doors']],
        'persons': persons_dict[row['persons']],
        'lug_boot': lug_boot_dict[row['lug_boot']],
        'safety': safety_dict[row['safety']],
        'class': class_dict[row['class']]
    }

    # Use the template to substitute the values
    if format == 'text':
        return template_car_text.substitute(formatted_values)
    elif format == 'csv':
        return template_car_csv.substitute(formatted_values)
    elif format == 'table':
        return template_car_table.substitute(formatted_values)
    elif format == 'div':
        return template_car_div.substitute(formatted_values)
    elif format == 'json':
        return template_car_json.substitute(formatted_values)
    elif format == 'xml':
        return template_car_xml.substitute(formatted_values)
    elif format == 'yaml':
        return template_car_yaml.substitute(formatted_values)
    elif format == "qa-text":
        return template_car_qa_text.substitute(formatted_values)
    elif format == "qa-table":
        return template_car_qa_table.substitute(formatted_values)

# Create the Together AI client
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

filePath = '/Users/yilmazkara/Documents/CS224/'
carData = filePath + 'car.data'
modelName = 'google/gemma-7b-it' #'meta-llama/Llama-3-70b-chat-hf'

# Set the prompt to describe the attributes
prompt_attrib = "Every car has these attributes: BuyingPrice, MaintenanceCost, Doors, Persons, TrunkSize, SafetyScore. \
                BuyingPrice can be set to 'very high', 'high', 'medium', or 'low'. \
                MaintenanceCost can be set to 'very high', 'high', 'medium', or 'low'. \
                Doors can be set to 'two', 'three', 'four', or 'five or more'. \
                Persons can be set to 'two', 'four', or 'more than four'. \
                TrunkSize which can be set to 'big', 'medium', or 'small'. \
                SafetyScore can be set to 'high', 'medium', or 'low'. "

# Set the prompt for general rules
"""
*,*,*,2,*,*,unacc
*,*,*,*,*,low,unacc
vhigh,vhigh/high,*,*,*,unacc
vhigh,med/low,*,*,*,high,acc
high,vhigh,*,*,*,unacc
high,*,*,*,*,high,acc
med,vhigh/high,*,*,*,high,acc
med,med,*,*,*,med,acc
med,med,*,*,*,high,vgood
med,low,*,*,*,med,acc/good
med,low,*,*,*,high,good/vgood
low,vhigh,*,*,*,high,acc
low,high,*,*,*,med,acc
low,high,*,*,*,high,vgood
low,med/low,*,*,*,med,acc/good
low,med/low,*,*,*,high,good/vgood
"""
prompt_rules = "\
If Persons are 'two' or Safety Score is 'low', then Recommendation should be 'unacceptable'.\n\
If Buying Price is 'very high' and Maintenance Cost is 'high' or 'very high', then Recommendation should be 'unacceptable'.\n\
If Buying Price is 'high' and Maintenance Cost is 'very high', then Recommendation should be 'unacceptable'.\n\
If Buying Price is 'very high', Maintenance Cost is 'medium' or 'low', and Safety Score is 'high', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'high' and Safety Score is 'high', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'medium', Maintenance Cost is 'high' or 'very high', and Safety Score is 'high', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'medium', Maintenance Cost is 'medium', and Safety Score is 'medium', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'medium', Maintenance Cost is 'medium', and Safety Score is 'high', then Recommendation should be 'very good'.\n\
If Buying Price is 'medium', Maintenance Cost is 'low', and Safety Score is 'medium', then Recommendation should be 'acceptable' or 'good'.\n\
If Buying Price is 'medium', Maintenance Cost is 'low', and Safety Score is 'high', then Recommendation should be 'good' or 'very good'.\n\
If Buying Price is 'low', Maintenance Cost is 'very high', and Safety Score is 'high', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'low', Maintenance Cost is 'high', and Safety Score is 'medium', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'low', Maintenance Cost is 'high', and Safety Score is 'high', then Recommendation should be 'very good'.\n\
If Buying Price is 'low', Maintenance Cost is 'low' or 'medium', and Safety Score is 'medium', then Recommendation should be 'acceptable' or 'good'.\n\
If Buying Price is 'low', Maintenance Cost is 'low' or 'medium', and Safety Score is 'high', then Recommendation should be 'good' or 'very good'.\n\n"

# Set the prompt for some examples
"""
med,low,2,4,small,med,acc
med,low,2,4,small,high,good
med,low,2,more,big,med,good
med,low,2,more,big,high,vgood
low,med,2,4,small,med,acc
low,med,2,4,small,high,good
low,med,2,more,big,med,good
low,med,2,more,big,high,vgood
low,low,2,4,small,med,acc
low,low,2,4,small,high,good
low,low,2,more,big,med,good
low,low,2,more,big,high,vgood
"""

prompt_examples = "\
If Buying Price is 'medium', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'medium', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'medium', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'high', then Recommendation should be 'good'.\n\
If Buying Price is 'medium', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'medium', then Recommendation should be 'good'.\n\
If Buying Price is 'medium', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'high', then Recommendation should be 'very good'.\n\
If Buying Price is 'low', Maintenance Cost is 'medium', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'medium', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'low', Maintenance Cost is 'medium', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'high', then Recommendation should be 'good'.\n\
If Buying Price is 'low', Maintenance Cost is 'medium', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'medium', then Recommendation should be 'good'.\n\
If Buying Price is 'low', Maintenance Cost is 'medium', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'high', then Recommendation should be 'very good'.\n\
If Buying Price is 'low', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'medium', then Recommendation should be 'acceptable'.\n\
If Buying Price is 'low', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'four', Trunk Size is 'small', and Safety Score is 'high', then Recommendation should be 'good'.\n\
If Buying Price is 'low', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'medium', then Recommendation should be 'good'.\n\
If Buying Price is 'low', Maintenance Cost is 'low', 'Doors' is 'two', Persons is 'more than four', Trunk Size is 'big', and Safety Score is 'high', then Recommendation should be 'very good'.\n\n"

def generate_train_test_data(format, train_count):
    # Read the car data into a pandas data frame
    df = pd.read_csv(carData, names=[ 'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class' ])

    # Apply the format function to each row to add a new column as description
    # contains template filled in with substitued attributes for each row
    # now only need ot look at description & class which you will check against
    df['qa_description'] = df.apply(lambda row: format_row('qa-' + format, row), axis=1)

    # Shuffle the rows of the data frame
    df = df.sample(frac=1, random_state=1)

    df_train = df.head(train_count)
    df_train[['qa_description']].to_csv(filePath + 'car_' + format + '_train_top_' + str(train_count) + '.csv', index=False, header=False)
    df_test = df.iloc[train_count:]
    df_test[['qa_description']].to_csv(filePath + 'car_' + format + '_test_wo_' + str(train_count) + '.csv', index=False, header=False)

#generate_train_test_data('text', 16)
#generate_train_test_data('text', 128)
#generate_train_test_data('text', 512)
#generate_train_test_data('table', 16)
#generate_train_test_data('table', 128)
#generate_train_test_data('table', 512)

"""
Arguments:
    format: text, csv, table, div, json, xml, yaml
    shuffle: true or false to shuffle the rows
    shot: shot count for zero or few shot training
"""
def evaluateCar(format, shot=0, attrib='', rules='', examples='', prompt='system', finetune_base='sukara13/gemma7bcars', finetune_train=0):
    # Set counters
    count = 0
    success = 0

    # Read the car data into a pandas data frame
    df = pd.read_csv(carData, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    # Apply the format function to each row to add a new column as description
    df['description'] = df.apply(lambda row: format_row(format, row), axis=1)

    # Shuffle the rows of the data frame
    df = df.sample(frac=1, random_state=1)

    # Check the shot count for zero or few shot training
    few_shots = ''
    if shot > 0:
        few_shots = 'Here are ' + str(shot) + ' examples showing the attributes of a car followed by a recommendation in ' + format + ' format:\n'
        dfShot = df.head(shot)
        exCount = 0
        for index, row in dfShot.iterrows():
            if format == 'table':
                if exCount == 0:
                    few_shots += row['description']
                    few_shots = few_shots.replace('<th>Safety Score</th>\n', '<th>Safety Score</th>\n      <th>Recommendation</th>\n')
                    few_shots = few_shots.replace('</tr>\n  </tbody>', '  <td>' + class_dict[row['class']] + '</td>\n    </tr>\n  </tbody>')
                else:
                    tbody_start = row['description'].find("<tbody>")
                    tr_start = row['description'].find("<tr>", tbody_start)
                    tr_end = row['description'].find("</tr>", tr_start) + len("</tr>")
                    new_tr = row['description'][tr_start:tr_end]
                    new_tr = new_tr.replace('</tr>', '  <td>' + class_dict[row['class']] + '</td>\n    </tr>')
                    few_shots = few_shots.replace('</tbody>', '  ' + new_tr + '\n  </tbody>')
            else:
                few_shots += 'Car ' + str(exCount + 1) + ':\n' + row['description'] + 'Recommendation is ' + class_dict[row['class']] + '.\n\n'
            exCount += 1

    if finetune_train > 0:
        modelName = finetune_base + str(finetune_train)
    fileName = filePath + modelName.replace('/', '_') + '-' + format + '-'
    if attrib != '':
        fileName += 'attrib-'
    if rules != '':
        fileName += 'rules-'
    if examples != '':
        fileName += 'examples-'
    if prompt != '':
        fileName += 'user-'
    if finetune_train > 0:
        fileName += 'finetune-'
    fileName += str(shot) + '.csv'

    with open(fileName, 'w') as file:
        dfTest = df.iloc[shot:]
        if finetune_train > 0:
            dfTest = df.iloc[finetune_train:]
        for index, row in dfTest.iterrows():
            prompt_system = "You are an expert car evaluator. Given the attributes of a car, you will recommend whether the car is 'unacceptable', 'acceptable', 'good', or 'very good'.\n\n"
            prompt_user = "Here are the attributes of a car in " + format + " format:\n" + row['description'] + "Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."
            if prompt == "system":
                prompt_system =  prompt_system + attrib + rules + examples + few_shots
            else:
                prompt_user =  attrib + rules + examples + few_shots + prompt_user

            response = client.chat.completions.create(
                model=modelName,
                temperature=0,
                messages=[{"role": "system", "content": prompt_system},
                          {"role": "user", "content":  prompt_user}]
            )

            llm_eval = response.choices[0].message.content.lower().replace('"', '')
            if 'unacceptable' in llm_eval:
                llm_eval = 'unacceptable'
            elif 'acceptable' in llm_eval:
                llm_eval = 'acceptable'
            elif 'very good' in llm_eval:
                llm_eval = 'very good'
            elif 'good' in llm_eval:
                llm_eval = 'good'
            else:
                llm_eval = 'fail: ' + row['description'] + llm_eval
            ground_truth = class_dict[row['class']]
            if llm_eval == ground_truth:
                success += 1
            count += 1
            file.write(ground_truth + ',' + llm_eval + '\n')

    print(fileName + ': ' + str(success) + '/' + str(count) + ' = ' + "{:.2%}".format(success/count))

"""
evaluateCar('text', 0)
evaluateCar('csv', 0)
evaluateCar('table', 0)
evaluateCar('div', 0)
evaluateCar('json', 0)
evaluateCar('xml', 0)
evaluateCar('yaml', 0)
evaluateCar('text', 1)
evaluateCar('text', 2)
evaluateCar('text', 4)
evaluateCar('text', 8)
evaluateCar('text', 16)
evaluateCar('text', 32)
evaluateCar('text', 64)
evaluateCar('text', 128)
evaluateCar('table', 1)
evaluateCar('table', 2)
evaluateCar('table', 4)
evaluateCar('table', 8)
evaluateCar('table', 16)
evaluateCar('table', 32)
evaluateCar('table', 64)
evaluateCar('table', 100)
"""

#evaluateCar('text', rules=prompt_rules)
#evaluateCar('table', rules=prompt_rules)
#evaluateCar('text', shot=1)
#evaluateCar('csv', shot=1)
#evaluateCar('table', shot=1)
#evaluateCar('div', shot=1)
#evaluateCar('json', shot=1)
#evaluateCar('xml', shot=1)
#evaluateCar('yaml', shot=1)
#evaluateCar('text', shot=1, prompt='user')
#evaluateCar('table', shot=1, prompt='user')
#evaluateCar('text', shot=2, prompt='user')
#evaluateCar('table', shot=2, prompt='user')
#evaluateCar('text', shot=4, prompt='user')
#evaluateCar('table', shot=4, prompt='user')
#evaluateCar('text', shot=8, prompt='user')
#evaluateCar('table', shot=8, prompt='user')
#evaluateCar('text', shot=16, prompt='user')
#evaluateCar('table', shot=16, prompt='user')
#evaluateCar('text', shot=32, prompt='user')
#evaluateCar('table', shot=32, prompt='user')
#evaluateCar('text', shot=64, prompt='user')
#evaluateCar('table', shot=64, prompt='user')
#evaluateCar('text', shot=128, prompt='user')
#evaluateCar('table', shot=128, prompt='user')
#evaluateCar('text', finetune_train=128)

"""
Llama-3-70b-chat-hf-text-False-0.csv: 1189/1728 = 68.81%, 46 min, 17 cents
Llama-3-70b-chat-hf-text-True-0.csv: 1189/1728 = 68.81% , 44 min, 18 cents
Llama-3-70b-chat-hf-text-col-True-0.csv: 1151/1728 = 66.61%, 45 min, 18 cents
Llama-3-70b-chat-hf-text-col-upp-True-0.csv: 1172/1728 = 67.82%, 46 min, 17 cents
prompt_attrib was less accurate

#start
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-0.csv: 1127/1728 = 65.22%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-csv-0.csv: 1109/1728 = 64.18%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-0.csv: 1236/1728 = 71.53% ***
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-div-0.csv: 1157/1728 = 66.96%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-json-0.csv: 1080/1728 = 62.50%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-xml-0.csv: 1149/1728 = 66.49%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-yaml-0.csv: 1124/1728 = 65.05%
#end

/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-1.csv: 1119/1727 = 64.79%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-2.csv: 891/1726 = 51.62%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-4.csv: 960/1724 = 55.68%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-8.csv: 930/1720 = 54.07%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-16.csv: 1128/1712 = 65.89%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-32.csv: 1139/1696 = 67.16%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-64.csv: 1092/1664 = 65.62%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-128.csv: 1185/1600 = 74.06%
256 shot: Error code: 400 - {"message": "Input validation error: `inputs` tokens + 
    `max_new_tokens` must be <= 8193. Given: 10160 `inputs` tokens and 1 `max_new_tokens`", 
    "type_": "invalid_request_error", "param": "max_tokens", "code": null}
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-csv-1.csv: 953/1727 = 55.18%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-1.csv: 1164/1727 = 67.40%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-2.csv: 850/1726 = 49.25%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-4.csv: 1103/1724 = 63.98%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-8.csv: 954/1720 = 55.47%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-16.csv: 1059/1712 = 61.86%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-32.csv: 1059/1696 = 62.44%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-64.csv: 1022/1664 = 61.42%
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-100.csv: 1035/1628 = 63.57%
128 shot: Error code: 400 - {"message": "Input validation error: `inputs` tokens + 
    `max_new_tokens` must be <= 8193. Given: 8866 `inputs` tokens and 1 `max_new_tokens`", 
    "type_": "invalid_request_error", "param": "max_tokens", "code": null}

/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-rules-0.csv: 1407/1728 = 81.42%, 49 min, 87 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-0.csv: 1455/1728 = 84.20%, 49 min, 104 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-user-0.csv: 1335/1728 = 77.26%, 49 min, 104 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-examples-0.csv: 1291/1728 = 74.71%, 47 194 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-cost-0.csv: 1383/1728 = 80.03%, 49 min, 105 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-price-0.csv: 1385/1728 = 80.15%, 55 min, 104 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-cost-price-0.csv: 1394/1728 = 80.67%, 47 min, 104 cents
/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-table-rules-sys-ex-usr-no-shuffle-0.csv: 1386/1728 = 80.21%,47 min, 173 cents

/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-rules-0.csv: 1167/1728 = 67.53%, 41 min, 20 cents
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-rules-0.csv: 1224/1728 = 70.83%, 40 min, 21 cents
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-1.csv: 1146/1727 = 66.36%, 44 min, 9 cents
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-1.csv: 462/1727 = 26.75%, 53 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-csv-1.csv: 1073/1727 = 62.13%, 46 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-div-1.csv: 951/1727 = 55.07%, 73 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-json-1.csv: 1133/1727 = 65.61%, 56 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-xml-1.csv: incomplete, extremely slow and sometimes fails
/Users/yilmazkara/Documents/CS224/gemma-7b-it-yaml-1.csv: 1045/1727 = 60.51%, 53 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-1.csv: 1154/1727 = 66.82%, 65 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-1.csv: 614/1727 = 35.55%, 49 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-2.csv: 1106/1726 = 64.08%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-2.csv: 630/1726 = 36.50%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-4.csv: 1204/1724 = 69.84%, 42 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-4.csv: 1059/1724 = 61.43%, 46 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-8.csv: 1181/1720 = 68.66%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-8.csv: 1146/1720 = 66.63%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-16.csv: 1212/1712 = 70.79%, 38 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-16.csv: 1198/1712 = 69.98%, 41 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-32.csv: 1183/1696 = 69.75%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-32.csv: 1187/1696 = 69.99%
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-64.csv: 1041/1664 = 62.56%, 89 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-64.csv: 1158/1664 = 69.59%, 36 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-text-user-128.csv: 1116/1600 = 69.75%, 35 min
/Users/yilmazkara/Documents/CS224/gemma-7b-it-table-user-128.csv: 1113/1600 = 69.56%, 45 min
"""

"""
# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-text-True-0.csv', header=None, names=['Value1', 'Value2'])
#df = pd.read_csv('/Users/yilmazkara/Documents/CS224/Llama-3-70b-chat-hf-yaml-True-0.csv', header=None, names=['Value1', 'Value2'])

df = df.head(1066)

# Check whether the values in each row are the same and store the result in a new column
df['Same'] = df['Value1'] == df['Value2']

num_true = df['Same'].sum()

print(len(df))
print(num_true)
"""

"""
# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/yilmazkara/Documents/CS224/sukara13_gemma7bcars16-text.csv', header=None, names=['Value1', 'Value2'])

# Check whether the values in each row are the same and store the result in a new column
df['Same'] = df['Value1'] == df['Value2']

num_true = df['Same'].sum()

print(len(df))
print(num_true)
"""

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
./datasets/sukara13_gemma7bcars16-text.csv: 1042/1712 = 60.86%
./datasets/sukara13_gemma7bcars128-text.csv: 1375/1600 = 85.94%
./datasets/sukara13_gemma7bcars512-text.csv: 1107/1216 = 91.04%
"""