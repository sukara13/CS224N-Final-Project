import modal

from modal import Image

inference_image = (
    Image.debian_slim(python_version="3.10")
    .pip_install("transformers", "torch")
)

app = modal.App("inference-car")
volume = modal.Volume.from_name("test", create_if_missing=True)

# Function to generate text
#@app.function(gpu="H100", image=inference_image)
@app.function(gpu=modal.gpu.A100(size="80GB", count=2), image=inference_image, volumes={"/datasets": volume})
def inference_car_test():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    #import os
    
    # Replace with your Hugging Face API key
    api_key = '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'

    # The model name
    model_name = 'google/gemma-2b-it'

    test_file = '/datasets/car_test.csv'
    eval_file = '/datasets/car_eval.csv'

    # Fetch latest changes
    #volume.reload()

    # List contents of the /datasets directory
    #print("Contents of /datasets directory:")
    #print(os.listdir('/datasets'))

    # Read test data
    #test_lines = []
    #with open(test_file, 'r') as fileTest:
    #    test_lines = fileTest.readlines()

    # Set up the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=api_key)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=api_key)

    #prompt = "You are an expert car evaluator. Given the attributes of a car, you will recommend whether the car is 'unacceptable', 'acceptable', 'good', or 'very good'. Here are the attributes of a car in text format: Buying Price is medium, Maintenance Cost is low, Doors are three, Persons are more than four, Trunk Size is small, Safety Score is low. Summarize your recommendation to buy this car in one or two words as 'unacceptable', 'acceptable', 'good', or 'very good' without providing any reasoning."
    prompt = "Once upon a time"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate the output
    outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
    
    # Decode the output to text
    llm_eval = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return llm_eval

    """
    count = 0
    success = 0
    content = ''
    all_content = ''
    with open(eval_file, 'w') as fileEval:
        for test_line in test_lines:
            field_list = test_line.split('\t')
            prompt = field_list[0]
            ground_truth = field_list[1]

            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate the output
            outputs = model.generate(inputs.input_ids, max_length=200, num_return_sequences=1)
            
            # Decode the output to text
            llm_eval = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if llm_eval == ground_truth:
                success += 1
            count += 1
            content = ground_truth + ',' + llm_eval + '\n'
            print(content)
            all_content += content
            fileEval.write(content)

    # Persist changes
    volume.commit()

    # Return results
    return all_content + '\n' + eval_file + ': ' + str(success) + '/' + str(count) + ' = ' + "{:.2%}".format(success / count)
    """

@app.local_entrypoint()
def main():
    #test_lines = []
    #with open('/Users/yilmazkara/Documents/CS224_Su/datasets/car_test.csv', 'r') as fileTest:
    #    test_lines = fileTest.readlines()

    output = inference_car_test.remote()

    print(output)
    
    