import os
import requests  # Make sure requests is installed
import json
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]
            variable_name_parts = variable_name_parts[1:]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def load_weights_into_gpt2(model, params):
    # Weight tie-ing: the token embeddings and output head weights are the same
    model.tok_emb.weight = torch.nn.Parameter(torch.from_numpy(params['wte']))
    model.pos_emb.weight = torch.nn.Parameter(torch.from_numpy(params['wpe']))

    for i, block in enumerate(model.trf_blocks):
        # LayerNorm 1
        block.norm1.scale = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['ln_1']['g']))
        block.norm1.shift = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['ln_1']['b']))

        # Attention
        w_query, w_key, w_value = np.split(params['blocks'][i]['attn']['c_attn']['w'], 3, axis=-1)
        block.att.W_query.weight = torch.nn.Parameter(torch.from_numpy(w_query.T))
        block.att.W_key.weight = torch.nn.Parameter(torch.from_numpy(w_key.T))
        block.att.W_value.weight = torch.nn.Parameter(torch.from_numpy(w_value.T))

        if block.att.W_query.bias is not None:
            b_query, b_key, b_value = np.split(params['blocks'][i]['attn']['c_attn']['b'], 3, axis=-1)
            block.att.W_query.bias = torch.nn.Parameter(torch.from_numpy(b_query))
            block.att.W_key.bias = torch.nn.Parameter(torch.from_numpy(b_key))
            block.att.W_value.bias = torch.nn.Parameter(torch.from_numpy(b_value))

        block.att.out_proj.weight = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['attn']['c_proj']['w'].T))
        block.att.out_proj.bias = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['attn']['c_proj']['b']))

        # LayerNorm 2
        block.norm2.scale = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['ln_2']['g']))
        block.norm2.shift = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['ln_2']['b']))

        # MLP
        block.ff.layers[0].weight = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['mlp']['c_fc']['w'].T))
        block.ff.layers[0].bias = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['mlp']['c_fc']['b']))
        block.ff.layers[2].weight = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['mlp']['c_proj']['w'].T))
        block.ff.layers[2].bias = torch.nn.Parameter(torch.from_numpy(params['blocks'][i]['mlp']['c_proj']['b']))

    # Final LayerNorm
    if 'ln_f' in params:
        model.final_norm.scale = torch.nn.Parameter(torch.from_numpy(params['ln_f']['g']))
        model.final_norm.shift = torch.nn.Parameter(torch.from_numpy(params['ln_f']['b']))
    elif 'g' in params and 'b' in params:
        model.final_norm.scale = torch.nn.Parameter(torch.from_numpy(params['g']))
        model.final_norm.shift = torch.nn.Parameter(torch.from_numpy(params['b']))
    else:
        raise KeyError("Final layer norm parameters not found in params. Expected 'ln_f' or top-level 'g'/'b'.")

    # Output head
    model.out_head.weight = torch.nn.Parameter(torch.from_numpy(params['wte']))
