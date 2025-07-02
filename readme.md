# LLM From Scratch
This is a fun little project to make a mini GPT-2 style language model trained on the "The Big Bang Theory" TV show transcripts. The model uses a decoder-only transformer architecture with character level tokenization, and features 128 sequence length at 3.2 million parameters, and was trained on a laptop CPU.

The model performs surprisingly well for its size and was able to learn most words in the dataset, as well as form some simple complete sentences. During inference after training, the model achieves a perplexity of about 3.1, demonstrating relative confidence during generation.

## Training
The model is trained with 80k steps at a batch size of 32, and was able to train within about 4 hours on a macbook m3 CPU. After training the model reached a final training loss of about 1.1-1.2, and a final validation loss on the test set of 1.151 in cross entropy loss. 

## Setup
1. To set it up and start running, first create a python virtual environment

    `python3 -m venv .venv`

2. Then activate the environment:
    - Mac/linux: `source .venv/bin/activate`
    - Windows: `venv\Scripts\activate`

3. Then install the dependencies in the virtual environment

    `pip install -r requirements.txt`

4. Next, you need to parse the raw csv data, this can be done by running the TBBT_parse.py script:

    `python3 TBBT_parse.py`

Once all of these steps are complete, you should be all set and can go to `Transformer.ipynb` to train your own!