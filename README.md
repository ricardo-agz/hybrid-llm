# EdgeLLM

Mixture-of-Depths Across Edge and API Models.

The goal is to use entropy and varentropy to dynamically swap between different models on a token-by-token basis. 
When the model is confident, we can rely on the outputs from a smaller model running locally, when entropy is high and 
confidence is low, we can switch into using a larger model (presumably running behind an API) for the following tokens 
until confidence is high and we can consider switching back to the smaller, local model.

It is inspired by [Entropix Entropy Based Sampling](https://github.com/xjdr-alt/entropix).

This is a research project and a work in progress. It is absolutely not optimized for running in production and will 
likely not have much to show for itself in terms of latency gains. It is meant to be a starting point and a proof of 
concept.


## Getting Started

Install requirements
```bash
pip install -r requirements.txt
```

CD into the `edge-llm` directory

### Running through the CLI
```bash
python main.py
```

### Running with the GUI

Running the API model
```bash
python serve_cloud_model.py
```

Running the main server
```bash
python main.py
```

### Running the GUI
CD into the `gui` directory
```bash
npm run dev
```
