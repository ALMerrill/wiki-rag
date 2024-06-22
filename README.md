

# Setup

## Python environment

Run the following commands to setup the environment
1. `conda create --name rag python=3.10 -y`
2. `conda activate rag`
3. `pip install -r requirements.txt`
4. `conda install -c pytorch faiss-cpu=1.8.0` (For RagRetriever)

## Gemini

Get an API key for Gemini, and add it to a `.env` file as `GEMINI_API_KEY`

## Etc

I had to run `export PYTORCH_ENABLE_MPS_FALLBACK=1` due to a torch issue running `torch.argsort` on my M1 Macbook.

I hadn't seen that before, but it seems to have gotten past the issue
