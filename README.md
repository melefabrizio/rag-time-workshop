# RAG Time!

### How to run the project

1. Clone the repository
2. Get your envs and paste them in a `.env` file in the root of the project
3. Install poetry with `pip install poetry`
4. Install the dependencies with `poetry install`
5. Start a poetry shell with `poetry shell`
6. Run the project with `python demo/main.py` or `python workshop/main.py`

If you want to run everything with docker, you can start the docker container with `docker compose up --build`,
log into the container with `docker exec -it app bash` and then resume from step 5.

### How to manage the dependencies

1. Add a new dependency with `poetry add <package>`
2. Remove a dependency with `poetry remove <package>`
3. Update the dependencies with `poetry update`

### Available models:

Using ChatBedrock you can acces the following models:
- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1
- meta.llama2-70b-chat-v1
- meta.llama3-8b-instruct-v1:0
- meta.llama3-70b-instruct-v1:0
- anthropic.claude-v2
- anthropic.claude-v2:1
- anthropic.claude-3-haiku-20240307-v1:0
- anthropic.claude-3-sonnet-20240229-v1:0
- amazon.titan-text-express-v1
- amazon.titan-text-lite-v1
- amazon.titan-text-premier-v1:0

Using BedrockEmbeddings you can acces the following models:
- amazon.titan-embed-text-v2:0
- amazon.titan-embed-text-v1
- cohere.embed-multilingual-v3
- cohere.embed-english-v3

### Useful resources:

- Langchain 0.2 documentation - https://python.langchain.com/v0.2/docs
- Prompt engineering guide - https://www.promptingguide.ai/it/introduction/tips
- Retrievers - https://python.langchain.com/v0.2/docs/how_to/#retrievers

