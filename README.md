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


### Useful resources:

- Langchain 0.2 documentation - https://python.langchain.com/v0.2/docs
- Prompt engineering guide - https://www.promptingguide.ai/it/introduction/tips
- Retrievers - https://python.langchain.com/v0.2/docs/how_to/#retrievers

