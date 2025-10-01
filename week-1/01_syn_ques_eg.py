import asyncio
import random
from asyncio import Semaphore

import braintrust
import datasets
import instructor
import openai
from pydantic import BaseModel
from rich import print
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm_asyncio

dataset = datasets.load_dataset("567-labs/bird-rag")["train"]

print(dataset[0])
for item in dataset:
    if item["difficulty"] == "challenging":
        print(item["query"])
        break


# This represents how we're representing our data from the dataset
class Chunk(BaseModel):
    chunk_id: str
    text: str


# This is the synthetic question that we want our model to generate
class Question(BaseModel):
    chain_of_thought: str
    question: str


# This is a single question-chunk pair that we'll be uploading to Braintrust as a dataset later on to be used for benchmarking in `benchmark_retrieval.py`
class ChunkEval(BaseModel):
    chunk_id: str
    question: str
    chunk: str


# Define Instructor Client
client = instructor.from_openai(openai.AsyncOpenAI())

# Define some constraints to make the question more challenging
constraints = [
    "If there's a time period mentioned in the snippet, modify it slightly (Eg. if the snippet is looking at the entire year, change it to 6 months or 1.5 years)",
    "Add in some irrelevant context (Eg. Add information about the weather, a random event or a backstory that isn't mentioned in the snippet)",
    "Changing the value of the filter (Eg. if the snippet is looking at the results in Canada, change the question to ask about another country or city instead)",
]


# retry decorator works by detecting any errors and retrying the function 3 times with a 10 second delay between attempts
@retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
async def generate_questions(chunk: Chunk, sem: Semaphore) -> ChunkEval:
    # GATE: Only 10 workers can execute this expensive API call section at once (semaphore defined in main())
    async with sem:
        # Step 1: Create a coroutine (promise) for the API call - NOT executed yet
        coro = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": """
                Generate a hypothetical question that can be answered using the following SQL snippet.

                SQL Snippet:
                {{ snippet }}

                Rules
                - If there are specific values in the snippet, do not use them directly in the question if possible.
                - The question should be at most 2 sentences long
                - if necessary, consider making the question more challenging using the following constraint of {{ constraint }}
                - The question must be answerable using the SQL snippet or at most with a small tweak
                """,
                }
            ],
            response_model=Question,
            context={"snippet": chunk.text, "constraint": random.choice(constraints)},
        )

        # Step 2: Execute the coroutine with a 30 second timeout
        # - await = actually make the API call and wait for response
        # - timeout=30 = if API takes longer than 30 seconds, raise TimeoutError
        # - If timeout occurs, the @retry decorator will automatically retry up to 3 times
        resp = await asyncio.wait_for(coro, timeout=30)

        return ChunkEval(
            chunk_id=chunk.chunk_id,
            question=resp.question,
            chunk=chunk.text,
        )


# in Jupyter/IPython notebooks, you can directly use await at the cell level without wrapping it in a function. This is a special feature that makes async code more convenient to work with interactively
async def main():
    sem = Semaphore(10)
    dataset = [
        item
        for item in datasets.load_dataset("567-labs/bird-rag")["train"]
        if item["difficulty"] == "challenging"
    ]
    dataset = [Chunk(chunk_id=item["id"], text=item["query"]) for item in dataset]

    coros = []

    num_samples = 2
    for chunk in dataset:
        for _ in range(num_samples):
            coros.append(generate_questions(chunk, sem))

    questions: list[ChunkEval] = await tqdm_asyncio.gather(*coros)

    # Initialise Braintrust Dataset
    dataset = braintrust.init_dataset(project="Text-2-SQL", name="Bird-Bench-Questions")

    # Insert Individual Questions row by row
    for question in questions:
        dataset.insert(
            input=question.question,
            expected=[question.chunk],
            metadata={"chunk_id": question.chunk_id, "chunk": question.chunk},
        )

    print(dataset.summarize())

    return questions


if __name__ == "__main__":
    questions = asyncio.run(main())
