import runpod
from utils import JobInput
from engine import FastDeployEngine
from runpod import RunPodLogger
log = RunPodLogger()
fd_engine = FastDeployEngine()

async def handler(job):
    job_input = JobInput(job["input"])
    engine = fd_engine
    results_generator = engine.generate(job_input)

    async for batch in results_generator:
        yield batch
    # results = []
    # async for batch in engine.generate(job_input):
    #     results.append(batch)
    # return {"batches": results}


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda _current: fd_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)

