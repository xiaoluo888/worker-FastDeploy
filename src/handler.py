import runpod
from utils import JobInput
from engine2 import FastDeployEngine
from runpod import RunPodLogger
log = RunPodLogger()
fd_engine = FastDeployEngine()

async def handler(job):
    job_input = JobInput(job["input"])
    
    # log.info(aaa)
    
    engine = fd_engine

    results_generator = engine.generate(job_input)

    #1
    # async for batch in results_generator:
    #     yield batch

    #2
    results = []
    async for batch in results_generator:
        results.append(batch)

    return results


runpod.serverless.start(
    {
        "handler": handler,
         "return_aggregate_stream": True,
    }
)

