import logging
import traceback
import coloredlogs
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from flow_langchain_full import inference
from config import *
coloredlogs.install(
    level="INFO", fmt="%(asctime)s %(name)s[%(process)d] %(levelname)-8s %(message)s"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()


class T(BaseModel):
    text: str

@app.post("/infer")
async def infer(data: T):
    try:
        text = data.text
        res = {
            "success": True,
            "result": inference(text),
        }
    except Exception as e:
        logger.critical(traceback.format_exc())
        res = {
            "success": False
        }
    return res


if __name__ == "__main__":
    uvicorn.run(app, host=str(SERVICE_HOST), port=int(SERVICE_PORT), log_level="debug")
