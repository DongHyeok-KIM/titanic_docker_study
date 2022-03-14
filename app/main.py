from controller.controller import Controller
from typing import Optional
from fastapi import FastAPI
import uvicorn
import os



controller = Controller()
app = FastAPI()


@app.get("/")
def read_root():
    print("실제경로",os.path.realpath(__file__))
    print("절대경로",os.path.abspath(__file__))
    print("디렉토리경로",os.getcwd())
    print("파일의폴더경로",os.path.dirname(os.path.realpath(__file__)))

    result = controller.strat()

    return result


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

if __name__ == '__main__':
    uvicorn.run(app, port="8010",host="0.0.0.0")