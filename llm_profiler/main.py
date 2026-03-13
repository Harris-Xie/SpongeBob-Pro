from module import TransformerModelProfile
from config import ModelConfig

import json
import sys

def main():

    json_path = sys.argv[1]

    # print(f'json_path: {json_path}')

    with open(json_path, "r", encoding="utf-8") as f:
        data :dict = json.load(f)

    model_config = ModelConfig(**data)
    model = TransformerModelProfile(model_config)

    print(f'total params: {model.total_params()}')
    print(f'human_readable: {f"{model.total_params() / 1_000_000:.2f}M"}')


if __name__ == "__main__":
    main()