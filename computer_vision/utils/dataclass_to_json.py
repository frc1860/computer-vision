import dataclasses
import json
import typing

from dacite import from_dict


def load_dataclass(obj: object) -> str:
    obj_dict = dataclasses.asdict(obj)
    return json.dumps(obj_dict)


def extract_dataclass(json_str: str, dataclass_model: typing.Any) -> object:
    obj_dict = json.loads(json_str)
    return from_dict(data_class=dataclass_model, data=obj_dict)
