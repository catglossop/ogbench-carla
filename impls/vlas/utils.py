# TODO: RemoteActor class and LocalActor class
import abc
from typing import Any, Optional

import numpy as np
import requests


def _state_to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy arrays (and scalars) for ``requests`` JSON bodies."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _state_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_state_to_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class RemoteActor(abc.ABC):
    def __init__(self, actor_url: str, request_options: Optional[dict[str, Any]] = None):
        self.actor_url = actor_url
        self.request_options = dict(request_options or {})

    def _request_body(self, state: dict) -> dict:
        body = _state_to_jsonable(state)
        if self.request_options:
            body = dict(body)
            body["__steervla_options__"] = _state_to_jsonable(self.request_options)
        return body
    
    def get_info(self) -> dict:
        '''
        Get the info from the remote actor.
        '''
        address = self.actor_url + "/get_info"
        response = requests.get(address)
        response.raise_for_status()
        print("Remote actor info: ", response.json())
        
        return response.json()

    def get_action(self, state: dict) -> np.ndarray:
        '''
        Get the action from the remote actor.
        '''
        address = self.actor_url + "/gen_action"
        response = requests.post(address, json=self._request_body(state))
        response.raise_for_status()
        return np.asarray(response.json(), dtype=np.float32)
    
    def get_cot(self, state: dict) -> dict:
        '''
        Get the chain-of-thought from the remote actor.
        '''
        address = self.actor_url + "/gen_cot"
        response = requests.post(address, json=self._request_body(state))
        response.raise_for_status()
        return response.json()
    
    def update(self):
        '''
        Trigger the update of the remote actor
        '''
        address = self.actor_url + "/update"
        response = requests.post(address)
        response.raise_for_status()
        return response.json()
    
class LocalActor(abc.ABC):
    
    def __init__(self, actor_config: str, checkpoint_path: Optional[str] = None):
        self.actor_config = actor_config
        self.checkpoint_path = checkpoint_path
      
    @abc.abstractmethod
    def setup(self):
        pass


    @abc.abstractmethod
    def get_action(self, state: dict) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def get_cot(self, state: dict) -> dict:
        pass
    
    @abc.abstractmethod
    def update(self):
        pass