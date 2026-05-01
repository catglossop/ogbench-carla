# TODO: RemoteActor class and LocalActor class
import abc
import json
from typing import Optional

import numpy as np
import requests


class RemoteActor(abc.ABC):
    def __init__(self, actor_url: str):
        self.actor_url = actor_url
    
    def get_info(self) -> dict:
        '''
        Get the info from the remote actor.
        '''
        address = self.actor_url + "/get_info"
        response = requests.get(address)
        
        print("Remote actor info: ", response.json())
        
        return response.json()

    def get_action(self, state: dict) -> np.ndarray:
        '''
        Get the action from the remote actor.
        '''
        # Convert state to JSON
        state_json = json.dumps(state)
        
        address = self.actor_url + "/gen_action"
        response = requests.post(address, json=state_json)
        
        return response.json()
    
    def get_cot(self, state: dict) -> dict:
        '''
        Get the chain-of-thought from the remote actor.
        '''
        # Convert state to JSON
        state_json = json.dumps(state)
        
        address = self.actor_url + "/gen_cot"
        response = requests.post(address, json=state_json)
        
        return response.json()
    
    def update(self):
        '''
        Trigger the update of the remote actor
        '''
        address = self.actor_url + "/update"
        response = requests.post(address)
        
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