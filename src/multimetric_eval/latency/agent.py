from inspect import signature
from typing import List, Union
from .basics import AgentStates, Action, Segment, EmptySegment, TextSegment, SpeechSegment

class GenericAgent:
    """同传 Agent 基类，用户需继承此类并实现 policy 方法"""
    def __init__(self):
        self.states = AgentStates()
        self.reset()
        
    def reset(self): self.states.reset()
    
    def policy(self, states=None) -> Action: 
        raise NotImplementedError("请实现 policy 方法")
        
    def push(self, segment: Segment, states=None) -> None:
        (states or self.states).update_source(segment)
    
    def pop(self, states=None) -> Segment:
        s = states or self.states
        if s.target_finished: return EmptySegment(finished=True)
        
        # 兼容 stateful/stateless
        if len(signature(self.policy).parameters) > 0:
            action = self.policy(s)
        else:
            action = self.policy()
        
        if action.is_read(): return EmptySegment()
        
        # 自动封装 Segment
        if isinstance(action.content, Segment): return action.content
        
        if isinstance(action.content, str):
            seg = TextSegment(content=action.content, finished=action.finished)
        elif isinstance(action.content, list):
            seg = SpeechSegment(content=action.content, finished=action.finished)
        else:
            raise ValueError(f"Unknown content type: {type(action.content)}")
            
        s.update_target(seg)
        return seg

    def pushpop(self, segment: Segment) -> Segment:
        self.push(segment)
        return self.pop()

class AgentPipeline(GenericAgent):
    """简单的串行 Pipeline"""
    def __init__(self, agents: List[GenericAgent]):
        self.pipeline = agents
        self.states = [agent.states for agent in agents]

    def reset(self):
        for agent in self.pipeline: agent.reset()

    def pushpop(self, segment: Segment) -> Segment:
        current_input = segment
        for i, agent in enumerate(self.pipeline):
            if i == 0:
                agent.push(current_input)
                current_output = agent.pop()
            else:
                if not current_output.is_empty:
                    agent.push(current_output)
                current_output = agent.pop()
            
            # 简单的流控：只要中间环节 Empty，则整体 Empty (除非是最后一个)
            if current_output.is_empty and i < len(self.pipeline) - 1:
                return EmptySegment()
        return current_output