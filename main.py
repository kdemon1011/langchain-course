from typing import Union, List
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str

load_dotenv()

from langchain_core.tools import Tool, render_text_description, tool
from callbacks import AgentCallbackHandler


@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by counting the number of characters"""
    text = text.strip("'\n'").strip("' '")
    return len(text)


def find_tool_by_name(tool_name: str, tools: List[Tool]) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(
        template=template,
        partial_variables={
            "tools": render_text_description(tools),
            "tool_names": ", ".join([tool.name for tool in tools]),
        },
    )
    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()]).bind(
        stop=["\nObservation", "Observation"]
    )
    intermediate_steps = []
    chain = prompt | llm | ReActSingleInputOutputParser()
    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = chain.invoke(
            {
                "input": "What is the length of the text 'Hello, world!'?",
                "agent_scratchpad": format_log_to_str(intermediate_steps),
            }
        )
        print(agent_step)
        print(type(agent_step))

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tool_name, tools)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, observation))
            print(observation)

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
