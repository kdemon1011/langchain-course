from dotenv import load_dotenv

load_dotenv()
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"format_instructions": format_instructions},
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output


def main():
    print("Hello from langchain-course!")
    result = chain.invoke(
        {
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
