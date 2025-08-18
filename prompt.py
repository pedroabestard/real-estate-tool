from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources.stuff_prompt import template

my_prompt = """
You are an experienced real estate assistant with deep knowledge of real estate terminology and trends.
When you present data always include relevant dates and numerical values.
Be clear and concise.
"""
new_template = my_prompt + template
PROMPT = PromptTemplate(template=new_template, input_variables=["question", "summaries"])

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)