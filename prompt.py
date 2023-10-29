import re
import json
import streamlit as st
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chat_models import ChatOpenAI

from docx import Document

def format_schema():
    
    response_schemas = [
        ResponseSchema(name="scenario", description="In less than 3 sentences, create a realistic scenario that leads up to a test question."),
        ResponseSchema(name="question", description="Create a difficult test question using the scenario. The reader should not be able to derive the answer from the given scenario."),
        ResponseSchema(name="choices", description="Options for the multiple-choice question in a), b), c), d) format. Each answer choice should be on a new line. Only one correct answer."),
        ResponseSchema(name="answer", description="Correct answer for the asked question."),
        ResponseSchema(name="explanation", description="An explanation for the correct answer for the asked question.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    return format_instructions

def prompt_template(format_instructions, answer):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, 
            create a realistic scenario leading up to a question from it along with the correct answer and 
            an answer explanation. 
            \n{format_instructions}\n{answer}""")  
        ],
        input_variables=["answer"],
        partial_variables={"format_instructions": format_instructions}
    )
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6, openai_api_key=st.session_state["OpenAI_API_Key"])
    final_query = prompt.format_prompt(answer = answer)
    final_query_output = chat_model(final_query.to_messages())
    # final_query_output = final_query.to_messages()
    markdown_text = final_query_output.content
    # markdown_text = final_query_output
    
    pattern = r'\{(.*?)\}'

    # Find all matches in the text
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # Extract and print the results
    for match in matches:
        # Parse the JSON content
        match = '{' + match + '}'  # Wrapping to make it a valid JSON
        data = json.loads(match)

        scenario = data["scenario"]
        question = data["question"]
        choices = data["choices"]
        answers = data["answer"]
        explanation = data["explanation"]

    return scenario, question, choices, answers, explanation



def format_schema_alt1():
    
    response_schemas = [
        ResponseSchema(name="question", description="Create a difficult test question using the provided text."),
        ResponseSchema(name="choices", description="Options for the multiple-choice question in a), b), c), d) format. Each answer choice should be on a new line. Only one correct answer."),
        ResponseSchema(name="answer", description="Correct answer for the asked question."),
        ResponseSchema(name="explanation", description="An explanation for the correct answer for the asked question.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions_alt1 = output_parser.get_format_instructions()
    
    return format_instructions_alt1

def prompt_template_alt1(format_instructions_alt1, answer):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, 
            create a test question from it along with the correct answer and an answer explanation. 
            \n{format_instructions}\n{answer}""")  
        ],
        input_variables=["answer"],
        partial_variables={"format_instructions": format_instructions_alt1}
    )
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=st.session_state["OpenAI_API_Key"])
    final_query = prompt.format_prompt(answer = answer)
    final_query_output = chat_model(final_query.to_messages())
    # final_query_output = final_query.to_messages()
    markdown_text = final_query_output.content
    # markdown_text = final_query_output
    
    pattern = r'\{(.*?)\}'

    # Find all matches in the text
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # Extract and print the results
    for match in matches:
        # Parse the JSON content
        match = '{' + match + '}'  # Wrapping to make it a valid JSON
        data = json.loads(match)

        question = data["question"]
        choices = data["choices"]
        answers = data["answer"]
        explanation = data["explanation"]

    return question, choices, answers, explanation

def format_schema_alt2():
    
    response_schemas = [
        ResponseSchema(name="question", description="Create a test question with a missing blank using the provided text."),
        ResponseSchema(name="choices", description="Options for the multiple-choice question in a), b), c), d) format. Each answer choice should be on a new line. Only one correct answer."),
        ResponseSchema(name="answer", description="Correct answer for the asked question."),
        ResponseSchema(name="explanation", description="An explanation for the correct answer for the asked question.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions_alt2 = output_parser.get_format_instructions()
    
    return format_instructions_alt2

def prompt_template_alt2(format_instructions_alt2, answer):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""When a text input is given by the user, 
            create a test question with a missing blank along with the correct answer and an answer explanation. 
            \n{format_instructions}\n{answer}""")  
        ],
        input_variables=["answer"],
        partial_variables={"format_instructions": format_instructions_alt2}
    )
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, openai_api_key=st.session_state["OpenAI_API_Key"])
    final_query = prompt.format_prompt(answer = answer)
    final_query_output = chat_model(final_query.to_messages())
    # final_query_output = final_query.to_messages()
    markdown_text = final_query_output.content
    # markdown_text = final_query_output
    
    pattern = r'\{(.*?)\}'

    # Find all matches in the text
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # Extract and print the results
    for match in matches:
        # Parse the JSON content
        match = '{' + match + '}'  # Wrapping to make it a valid JSON
        data = json.loads(match)

        question = data["question"]
        choices = data["choices"]
        answers = data["answer"]
        explanation = data["explanation"]

    return question, choices, answers, explanation

# def keywords_format_schema():

#     ResponseSchema(name:"keywords", description="Extract comma seperated keywords from the uploaded document.")

formats = [format_schema(), format_schema_alt1(), format_schema_alt2()]
prompts = [prompt_template, prompt_template_alt1, prompt_template_alt2]

def format_schema_keywords(amount):
    
    response_schemas = [
        ResponseSchema(name="context", description="Determine the subject of the given text and create a category for it."),
        ResponseSchema(name="keywords", description=f"Extract {amount} comma seperated key terminology from the text.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    
    return format_instructions

def keyword_extractor(format_instructions, answer, amount):
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template("""Given the context of the provided text,
            extract {amount} key terms from the document. Seperate key terms by 
            a comma. Give preference to words in bold font.
            \n{format_instructions}\n{answer}""")
        ], 
        input_variables=["answer"],
        partial_variables={"amount": amount, "format_instructions": format_instructions}
    )
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=st.session_state["OpenAI_API_Key"])
    final_query = prompt.format_prompt(answer = answer)    
    final_query_output = chat_model(final_query.to_messages())
    # final_query_output = final_query.to_messages()
    markdown_text = final_query_output.content
    pattern = r'\{(.*?)\}'

    # Find all matches in the text
    matches = re.findall(pattern, markdown_text, re.DOTALL)

    # Extract and print the results
    for match in matches:
        # Parse the JSON content
        match = '{' + match + '}'  # Wrapping to make it a valid JSON
        data = json.loads(match)

        words = data["keywords"]
        context = data["context"]
    
    return words, context
