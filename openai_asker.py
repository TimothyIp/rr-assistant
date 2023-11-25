from env import PINCONE_API_KEY, PINECONE_ENV, VECTOR_INDEX_NAME
from langchain.vectorstores.pinecone import Pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
import re
import pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from typing import List, Dict
from slack_bolt import BoltContext
import tiktoken
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


MAX_TOKENS = 1024


# Conversion from Slack mrkdwn to OpenAI markdown
# See also: https://api.slack.com/reference/surfaces/formatting#basics
def slack_to_markdown(content: str) -> str:
    # Split the input string into parts based on code blocks and inline code
    parts = re.split(r"(```.+?```|`[^`\n]+?`)", content)

    # Apply the bold, italic, and strikethrough formatting to text not within code
    result = ""
    for part in parts:
        if part.startswith("```") or part.startswith("`"):
            result += part
        else:
            for o, n in [
                (r"\*(?!\s)([^\*\n]+?)(?<!\s)\*", r"**\1**"),  # *bold* to **bold**
                (r"_(?!\s)([^_\n]+?)(?<!\s)_", r"*\1*"),  # _italic_ to *italic*
                (r"~(?!\s)([^~\n]+?)(?<!\s)~", r"~~\1~~"),  # ~strike~ to ~~strike~~
            ]:
                part = re.sub(o, n, part)
            result += part
    return result


# Conversion from OpenAI markdown to Slack mrkdwn
# See also: https://api.slack.com/reference/surfaces/formatting#basics
def markdown_to_slack(content: str) -> str:
    # Split the input string into parts based on code blocks and inline code
    parts = re.split(r"(```.+?```|`[^`\n]+?`)", content)

    # Apply the bold, italic, and strikethrough formatting to text not within code
    result = ""
    for part in parts:
        if part.startswith("```") or part.startswith("`"):
            result += part
        else:
            for o, n in [
                (
                    r"\*\*\*(?!\s)([^\*\n]+?)(?<!\s)\*\*\*",
                    r"_*\1*_",
                ),  # ***bold italic*** to *_bold italic_*
                (
                    r"(?<![\*_])\*(?!\s)([^\*\n]+?)(?<!\s)\*(?![\*_])",
                    r"_\1_",
                ),  # *italic* to _italic_
                (r"\*\*(?!\s)([^\*\n]+?)(?<!\s)\*\*", r"*\1*"),  # **bold** to *bold*
                (r"__(?!\s)([^_\n]+?)(?<!\s)__", r"*\1*"),  # __bold__ to *bold*
                (r"~~(?!\s)([^~\n]+?)(?<!\s)~~", r"~\1~"),  # ~~strike~~ to ~strike~
            ]:
                part = re.sub(o, n, part)
            result += part
    return result


# Format message from Slack to send to OpenAI
def format_openai_message_content(content: str, translate_markdown: bool) -> str:
    if content is None:
        return None

    # Unescape &, < and >, since Slack replaces these with their HTML equivalents
    # See also: https://api.slack.com/reference/surfaces/formatting#escaping
    content = content.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    # Convert from Slack mrkdwn to markdown format
    if translate_markdown:
        content = slack_to_markdown(content)

    return content


def calculate_num_tokens(
    messages: List[Dict[str, str]],
    # TODO: adjustment for gpt-4
    model: str = "gpt-3.5-turbo-1106",
) -> int:
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-1106":
        # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        error = (
            f"Calculating the number of tokens for for model {model} is not yet supported. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md "
            "for information on how messages are converted to tokens."
        )
        raise NotImplementedError(error)


def ask_llm(*, messages: List[Dict[str, str]], context: BoltContext) -> str:
    # Remove old messages to make sure we have room for max_tokens
    # See also: https://platform.openai.com/docs/guides/chat/introduction
    # > total tokens must be below the model’s maximum limit (4096 tokens for gpt-3.5-turbo-0301)
    # TODO: currently we don't pass gpt-4 to this calculation method
    while calculate_num_tokens(messages) >= 4096 - MAX_TOKENS:
        removed = False
        for i, message in enumerate(messages):
            if message["role"] in ("user", "assistant"):
                del messages[i]
                removed = True
                break
        if not removed:
            # Fall through and let the OpenAI error handler deal with it
            break

    prompt = ""

    latest_message = messages[-1]
    if latest_message:
        prompt = latest_message["content"]

    parsed_prompt = re.sub(f"<@{context.bot_user_id}>\\s*", "", prompt)
    print("PROMPT:", parsed_prompt)

    # for i, message in enumerate(messages[-]):
    #     prompt += message["content"] + "\n"

    return ask_openai(parsed_prompt)


def ask_openai(question) -> str:
    # initialize pinecone
    print("Init retriever")
    pinecone.init(api_key=PINCONE_API_KEY, environment=PINECONE_ENV)
    embeddings = OpenAIEmbeddings()

    pinecone_client = Pinecone.from_existing_index(VECTOR_INDEX_NAME, embeddings)

    retriever = pinecone_client.as_retriever(search_type="mmr")

    matched_docs = retriever.get_relevant_documents(question)
    for i, d in enumerate(matched_docs):
        print(f"\n## Document {i}\n")
        print(d.page_content)
        print(d.metadata)

    memory = ConversationBufferMemory(
        memory_key="history", input_key="question", output_key="answer"
    )
    # template = """You are the helpful Rose Rocket assistant, please answer the question as descriptive as possible from the context given to you.
    # If you do not know the answer to the question, simply respond with "I don't know the answer to that question.".
    # If questions are asked where there is no relevant context available, simply respond with "I'm sorry, I don't have enough information to answer that question."
    # Context: {context}

    # Human: {question}
    # Assistant:"""

    # prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    res = ""
    llm = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], temperature=0)
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        memory=memory,
        # chain_type_kwargs={
        #     "prompt": prompt,
        # },
    )

    # Get the answer from the chain
    res = qa(question)
    # res = qa(
    #     "---------------------\n Given the context above, answer to the following question: "
    #     + prompt
    # )
    (
        answer,
        docs,
    ) = (
        res["answer"],
        res["source_documents"],
    )

    print("RESULT:", docs)

    if (
        answer == "I'm sorry, I don't have enough information to answer that question."
        or answer == "I don't know the answer to that question."
        or answer == "I don't know the answer."
    ):
        res = answer
    else:
        res = answer + "\n\n\n" + "Sources:\n"

        sources = set()  # To store unique sources

        # Collect unique sources
        for document in docs:
            if document.metadata:
                notion_url = document.metadata["notion_url"].replace("#", "")
                sources.add(notion_url)

        # Print the relevant sources used for the answer
        for source in sources:
            if source.startswith("https"):
                res += "- " + source + "\n"
            else:
                res += "- source code: " + source + "\n"

    return res


def build_system_text(
    system_text_template: str, translate_markdown: bool, context: BoltContext
):
    system_text = system_text_template.format(bot_user_id=context.bot_user_id)
    # Translate format hint in system prompt
    if translate_markdown is True:
        system_text = slack_to_markdown(system_text)
    return system_text
