from env import (
    PINCONE_API_KEY,
    PINECONE_ENV,
    VECTOR_INDEX_NAME,
    ZILLIZ_CLOUD_URI,
    ZILLIZ_CLOUD_API_KEY,
    VECTOR_STORE,
    OPENAI_MODEL,
    OPENAI_API_KEY,
)
import os
import logging
from langchain.vectorstores.pinecone import Pinecone
from langchain.vectorstores.milvus import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter


from markdown import markdown_to_slack
from personality_bank import ANIME_WEEB


import re
import pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from typing import List, Dict
from slack_bolt import BoltContext
import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletion
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from markdown import slack_to_markdown
import time


MAX_TOKENS = 1024


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


def generate_slack_thread_summary(
    *,
    context: BoltContext,
    logger: logging.Logger,
    prompt: str,
    thread_content: str,
    timeout_seconds: int,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You're an assistant tasked with helping Slack users by summarizing threads. "
                "You'll receive a collection of replies in this format: <@user_id>: reply text\n"
                "Your role is to provide a very detailed summary that highlights key facts and which <@user_id made which decision. Always return this summary in Markdown format in a bullet list form with bold section categories"
            ),
        },
        {
            "role": "user",
            "content": f"{prompt}\n\n{thread_content}",
        },
    ]
    start_time = time.time()
    ai_client = OpenAI(api_key=OPENAI_API_KEY)

    openai_response: ChatCompletion = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        top_p=1,
        n=1,
        max_tokens=MAX_TOKENS,
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias={},
        user=context.actor_user_id,
        stream=False,
        timeout=timeout_seconds,
    )
    spent_time = time.time() - start_time
    logger.debug(f"Making a summary took {spent_time} seconds")
    return openai_response.choices[0].message.content


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
    parsed_prompt = re.sub(f"<@{context.user_id}>\\s*", "", parsed_prompt)

    return ask_openai(context, parsed_prompt)


def get_pinecone_retriever():
    # initialize pinecone
    print("Init retriever - pinecone")
    pinecone.init(api_key=PINCONE_API_KEY, environment=PINECONE_ENV)
    embeddings = OpenAIEmbeddings()

    pinecone_client = Pinecone.from_existing_index(VECTOR_INDEX_NAME, embeddings)

    return pinecone_client.as_retriever(
        search_type="mmr",
    )


def get_milvus_retriever():
    print("Init retriever - milvus")
    embeddings = OpenAIEmbeddings()
    return Milvus(
        embeddings,
        connection_args={
            "uri": ZILLIZ_CLOUD_URI,
            "token": ZILLIZ_CLOUD_API_KEY,
            "secure": True,
            "collection_name": "LangChainCollection",
        },
    ).as_retriever(search_type="mmr", search_kwargs={"k": 20})


def get_vector_store_retriever():
    if VECTOR_STORE == "pinecone":
        return get_pinecone_retriever()

    if VECTOR_STORE == "milvus":
        return get_milvus_retriever()


## Primarily for testing new functionality out
def is_test_user(context: BoltContext):
    test_user_slack_ids = [
        # "U8807CX62",  # Tim
        # "U049XMRT755",  # Ping
        "U056XGEUX4G"  # Fortunado
        "U01HMVAM49M",  # Mibo
    ]

    return context.user_id in test_user_slack_ids


def ask_openai(context: BoltContext, question) -> str:
    print("USER_ID:", context.user_id)
    print("QUESTION:", question)
    print("AI MODEL:", OPENAI_MODEL)
    retriever = get_vector_store_retriever()

    cohere_rerank_compressor = CohereRerank(
        top_n=5, cohere_api_key=os.environ["COHERE_API_KEY"], user_agent="langchain"
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank_compressor, base_retriever=retriever
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        ai_prefix="AI Assistant",
        human_prefix="Friend",
    )
    # Many things to see, many noises, big tribe life in Toronto.

    template = """You are an helpful AI assistant for answering questions about Rose Rocket.
    You are given the following question and context. Provide a concise and detailed answer following Slack's markdown language mrkdwn for various formatting elements.
    For URLs you should escape them like this:<http://example.com|This is a clickable link>. If you want to combine bullet list items with bold markdown, this is how it should work: – *A bold text*: A normal text. Never use 2 asterisks after anther like this **. It's an invalid markdown!

    If you don't know the answer, just say "I don't know the answer to that question."

    Question: {question}
    =========
    Context: {summaries}
    =========
    """
    if is_test_user(context):
        template = (
            ANIME_WEEB
            + """
            Question: {question}
            =========
            Context: {summaries}
            =========
            """
        )

    # template = """You are an helpful AI assistant for answering questions about Rose Rocket.
    # You are given the following context and a question. Provide a detailed answer.
    # If you don't know the answer, just say "I don't know the answer to that question."

    # Question: {question}
    # =========
    # {summaries}
    # =========
    # Answer in Markdown:"""

    # template = """You are a caveman.
    # You are given the following question.

    # Question: {question}
    # =========
    # {summaries}
    # =========
    # ALWAYS return the answer as caveman speak and return the best answer you can:"""

    QA_PROMPT = PromptTemplate(
        template=template, input_variables=["question", "summaries"]
    )

    DOC_PROMPT = PromptTemplate(
        template="Content: {page_content}\n",
        input_variables=["page_content"],
    )

    res = ""
    llm = ChatOpenAI(
        model=OPENAI_MODEL, api_key=os.environ["OPENAI_API_KEY"], temperature=0
    )
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        memory=memory,
        chain_type_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": DOC_PROMPT,
            "verbose": True,
        },
    )

    # Get the answer from the chain
    res = qa(question)
    (
        answer,
        docs,
    ) = (
        res["answer"],
        res["source_documents"],
    )

    answer = markdown_to_slack(answer)

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
