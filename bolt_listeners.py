import logging
from typing import Optional, List, Dict
import re
import time

from slack_bolt import App, Ack, BoltContext, BoltResponse
from slack_bolt.request.payload_utils import is_event
from slack_sdk.web import WebClient, SlackResponse

import openai


from env import (
    OPENAI_TIMEOUT_SECONDS,
    SYSTEM_TEXT,
    TRANSLATE_MARKDOWN,
)

# from app.memory_ops import update_memory

# from app.i18n import translate
# from app.openai_ops import (
#     ask_llm,
#     format_openai_message_content,
#     build_system_text,
# )
# from app.slack_ops import (
#     find_parent_message,
#     is_no_mention_thread,
#     post_wip_message,
#     update_wip_message,
# )
from openai_asker import (
    ask_llm,
    format_openai_message_content,
    build_system_text,
)


#
# Listener functions
#


def just_ack(ack: Ack):
    ack()


TIMEOUT_ERROR_MESSAGE = (
    f":warning: Sorry! It looks like OpenAI didn't respond within {OPENAI_TIMEOUT_SECONDS} seconds. "
    "Please try again later. :bow:"
)
DEFAULT_LOADING_TEXT = "Wait a second, please ...:rocket:"


def find_parent_message(
    client: WebClient, channel_id: Optional[str], thread_ts: Optional[str]
) -> Optional[dict]:
    if channel_id is None or thread_ts is None:
        return None

    messages = client.conversations_history(
        channel=channel_id,
        latest=thread_ts,
        limit=1,
        inclusive=1,
    ).get("messages", [])

    return messages[0] if len(messages) > 0 else None


# ----------------------------
# WIP reply message stuff
# ----------------------------


def post_wip_message(
    *,
    client: WebClient,
    channel: str,
    thread_ts: str,
    loading_text: str,
    messages: List[Dict[str, str]],
    user: str,
) -> SlackResponse:
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    return client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=loading_text,
        metadata={
            "event_type": "chat-gpt-convo",
            "event_payload": {"messages": system_messages, "user": user},
        },
    )


def update_wip_message(
    client: WebClient,
    channel: str,
    ts: str,
    text: str,
    messages: List[Dict[str, str]],
    user: str,
) -> SlackResponse:
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    return client.chat_update(
        channel=channel,
        ts=ts,
        text=text,
        metadata={
            "event_type": "chat-gpt-convo",
            "event_payload": {"messages": system_messages, "user": user},
        },
    )


def is_no_mention_thread(context: BoltContext, parent_message: dict) -> bool:
    parent_message_text = parent_message.get("text", "")
    return f"<@{context.bot_user_id}>" in parent_message_text


def respond_to_app_mention(
    context: BoltContext,
    payload: dict,
    client: WebClient,
    logger: logging.Logger,
):
    print("APP_MENTION_CALLED")
    if payload.get("thread_ts") is not None:
        parent_message = find_parent_message(
            client, context.channel_id, payload.get("thread_ts")
        )
        if parent_message is not None:
            if is_no_mention_thread(context, parent_message):
                # The message event handler will reply to this
                return

    wip_reply = None
    # Replace placeholder for Slack user ID in the system prompt
    system_text = build_system_text(SYSTEM_TEXT, TRANSLATE_MARKDOWN, context)
    messages = [{"role": "system", "content": system_text}]

    print("system text:" + system_text, flush=True)

    openai_api_key = context.get("OPENAI_API_KEY")
    try:
        if openai_api_key is None:
            client.chat_postMessage(
                channel=context.channel_id,
                text="To use this app, please configure your OpenAI API key first",
            )
            return

        user_id = context.actor_user_id or context.user_id
        content = ""
        if payload.get("thread_ts") is not None:
            # Mentioning the bot user in a thread
            replies_in_thread = client.conversations_replies(
                channel=context.channel_id,
                ts=payload.get("thread_ts"),
                include_all_metadata=True,
                limit=1000,
            ).get("messages", [])
            reply = replies_in_thread[-1]
            # for reply in replies_in_thread:
            c = reply["text"] + "\n\n"
            content += c
            role = "assistant" if reply["user"] == context.bot_user_id else "user"
            messages.append(
                {
                    "role": role,
                    "content": (
                        format_openai_message_content(reply["text"], TRANSLATE_MARKDOWN)
                    ),
                }
            )
            # update_memory(content)
        else:
            # Strip bot Slack user ID from initial message
            msg_text = re.sub(f"<@{context.bot_user_id}>\\s*", "", payload["text"])
            # update_memory(msg_text)
            messages.append(
                {
                    "role": "user",
                    "content": format_openai_message_content(
                        msg_text, TRANSLATE_MARKDOWN
                    ),
                }
            )

        wip_reply = post_wip_message(
            client=client,
            channel=context.channel_id,
            thread_ts=payload["ts"],
            loading_text=DEFAULT_LOADING_TEXT,
            messages=messages,
            user=context.user_id,
        )

        resp = ask_llm(messages=messages, context=context)
        print("Reply " + resp)

        update_wip_message(
            client=client,
            channel=context.channel_id,
            ts=wip_reply["message"]["ts"],
            text=resp,
            messages=messages,
            user=user_id,
        )

    except openai.APITimeoutError:
        if wip_reply is not None:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + TIMEOUT_ERROR_MESSAGE,
            )
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )
    except Exception as e:
        text = (
            (
                wip_reply.get("message", {}).get("text", "")
                if wip_reply is not None
                else ""
            )
            + "\n\n"
            + f":warning: Failed to start a conversation with ChatGPT: {e}"
        )
        logger.exception(text, e)
        if wip_reply is not None:
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )


def respond_to_new_message(
    context: BoltContext,
    payload: dict,
    client: WebClient,
    logger: logging.Logger,
):
    if payload.get("bot_id") is not None and payload.get("bot_id") != context.bot_id:
        # Skip a new message by a different app
        return

    wip_reply = None
    try:
        is_in_dm_with_bot = payload.get("channel_type") == "im"
        is_no_mention_required = False
        thread_ts = payload.get("thread_ts")
        if is_in_dm_with_bot is False and thread_ts is None:
            return

        openai_api_key = context.get("OPENAI_API_KEY")
        if openai_api_key is None:
            return

        messages_in_context = []
        if is_in_dm_with_bot is True and thread_ts is None:
            # In the DM with the bot
            past_messages = client.conversations_history(
                channel=context.channel_id,
                include_all_metadata=True,
                limit=100,
            ).get("messages", [])
            past_messages.reverse()
            # Remove old messages
            for message in past_messages:
                seconds = time.time() - float(message.get("ts"))
                if seconds < 86400:  # less than 1 day
                    messages_in_context.append(message)
            is_no_mention_required = True
        else:
            # In a thread with the bot in a channel
            messages_in_context = client.conversations_replies(
                channel=context.channel_id,
                ts=thread_ts,
                include_all_metadata=True,
                limit=1000,
            ).get("messages", [])
            if is_in_dm_with_bot is True:
                is_no_mention_required = True
            else:
                the_parent_message_found = False
                for message in messages_in_context:
                    if message.get("ts") == thread_ts:
                        the_parent_message_found = True
                        # Allow mentions in threads
                        is_no_mention_required = is_no_mention_thread(context, message)
                        latest_message = messages_in_context[-1].get("text", "")
                        if f"<@{context.bot_user_id}>" in latest_message:
                            is_no_mention_required = True
                        break
                if the_parent_message_found is False:
                    parent_message = find_parent_message(
                        client, context.channel_id, thread_ts
                    )
                    if parent_message is not None:
                        is_no_mention_required = is_no_mention_thread(
                            context, parent_message
                        )

        messages = []
        user_id = context.actor_user_id or context.user_id
        last_assistant_idx = -1
        indices_to_remove = []
        for idx, reply in enumerate(messages_in_context):
            maybe_event_type = reply.get("metadata", {}).get("event_type")
            if maybe_event_type == "chat-gpt-convo":
                if context.bot_id != reply.get("bot_id"):
                    # Remove messages by a different app
                    indices_to_remove.append(idx)
                    continue
                maybe_new_messages = (
                    reply.get("metadata", {}).get("event_payload", {}).get("messages")
                )
                if maybe_new_messages is not None:
                    if len(messages) == 0 or user_id is None:
                        new_user_id = (
                            reply.get("metadata", {})
                            .get("event_payload", {})
                            .get("user")
                        )
                        if new_user_id is not None:
                            user_id = new_user_id
                    messages = maybe_new_messages
                    last_assistant_idx = idx

        if is_no_mention_required is False:
            return
        if (
            is_in_dm_with_bot is False
            and is_no_mention_required is False
            and last_assistant_idx == -1
        ):
            return

        if is_in_dm_with_bot is True:
            # To know whether this app needs to start a new convo
            if not next(filter(lambda msg: msg["role"] == "system", messages), None):
                # Replace placeholder for Slack user ID in the system prompt
                system_text = build_system_text(
                    SYSTEM_TEXT, TRANSLATE_MARKDOWN, context
                )
                messages.insert(0, {"role": "system", "content": system_text})

        filtered_messages_in_context = []
        for idx, reply in enumerate(messages_in_context):
            # Strip bot Slack user ID from initial message
            if idx == 0:
                reply["text"] = re.sub(
                    f"<@{context.bot_user_id}>\\s*", "", reply["text"]
                )
            if idx not in indices_to_remove:
                filtered_messages_in_context.append(reply)
        if len(filtered_messages_in_context) == 0:
            print("SHOULD NOT BE HERE")
            return

        for reply in filtered_messages_in_context:
            msg_user_id = reply.get("user")
            # update_memory(reply.get("text"))
            messages.append(
                {
                    "content": format_openai_message_content(
                        reply.get("text"), TRANSLATE_MARKDOWN
                    ),
                    "role": "user",
                }
            )

        wip_reply = post_wip_message(
            client=client,
            channel=context.channel_id,
            thread_ts=payload.get("thread_ts") if is_in_dm_with_bot else payload["ts"],
            loading_text=DEFAULT_LOADING_TEXT,
            messages=messages,
            user=user_id,
        )

        latest_replies = client.conversations_replies(
            channel=context.channel_id,
            ts=wip_reply.get("ts"),
            include_all_metadata=True,
            limit=1000,
        )
        if latest_replies.get("messages", [])[-1]["ts"] != wip_reply["message"]["ts"]:
            # Since a new reply will come soon, this app abandons this reply
            client.chat_delete(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
            )
            return

        resp = ask_llm(messages=messages, context=context)
        print("Reply " + resp)
        update_wip_message(
            client=client,
            channel=context.channel_id,
            ts=wip_reply["message"]["ts"],
            text=resp,
            messages=messages,
            user=user_id,
        )
    except openai.APITimeoutError:
        if wip_reply is not None:
            text = (
                (
                    wip_reply.get("message", {}).get("text", "")
                    if wip_reply is not None
                    else ""
                )
                + "\n\n"
                + TIMEOUT_ERROR_MESSAGE
            )
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )
    except Exception as e:
        text = (
            (
                wip_reply.get("message", {}).get("text", "")
                if wip_reply is not None
                else ""
            )
            + "\n\n"
            + f":warning: Failed to reply: {e}"
        )
        logger.exception(text, e)
        if wip_reply is not None:
            client.chat_update(
                channel=context.channel_id,
                ts=wip_reply["message"]["ts"],
                text=text,
            )


def register_listeners(app: App):
    app.event("message")(ack=just_ack, lazy=[respond_to_new_message])
    app.event("app_mention")(ack=just_ack, lazy=[respond_to_app_mention])


MESSAGE_SUBTYPES_TO_SKIP = ["message_changed", "message_deleted"]


# To reduce unnecessary workload in this app,
# this before_authorize function skips message changed/deleted events.
# Especially, "message_changed" events can be triggered many times when the app rapidly updates its reply.
def before_authorize(
    body: dict,
    payload: dict,
    logger: logging.Logger,
    next_,
):
    if (
        is_event(body)
        and payload.get("type") == "message"
        and payload.get("subtype") in MESSAGE_SUBTYPES_TO_SKIP
    ):
        logger.debug(
            "Skipped the following middleware and listeners "
            f"for this message event (subtype: {payload.get('subtype')})"
        )
        return BoltResponse(status=200, body="")
    next_()
