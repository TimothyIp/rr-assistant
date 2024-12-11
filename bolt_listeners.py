import logging
import json
from typing import Optional, List, Dict
import re
import time
import datetime

from slack_bolt import App, Ack, BoltContext, BoltResponse
from slack_bolt.request.payload_utils import is_event
from slack_sdk.web import WebClient, SlackResponse
from slack_sdk.errors import SlackApiError

from markdown import slack_to_markdown, markdown_to_slack

import openai


from env import (
    OPENAI_TIMEOUT_SECONDS,
    SYSTEM_TEXT,
    TRANSLATE_MARKDOWN,
)


from openai_asker import (
    ask_llm,
    format_openai_message_content,
    build_system_text,
    generate_slack_thread_summary,
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
DEFAULT_LOADING_TEXT = "Wait a second, please... :rocket:"


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


def is_user_allowed(context: BoltContext, client: WebClient):
    return True
    # payments_dev_user_group_id = "S0306L360JH"

    # resp = client.usergroups_users_list(usergroup=payments_dev_user_group_id)

    # allowed_user_ids = resp["users"]
    # allowed_slack_ids = [
    #     # Tim
    #     "U8807CX62",
    #     # Aaron
    #     "U03FJ97TNN5",
    # ]
    # allowed_slack_ids.extend(allowed_user_ids)

    # return context.user_id in allowed_slack_ids


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
            return

        for reply in filtered_messages_in_context:
            msg_user_id = reply.get("user")
            messages.append(
                {
                    "content": f"<@{msg_user_id}>: "
                    + format_openai_message_content(
                        reply.get("text"), TRANSLATE_MARKDOWN
                    ),
                    "role": (
                        "assistant" if reply["user"] == context.bot_user_id else "user"
                    ),
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


def show_summarize_option_modal(
    ack: Ack,
    client: WebClient,
    body: dict,
    context: BoltContext,
):
    prompt = "All replies posted in a Slack thread will be provided below. Could you summarize the discussion in 400 characters or less?"
    thread_ts = body.get("message").get("thread_ts", body.get("message").get("ts"))
    where_to_display_options = [
        {
            "text": {
                "type": "plain_text",
                "text": "Here, on this modal",
            },
            "value": "modal",
        },
        {
            "text": {
                "type": "plain_text",
                "text": "As a reply in the thread",
            },
            "value": "reply",
        },
        {
            "text": {
                "type": "plain_text",
                "text": "As a reply in the thread that is visible only to you",
            },
            "value": "ephemeral_reply",
        },
    ]
    is_error = False
    blocks = []
    try:
        # Test if this bot is in the channel
        client.conversations_replies(
            channel=context.channel_id,
            ts=thread_ts,
            limit=1,
        )
        blocks = [
            {
                "type": "input",
                "block_id": "where-to-share-summary",
                "label": {
                    "type": "plain_text",
                    "text": "How would you like to see the summary?",
                },
                "element": {
                    "action_id": "input",
                    "type": "radio_buttons",
                    "initial_option": where_to_display_options[0],
                    "options": where_to_display_options,
                },
            },
            {
                "type": "input",
                "optional": True,
                "block_id": "prompt",
                "element": {
                    "type": "plain_text_input",
                    "action_id": "input",
                    "multiline": True,
                    "placeholder": {"type": "plain_text", "text": prompt},
                },
                "label": {
                    "type": "plain_text",
                    "text": "Customize the prompt as you prefer:",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "Note that after the instruction you provide, this app will append all the replies in the thread.",
                    }
                ],
            },
        ]
    except SlackApiError as e:
        is_error = True
        error_code = e.response["error"]
        if error_code == "not_in_channel":
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "It appears that this app's bot user is not a member of the specified channel. "
                        f"Could you please invite <@{context.bot_user_id}> to <#{context.channel_id}> "
                        "to make this app functional?",
                    },
                }
            ]
        else:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Something is wrong! (error: {error_code})",
                    },
                }
            ]

    view = {
        "type": "modal",
        "callback_id": "request-thread-summary",
        "title": {"type": "plain_text", "text": "Summarize the thread"},
        "submit": {"type": "plain_text", "text": "Summarize"},
        "close": {"type": "plain_text", "text": "Close"},
        "private_metadata": json.dumps(
            {
                "thread_ts": thread_ts,
                "channel": context.channel_id,
            }
        ),
        "blocks": blocks,
    }
    if is_error is True:
        del view["submit"]

    client.views_open(
        trigger_id=body.get("trigger_id"),
        view=view,
    )
    ack()


def show_summarize_channel_option_modal(
    ack: Ack,
    client: WebClient,
    body: dict,
    context: BoltContext,
):
    prompt = "All replies posted in a Slack channel will be provided below. Could you summarize the discussion in 600 characters or less?"
    where_to_display_options = [
        {
            "text": {
                "type": "plain_text",
                "text": "Here, on this modal",
            },
            "value": "modal",
        },
        {
            "text": {
                "type": "plain_text",
                "text": "As a reply in the channel",
            },
            "value": "reply",
        },
        {
            "text": {
                "type": "plain_text",
                "text": "As a reply in the channel that is visible only to you",
            },
            "value": "ephemeral_reply",
        },
    ]

    summary_type_options = [
        {
            "text": {
                "type": "plain_text",
                "text": "Summarize the last 24 hours",
            },
            "value": "last_nth_hours",
        },
        {
            "text": {
                "type": "plain_text",
                "text": "Summarize the last 50 messages",
            },
            "value": "last_nth_messages",
        },
    ]

    is_error = False

    blocks = [
        {
            "type": "input",
            "block_id": "where-to-share-summary",
            "label": {
                "type": "plain_text",
                "text": "How would you like to see the summary?",
            },
            "element": {
                "action_id": "input",
                "type": "radio_buttons",
                "initial_option": where_to_display_options[0],
                "options": where_to_display_options,
            },
        },
        {
            "type": "input",
            "block_id": "summary-type",
            "label": {
                "type": "plain_text",
                "text": "Type of summary",
            },
            "element": {
                "action_id": "input",
                "type": "radio_buttons",
                "initial_option": summary_type_options[0],
                "options": summary_type_options,
            },
        },
        {
            "type": "section",
            "block_id": "channel-selected",
            "text": {
                "type": "mrkdwn",
                "text": "Pick a channel from the dropdown list",
            },
            "accessory": {
                "action_id": "input",
                "type": "conversations_select",
                "default_to_current_conversation": True,
                "placeholder": {"type": "plain_text", "text": "Select a channel"},
                "filter": {"include": ["public", "private"]},
            },
        },
        {
            "type": "input",
            "block_id": "prompt",
            "optional": True,
            "element": {
                "type": "plain_text_input",
                "action_id": "input",
                "multiline": True,
                "placeholder": {"type": "plain_text", "text": prompt},
            },
            "label": {
                "type": "plain_text",
                "text": "Customize the prompt as you prefer:",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Note that after the instruction you provide, this app will append all the replies in the thread.",
                }
            ],
        },
    ]
    view = {
        "type": "modal",
        "callback_id": "request-channel-summary",
        "title": {"type": "plain_text", "text": "Summarize the channel"},
        "submit": {"type": "plain_text", "text": "Summarize"},
        "close": {"type": "plain_text", "text": "Close"},
        "blocks": blocks,
    }
    if is_error is True:
        del view["submit"]

    client.views_open(
        trigger_id=body.get("trigger_id"),
        view=view,
    )
    ack()


def extract_state_value(payload: dict, block_id: str, action_id: str = "input") -> dict:
    state_values = payload["state"]["values"]
    return state_values[block_id][action_id]


def ack_summarize_thread_options_modal_submission(
    ack: Ack, payload: dict, client: WebClient
):
    return ack_summarize_options_modal_submission(
        ack, payload, "request-thread-summary", client
    )


def ack_summarize_channel_options_modal_submission(
    ack: Ack, payload: dict, client: WebClient
):
    return ack_summarize_options_modal_submission(
        ack, payload, "request-channel-summary", client
    )


def ack_summarize_options_modal_submission(
    ack: Ack, payload: dict, callback_id: str, client: WebClient
):
    where_to_display = (
        extract_state_value(payload, "where-to-share-summary")
        .get("selected_option")
        .get("value", "modal")
    )

    method = ""
    if callback_id == "request-thread-summary":
        method = "thread"
    if callback_id == "request-channel-summary":
        method = "channel"

        channel_selected = extract_state_value(payload, "channel-selected").get(
            "selected_conversation"
        )
        try:
            # Test if this bot is in the channel
            client.conversations_history(
                channel=channel_selected,
                limit=1,
            )
        except SlackApiError as e:
            error_code = e.response["error"]
            if error_code == "not_in_channel":
                blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "It appears that this app's bot user is not a member of the specified channel. "
                            f"Could you please invite me to <#{channel_selected}> "
                            "to make this app functional?",
                        },
                    }
                ]
            else:
                blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"Something is wrong! (error: {error_code})",
                        },
                    }
                ]

            ack(
                response_action="update",
                view={
                    "type": "modal",
                    "callback_id": callback_id,
                    "title": {"type": "plain_text", "text": f"Summarize the {method}"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": blocks,
                },
            )
    if where_to_display == "modal":
        ack(
            response_action="update",
            view={
                "type": "modal",
                "callback_id": callback_id,
                "title": {"type": "plain_text", "text": f"Summarize the {method}"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": "Got it! Working on the summary now ... :hourglass:",
                        },
                    },
                ],
            },
        )
    else:
        ack(
            response_action="update",
            view={
                "type": "modal",
                "callback_id": callback_id,
                "title": {"type": "plain_text", "text": f"Summarize the {method}"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": f"Got it! Once the summary is ready, I will post it in the {method}.",
                        },
                    },
                ],
            },
        )


def build_thread_replies_as_combined_text(
    *,
    context: BoltContext,
    client: WebClient,
    channel: str,
    thread_ts: str,
) -> str:
    thread_content = ""
    for page in client.conversations_replies(
        channel=channel,
        ts=thread_ts,
        limit=1000,
    ):
        for reply in page.get("messages", []):
            user = reply.get("user")
            if user == context.bot_user_id:  # Skip replies by this app
                continue
            if user is None:
                user = client.bots_info(bot=reply.get("bot_id"))["bot"]["user_id"]
                if user is None or user == context.bot_user_id:
                    continue
            text = slack_to_markdown("".join(reply["text"].splitlines()))
            thread_content += f"<@{user}>: {text}\n"
    return thread_content


def build_channel_replies_as_combined_text(
    *,
    context: BoltContext,
    client: WebClient,
    channel: str,
    is_limited_by_time: bool,
) -> str:
    thread_content = ""
    thirty_days_from_now_unix = datetime.date.today() - datetime.timedelta(30)
    oldest = thirty_days_from_now_unix.strftime("%s")
    limit = 50
    if is_limited_by_time:
        n_days_from_now_unix = datetime.date.today() - datetime.timedelta(1)
        oldest = n_days_from_now_unix.strftime("%s")

    resp = client.conversations_history(channel=channel, limit=limit, oldest=oldest)
    for message in resp.get("messages"):
        user = message.get("user")
        if user == context.bot_user_id:  # Skip replies by this app
            continue
        if user is None:
            ## ignore bot users
            # user = client.bots_info(bot=reply.get("bot_id"))["bot"]["user_id"]
            if user is None or user == context.bot_user_id:
                continue
        text = slack_to_markdown("".join(message["text"].splitlines()))
        thread_content += f"<@{user}>: {text}\n"
    return thread_content


def prepare_and_share_channel_summary(
    payload: dict,
    client: WebClient,
    context: BoltContext,
    logger: logging.Logger,
):
    if not is_user_allowed(context=context, client=client):
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-channel-summary",
                "title": {"type": "plain_text", "text": "Summarize the channel"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Need to request permission to use",
                        },
                    },
                ],
            },
        )
        return

    try:
        where_to_display = (
            extract_state_value(payload, "where-to-share-summary")
            .get("selected_option")
            .get("value", "modal")
        )

        summary_type = (
            extract_state_value(payload, "summary-type")
            .get("selected_option")
            .get("value", "modal")
        )

        channel_selected = extract_state_value(payload, "channel-selected").get(
            "selected_conversation"
        )

        prompt = extract_state_value(payload, "prompt").get("value")

        thread_content = build_channel_replies_as_combined_text(
            context=context,
            client=client,
            channel=channel_selected,
            is_limited_by_time=summary_type == "last_nth_hours",
        )

        summary = generate_slack_thread_summary(
            context=context,
            logger=logger,
            prompt=prompt,
            thread_content=thread_content,
            timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        )

        summary = markdown_to_slack(summary)

        slack_user_id_format = f"<@{context.user_id}>"

        if where_to_display == "modal":
            client.views_update(
                view_id=payload["id"],
                view={
                    "type": "modal",
                    "callback_id": "request-channel-summary",
                    "title": {"type": "plain_text", "text": "Summarize the channel"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"Here is the summary\n\n{summary}",
                            },
                        },
                    ],
                },
            )

        summary_type_text = ""
        if summary_type == "last_nth_hours":
            summary_type_text = "of the last 72 hours"
        else:
            summary_type_text = "of the last 50 messages"
        if where_to_display == "ephemeral_reply":
            client.chat_postEphemeral(
                channel=channel_selected,
                user=context.user_id,
                text=f"{slack_user_id_format}, here is the request summary {summary_type_text}\n\n{summary}",
            )

        if where_to_display == "reply":
            client.chat_postMessage(
                channel=channel_selected,
                text=f"{slack_user_id_format}, here is the request summary {summary_type_text}\n\n{summary}",
            )
    except openai.APITimeoutError:
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-channel-summary",
                "title": {"type": "plain_text", "text": "Summarize the channel"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": TIMEOUT_ERROR_MESSAGE,
                        },
                    },
                ],
            },
        )
    except Exception as e:
        logger.error(f"Failed to share a channel summary: {e}")
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-channel-summary",
                "title": {"type": "plain_text", "text": "Summarize the channel"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":warning: My apologies! "
                            f"An error occurred while generating the summary of this channel: {e}",
                        },
                    },
                ],
            },
        )


def prepare_and_share_thread_summary(
    payload: dict,
    client: WebClient,
    context: BoltContext,
    logger: logging.Logger,
):
    if not is_user_allowed(context=context, client=client):
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-thread-summary",
                "title": {"type": "plain_text", "text": "Summarize the thread"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Need to request permission to use",
                        },
                    },
                ],
            },
        )
        return
    try:
        where_to_display = (
            extract_state_value(payload, "where-to-share-summary")
            .get("selected_option")
            .get("value", "modal")
        )
        prompt = extract_state_value(payload, "prompt").get("value")
        private_metadata = json.loads(payload.get("private_metadata"))
        thread_content = build_thread_replies_as_combined_text(
            context=context,
            client=client,
            channel=private_metadata.get("channel"),
            thread_ts=private_metadata.get("thread_ts"),
        )

        summary = generate_slack_thread_summary(
            context=context,
            logger=logger,
            prompt=prompt,
            thread_content=thread_content,
            timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        )

        summary = markdown_to_slack(summary)

        slack_user_id_format = f"<@{context.user_id}>"
        if where_to_display == "modal":
            client.views_update(
                view_id=payload["id"],
                view={
                    "type": "modal",
                    "callback_id": "request-thread-summary",
                    "title": {"type": "plain_text", "text": "Summarize the thread"},
                    "close": {"type": "plain_text", "text": "Close"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"Here is the summary\n\n{summary}",
                            },
                        },
                    ],
                },
            )

        if where_to_display == "ephemeral_reply":
            client.chat_postEphemeral(
                channel=private_metadata.get("channel"),
                thread_ts=private_metadata.get("thread_ts"),
                user=context.user_id,
                text=f"{slack_user_id_format}, here is the requested summary\n\n{summary}",
            )

        if where_to_display == "reply":
            client.chat_postMessage(
                channel=private_metadata.get("channel"),
                thread_ts=private_metadata.get("thread_ts"),
                text=f"{slack_user_id_format}, here is the requested summary\n\n{summary}",
            )

    except openai.APITimeoutError:
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-thread-summary",
                "title": {"type": "plain_text", "text": "Summarize the thread"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": TIMEOUT_ERROR_MESSAGE,
                        },
                    },
                ],
            },
        )
    except Exception as e:
        logger.error(f"Failed to share a thread summary: {e}")
        client.views_update(
            view_id=payload["id"],
            view={
                "type": "modal",
                "callback_id": "request-thread-summary",
                "title": {"type": "plain_text", "text": "Summarize the thread"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": ":warning: My apologies! "
                            f"An error occurred while generating the summary of this thread: {e}",
                        },
                    },
                ],
            },
        )


def register_listeners(app: App):
    app.event("message")(ack=just_ack, lazy=[respond_to_new_message])
    app.event("app_mention")(ack=just_ack, lazy=[respond_to_app_mention])

    # Summarize a thread
    app.shortcut("summarize-thread")(show_summarize_option_modal)
    app.view("request-thread-summary")(
        ack=ack_summarize_thread_options_modal_submission,
        lazy=[prepare_and_share_thread_summary],
    )
    app.shortcut("summarize-channel")(show_summarize_channel_option_modal)
    app.view("request-channel-summary")(
        ack=ack_summarize_channel_options_modal_submission,
        lazy=[prepare_and_share_channel_summary],
    )


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
