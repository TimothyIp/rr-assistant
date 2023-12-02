import logging
from dotenv import load_dotenv
import os
from slack_bolt import App, BoltContext
from slack_bolt.adapter.socket_mode import SocketModeHandler
from bolt_listeners import before_authorize, register_listeners
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request
from slack_bolt.adapter.flask import SlackRequestHandler


load_dotenv()


logging.basicConfig(level="INFO")

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    before_authorize=before_authorize,
    process_before_response=True,
)

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


# Listens to incoming messages that contain "hello"
# To learn available listener arguments,
# visit https://slack.dev/bolt-python/api-docs/slack_bolt/kwargs_injection/args.html
@app.message("hello")
def message_hello(message, say):
    # say() sends a message to the channel where the event was triggered
    say(f"Hey there <@{message['user']}>!")


flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

app.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=2))
register_listeners(app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


@app.middleware
def set_openai_api_key(context: BoltContext, next_):
    context["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    context["OPENAI_MODEL"] = os.environ["OPENAI_MODEL"]
    next_()
