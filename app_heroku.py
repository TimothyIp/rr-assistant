import logging
from dotenv import load_dotenv
import os
from slack_bolt import App, BoltContext
from slack_bolt.adapter.socket_mode import SocketModeHandler
from bolt_listeners import before_authorize, register_listeners
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
from slack_bolt.adapter.flask import SlackRequestHandler
from flask import Flask, request


load_dotenv()


logging.basicConfig(level="INFO")

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    before_authorize=before_authorize,
    process_before_response=True,
)


app.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=2))
register_listeners(app)

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


@app.middleware
def set_openai_api_key(context: BoltContext, next_):
    context["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
    context["OPENAI_MODEL"] = os.environ["OPENAI_MODEL"]
    next_()
