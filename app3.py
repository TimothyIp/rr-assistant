import logging
from dotenv import load_dotenv
import os
from slack_bolt import App, BoltContext
from slack_bolt.adapter.socket_mode import SocketModeHandler
from bolt_listeners import before_authorize, register_listeners
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler


load_dotenv()

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    before_authorize=before_authorize,
    process_before_response=True,
)


# Listens to incoming messages that contain "hello"
# To learn available listener arguments,
# visit https://slack.dev/bolt-python/api-docs/slack_bolt/kwargs_injection/args.html
@app.message("hello")
def message_hello(message, say):
    # say() sends a message to the channel where the event was triggered
    say(f"Hey there <@{message['user']}>!")


# Start your app
if __name__ == "__main__":
    app.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=2))
    register_listeners(app)

    @app.middleware
    def set_openai_api_key(context: BoltContext, next_):
        context["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        context["OPENAI_MODEL"] = os.environ["OPENAI_MODEL"]
        next_()

    # Initializes your app with your bot token and socket mode handler
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
    logging.basicConfig(level="INFO")

    PORT = os.environ["PORT"]
    # if (port == null || port == "") {
    #   port = 8000;
    # }
    if not PORT:
        PORT = 8000
    app.listen(PORT)
