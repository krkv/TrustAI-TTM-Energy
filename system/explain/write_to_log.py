"""A simple logging routine to store inputs to the system."""
from datetime import datetime
from pytz import timezone, utc

from flask import Flask
import gin

app = Flask(__name__)


def pst():
    # From https://gist.github.com/vladwa/8cd97099e32c1088025dfaca5f1bfd33
    date_format = '%m_%d_%Y_%H_%M_%S_%Z'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime = date.strftime(date_format)
    return pstDateTime


@gin.configurable
def log_dialogue_input(log_dict, dynamodb_table):
    """Logs dialogue input to file."""
    if not isinstance(log_dict, dict):
        raise NameError(f"Logging information must be dictionary, not type {type(log_dict)}")

    # Log in PST
    log_dict["time"] = pst()
    if dynamodb_table is not None:
        try:
            dynamodb_table.put_item(Item=log_dict)
            app.logger.info("DB write successful")
        except Exception as e:
            app.logger.info(f"Could not write to database: {e}")
    # If no db is specified, write logs to info
    app.logger.info(log_dict)


def load_aws_keys(filepath):
    with open(filepath, "r") as file:
        data = file.readlines()
    return {"access_key": data[0].replace("\n", ""), "secret_key": data[1].replace("\n", "")}

