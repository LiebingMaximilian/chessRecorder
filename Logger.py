import datetime
from termcolor import colored


def getMessage(message):
    return datetime.datetime.now().__str__() + ": " + message


def logInfo(message):
    print("[INFO]: " + getMessage(message))


def logError(message):
    print(colored("[ERROR]: " + getMessage(message), 'red'))


def logWarning(message):
    print(colored("[WARN]: " + getMessage(   message), 'yellow'))
