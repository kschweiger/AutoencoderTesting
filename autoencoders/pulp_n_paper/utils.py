import coloredlogs
import logging
import os

def initLogging(thisLevel, funclen=30):
    """
    Helper function for setting up python logging
    """
    log_format = ('[%(asctime)s] %(funcName)-'+str(funclen)+'s %(levelname)-8s %(message)s')
    if thisLevel == 20:
        thisLevel = logging.INFO
    elif thisLevel == 10:
        thisLevel = logging.DEBUG
    elif thisLevel == 30:
        thisLevel = logging.WARNING
    elif thisLevel == 40:
        thisLevel = logging.ERROR
    elif thisLevel == 50:
        thisLevel = logging.CRITICAL
    else:
        thisLevel = logging.NOTSET

    # logging.basicConfig(
    #     format=log_format,
    #     level=thisLevel,
    # )

    coloredlogs.install(
        level=thisLevel,
        fmt=log_format
    )

    return True
