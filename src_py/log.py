import logging, os, sys, settings
from pycrayon import CrayonClient
from pathlib import Path
os.environ['no_proxy'] = '127.0.0.1,localhost'

def init_logger(tensorboard=True, prepend_text=""):
    global logger, experimentLogger
    logger = logging.getLogger('heel-contour-prediction')
    
    #log file handler
    print(settings.opt)
    fileHandler = logging.FileHandler(os.path.join(settings.opt['save'], prepend_text + settings.opt['description'] + '.log'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler) 
    
    #output stream handler
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('file handler and stream handler are ready for logging')
    
    if(tensorboard == True):
        cc = CrayonClient(hostname="localhost")
        experimentLogger = cc.create_experiment(Path(settings.opt['save']).name)

    # log the configuration
    logger.info(settings.opt)
