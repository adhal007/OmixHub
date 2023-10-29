import logging 
import inspect
import sys 
## have flexibility to pass log level
class CustomLogger:
    def custlogger(self, loglevel=logging.DEBUG): 
        ## Set class or method name from where logger is called 
        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        logger_name = f'{the_class}'
        # print(inspect.stack())
        ## create logger  
        logger = logging.getLogger(logger_name)
        logger.setLevel(loglevel)

        ## create a console handler or file handler and set the log level
        fh = logging.FileHandler("out.log")
        stdout = logging.StreamHandler(stream=sys.stdout)
        ## create formatter - how you want your logs to be formatted
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s', datefmt='%d/%m//%Y %I:%M:%s %p')

        fh.setFormatter(formatter)
        stdout.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(stdout)
        return logger 
    # Create a base class

    def log(self):
        stack = inspect.stack()
        try:
            print("Whole stack is:")
            print("\n".join([str(x[4]) for x in stack]))
            print("-"*20)
            print("Caller was %s" %(str(stack[2][4])))
        finally:
            del stack

## An easy logger to include class name
## Does not work with my setup currently 
## Might want to figure out the exact handling in logger module 
class LoggingHandler:
    def __init__(self, *args, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)