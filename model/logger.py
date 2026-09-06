import logging, socket, os
# logging 记录日志

model_dir = os.environ.get("MODEL_DIR", "./workdir/ellipsoid_symmetric") # get the value of the environment variable "MODEL_DIR", if not set, use "./workdir/default"

class Rank(logging.Filter): # custom logging filter class named Rank that inherits from logging.Filter.
    def filter(self, record):
        record.global_rank = os.environ.get("GLOBAL_RANK", 0)
        record.local_rank = os.environ.get("LOCAL_RANK", 0)
        return True # Returns True to allow the log record to proceed (never blocks any logs).


def get_logger(name):
    logger = logging.Logger(name)
    # logger.addFilter(Rank())
    level = {"crititical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}[
        os.environ.get("LOGGER_LEVEL", "info")
    ]
    logger.setLevel(level) # Sets the minimum level for this logger. Only messages at this level or higher will be processed.

    ch = logging.StreamHandler() # outputs log messages to the console
    ch.setLevel(logging.INFO)
    os.makedirs(model_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(model_dir, "log.out")) # Creates a file handler that writes logs to a file named log.out inside model_dir.
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(f'%(asctime)s [{socket.gethostname()}:%(process)d:%(global_rank)s:%(local_rank)s]
    # [%(levelname)s] %(message)s') #  (%(name)s)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    ) # Timestamp, hostname, process ID, log level, and the actual log message.
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def log_args_and_help(logger, args, parser):
    """Extracts help strings from the parser and logs formatted arguments."""
    help_dict = {action.dest: action.help for action in parser._actions if action.dest}

    logger.info("=" * 100)
    logger.info("STARTING SCRIPT WITH ARGUMENTS:")
    logger.info("=" * 100)
    for arg_name, arg_value in vars(args).items():
        help_text = help_dict.get(arg_name, "No description available.")
        # Formats the string so columns are padded and aligned nicely
        logger.info(f"  --{arg_name:<20}: {str(arg_value):<15} | Help: {help_text}")
    logger.info("=" * 100)