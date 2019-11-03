import logging
import os


def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fh = logging.FileHandler(os.path.join(save_dir, filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger