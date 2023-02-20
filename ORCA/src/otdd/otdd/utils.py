import os
import sys
import pickle as pkl
import pdb
import shutil
import logging
import tempfile

def launch_logger(console_level='warning'):
    ############################### Logging Config #################################
    ## Remove all handlers of root logger object -> needed to override basicConfig above
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO) # Has to be min of all the others

    ## create file handler which logs even debug messages, use random logfile name
    logfile = tempfile.NamedTemporaryFile(prefix="otddlog_", dir='/tmp').name
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)

    ## create console handler with a higher log level
    ch = logging.StreamHandler(stream=sys.stdout)
    if console_level == 'warning':
        ch.setLevel(logging.WARNING)
    elif console_level == 'info':
        ch.setLevel(logging.INFO)
    else:
        raise ValueError()
    ## create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    _logger.addHandler(fh)
    _logger.addHandler(ch)
    ################################################################################
    return _logger

def safedump(d,f):
    try:
        pkl.dump(d, open(f, 'wb'))
    except:
        pdb.set_trace()

def append_to_file(fname, l):
    with open(fname, "a") as f:
        f.write('\t'.join(l) + '\n')

def delete_if_exists(path, typ='f'):
    if typ == 'f' and os.path.exists(path):
        os.remove(path)
    elif typ == 'd' and os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("Unrecognized path type")
