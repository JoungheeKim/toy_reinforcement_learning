from argparse import ArgumentParser
import logging

##LOGGING PROPERTY
LOG_FILE = 'logfile'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG

def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--device", dest="device", metavar="device", default="gpu")
    parser.add_argument("--env",dest="env", metavar="env", default="BreakoutDeterministic-v4")
    parser.add_argument("--memory_size", dest="memory_size", metavar="memory_size", default=1000000)
    parser.add_argument("--update_freq", dest="update_freq", metavar="update_freq", default=4)
    parser.add_argument("--learn_start", dest="learn_start", metavar="learn_start", default=50000)
    parser.add_argument("--history_size", dest="history_size", metavar="history_size", default=2)
    parser.add_argument("--target_update", dest="target_update", metavar="target_update", default=10000)

    ##Learning rate
    parser.add_argument("--batch_size", dest="batch_size", metavar="batch_size", default=32)
    parser.add_argument("--ep", dest="ep", metavar="ep", default=1)
    parser.add_argument("--eps_end", dest="eps_end", metavar="eps_end", default=0.1)
    parser.add_argument("--eps_endt", dest="eps_endt", metavar="eps_endt", default=1000000)
    parser.add_argument("--lr", dest="lr", metavar="lr", default=0.00025)
    parser.add_argument("--discount", dest="discount", metavar="discount", default=0.99)


    parser.add_argument("--agent_type", dest="agent_type", metavar="agent_type", default="DQN_ln")
    parser.add_argument("--max_steps", dest="max_steps", metavar="max_steps", default=10000000)
    parser.add_argument("--eval_freq", dest="eval_freq", metavar="eval_freq", default=10000)
    parser.add_argument("--eval_steps", dest="eval_steps", metavar="eval_steps", default=50000)
    return parser