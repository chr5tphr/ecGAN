from argparse import ArgumentParser

commands = {}
def register_command(func):
    commands[func.__name__] = func
    return func

def call(argv):
    parser = ArgumentParser()
    
    parser.add_argument('command',choices=commands.keys())

    args,rargv = parser.parse_known_args(argv)

    commands[args.command](rargv)

@register_command
def train(argv):
    parser = ArgumentParser()
    parser.add_argument('-f','--config')

    args = parser.parse_args(argv)
