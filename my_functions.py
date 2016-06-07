import time

def maketime_sec(x):
    """ returns time in seconds from timestamp """
    new1 = time.strptime(x,"%Y-%m-%d %H:%M:%S")
    new2 = time.mktime(new1)
    return new2
