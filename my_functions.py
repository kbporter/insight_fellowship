import time

def maketime_sec(x):
    """ returns time in seconds from timestamp """
    new1 = time.strptime(x,"%Y-%m-%d %H:%M:%S")
    new2 = time.mktime(new1)
    return new2

def isactive(x):
	""" determines whether user is active based on # days since last activity, cutoff is 3 weeks """
	if x > 21:
		x = 0
	else:
		x = 1
	return x
