import logging
from math import log,exp
def LOG(s:str)->None:
    """
    print s as LOG Format
    """
    print("LOG {}".format(s))
    logger=logging.getLogger("*")
    print(logger.info(s))


def UTF8ToUnicodeText(c:chr)->str:
    """ return unicode of c
    padding 0 to len==4 like 0x0000
    """
    return ord(c)
    return "{:#06x}".format(ord(c))

def UnicodeCharToUTF8(x:int)->chr:
    return chr(x)


def LogSumExp(x,y,init):
    kMinusLogEpsilon = 50
    if init:
        return y
    vmin,vmax=min(x,y),max(x,y)

    if vmax>vmin+kMinusLogEpsilon:
        return vmax
    else:
        return vmax+log(exp(vmin-vmax)+1)


def Digamma(x:float)->float:
    result=0
    while x<7:
        result-=1/x
        x+=1

    x-=1/2
    xx=1/x
    xx2=xx*xx
    xx4 = xx2*xx2

    result += log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 +(31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4
    return result

