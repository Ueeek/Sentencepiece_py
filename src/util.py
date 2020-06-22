import logging
from math import log, exp


def UTF8ToUnicodeText(c: chr) -> str:
    """ return unicode of c
    padding 0 to len==4 like 0x0000
    """
    return ord(c)
    return "{:#06x}".format(ord(c))


def UnicodeCharToUTF8(x: int) -> chr:
    """ reverese operation of UTF8ToUnicodeText"""
    return chr(x)


def LogSumExp(x: float, y: float, init: bool) -> float:
    if init:
        return y
    kMinusLogEpsilon = 50
    vmin, vmax = min(x, y), max(x, y)

    if vmax > vmin+kMinusLogEpsilon:
        return vmax
    else:
        return vmax+log(exp(vmin-vmax)+1)


def Digamma(x: float) -> float:
    result = 0
    while x < 7:
        result -= 1/x
        x += 1

    x -= 1/2
    xx = 1/x
    xx2 = xx*xx
    xx4 = xx2*xx2

    result += log(x) + (1.0 / 24.0) * xx2 - (7.0 / 960.0) * xx4 + \
        (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4
    return result

def isValidCodepoint(c:chr)->bool:
    v=ord(c)
    return v<0xD800 or 0xE000<=v<=0x10FFFF

def isValidSentencePiece(sb:str)->bool:
    if len(sb)<=0:
        return False

    is_number= 0x30<=ord(sb)<=0x39

    for c in sb:
        if ord(c)==0x0000:
            return False
        if ord(c)==0x0020:
            return False
        if isValidCodepoint(c):
            return False
        #originalはunicode平仮名、katakanaをhanに変換してる

def arg_parser(args,key,default_val=None,required=False):
    """
    argsのなかにkeyがあるならその値をないならdefaultの値をreturnする。

    default_val: if not specified, None is set
    """
    if key in args.keys():
        return args[key]
    else:
        assert required==False,"arg ::{}:: is requiread"
        return default_val
