import logging
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
