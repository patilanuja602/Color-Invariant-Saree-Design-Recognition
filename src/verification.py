from . import config as C
def verify(ea, eb, threshold=C.DEFAULT_THRESHOLD):
    s = float(ea @ eb)
    return dict(similarity=s, same=s >= threshold, threshold=threshold)
