import math

def softmax(scores: list[float]) -> list[float]:
    result=[]
    max_=max(scores)
    total=sum([(math.exp(x-max_)) for x in scores])
    for i in scores:
        result.append((math.exp(i-max_))/total)
    return result
