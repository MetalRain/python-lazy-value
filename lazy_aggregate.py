import random
from lazy_value.models import LazyValue

def lazy_aggregate():
    sum = LazyValue(0, evaluated=False)
    rows = [
        dict(a=1),
        dict(a=2),
        dict(a=3),
        dict(a=4),
        dict(a=5),
        dict(a=6),
    ]

    for row in rows:
        row['sum'] = sum
        row['perc'] = LazyValue(lambda a, b: a/b, row['a'], sum)
        sum += row['a']

    # calculates lazy values dependent on sum
    sum.evaluate()

    for row in rows:
        print(row)

if __name__ == '__main__':
    lazy_aggregate()