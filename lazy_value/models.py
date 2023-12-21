from typing import Any, Callable, List, Union

# Lazy aggregator framework

Primitive = Union[bool, int, float, str]
primitive_tuple = (bool, int, float, str)


class LazyValue:
    evaluated: bool
    dependees: List[object]
    value: Primitive
    parameters: List[object]
    fn: Callable[..., Primitive]

    def __init__(self, *args, **kwargs):
        self.evaluated = kwargs.get('evaluated', False) is True
        self.dependees = []

        argc = len(args)
        if argc >= 1:
            arg = args[0]
        
        # Primitive value
        if argc == 1 and isinstance(arg, primitive_tuple):
            self.value: Primitive = arg
            self.evaluated = True
            return

        # Partial function
        if callable(arg):
            self.fn = arg
            if argc > 1:
                self.parameters = args[1:]
                # If parameter is lazy value
                # become dependent on it
                for p in self.parameters:
                    if isinstance(p, LazyValue):
                        self.bind_into(p)
            else:
                self.parameters = []
        else:
            raise Exception('Invalid calling convention, use either primitive value or partial function')

    def bind_into(self, dependency):
        # Make this values dependent on other
        dependency.dependees.append(self)
    
    def evaluate(self) -> Primitive:
        if self.evaluated == True:
            return self.value
        if self.fn:
            # Eval values this depends on
            raw_params = [
                p.evaluate() if isinstance(p, LazyValue) else p
                for p in self.parameters
            ]
            # Eval self
            self.value = self.fn(*raw_params)
        self.evaluated = True
        # Eval values that depend on this
        for d in self.dependees:
            d.evaluate()
        return self.value

    def invalidate(self):
        self.evaluated = False
        return self

    def __add__(self, value):
        if isinstance(value, LazyValue):
            raise Exception('Would return new LazyValue')
        else:
            self.value += value
            return self

    def __sub__(self, value):
        if isinstance(value, LazyValue):
            raise Exception('Would return new LazyValue')
        else:
            self.value -= value
            return self
    
    def __mul__(self, value):
        if isinstance(value, LazyValue):
            raise Exception('Would return new LazyValue')
        else:
            self.value *= value
            return self
    
    def __div__(self, value):
        if isinstance(value, LazyValue):
            raise Exception('Would return new LazyValue')
        else:
            self.value /= value
            return self

    def __repr__(self):
        if hasattr(self, 'fn'):
            return f'LazyValue({self.fn.__repr__()}, {", ".join([p.__repr__() for p in self.parameters])})'
        else:
            return f'LazyValue({self.value})'
