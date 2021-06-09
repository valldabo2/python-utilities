from dataclasses import dataclass, FrozenInstanceError


@dataclass(frozen=True)
class A:
    a: int


x = A(1)
try:
    x.a = 1
except FrozenInstanceError:
    pass
