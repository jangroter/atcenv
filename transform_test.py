from atcenv.MASAC_transform.masac_agent import MaSacAgent
from atcenv.MASAC_transform.mactor_critic import Actor
import torch

actor = Actor(4,2)

b, t, k = 2, 4, 4

x = torch.randn((b, t, k))

y, _ = actor.forward(x)
print(y)

