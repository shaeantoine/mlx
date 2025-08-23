import mlx.distributions as dist
from torch import FloatTensor
from torch.distributions import normal

mlx_normal = dist.Normal(0,1)
print(mlx_normal.log_prob(27))


torch_loc = 0.0
torch_scale = 1.0
log_prob_ten = FloatTensor([27.0])
torch_normal = normal.Normal(torch_loc, torch_scale)
print(torch_normal.log_prob(log_prob_ten))