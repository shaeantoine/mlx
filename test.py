import mlx.distributions as dist

dist_normal = dist.Normal(0,1)
print(dist_normal.log_prob(27))