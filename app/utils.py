
class FeatureLoss_Wass():
    def __init__(self):
        super().__init__()
    def make_features(self, x, clone=False):
        return []
    def forward(self, input, target):
        return target.mean()
    def __del__(self): self.hooks.remove()

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def get_resize(y, z, max_size):
    if y*2 <= max_size and z*2 <= max_size:
        y_new = y*2
        z_new = z*2
    else:
        if y > z:
            y_new = max_size
            z_new = int(round_up_to_even(z * max_size / y))

        else:
            z_new = max_size
            y_new = int(round_up_to_even(y * max_size / z))
    return (y_new, z_new)