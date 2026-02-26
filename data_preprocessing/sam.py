import jittor as jt


class SAM:
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(list(params), **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._old_params = {}

    def _get_grad(self, p):
        return p.opt_grad(self.base_optimizer)

    def first_step(self, loss):
        self.base_optimizer.backward(loss)
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                g = self._get_grad(p)
                if g is None:
                    continue
                self._old_params[id(p)] = p.detach().clone()
                e_w = (jt.pow(p, 2) if self.adaptive else 1.0) * g * scale
                p.assign(p.detach() + e_w)

    def second_step(self, loss):
        self.base_optimizer.backward(loss)
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) in self._old_params:
                    p.assign(self._old_params[id(p)])
        self._old_params.clear()
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                g = self._get_grad(p)
                if g is not None:
                    v = (jt.abs(p) if self.adaptive else 1.0) * g
                    grads.append(v.flatten())
        if not grads:
            return jt.array(0.0)
        return jt.norm(jt.concat(grads, dim=0), p=2)

    @property
    def lr(self):
        return self.base_optimizer.lr

    @lr.setter
    def lr(self, value):
        self.base_optimizer.lr = value

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.base_optimizer.state_dict()
