import torch


class OneStepMethod:
    """
    One step method class
    """

    def __init__(self, odefun, method):
        self.odefun = odefun
        self.step = self.set_method(method)

    def update(self, time, state, stepsize):
        """
        Perform one update step using the one step method specified in fun
        """
        delta = self.step(time, state, stepsize)
        new_state = state+stepsize*delta
        return(new_state)

    def set_method(self, method):
        """
        assign the chosen one step method
        """
        method_list = {'euler': self.euler,
                        'rk': self.rk,
                       'rk45': self.rk45
                       }
        return(method_list[method])

    def euler(self, time, state, stepsize):
        """ 
        Compute euler one step
        """
        delta = self.odefun(time, state)
        return(delta)

    def rk(self, time, state, stepsize):
        """
        Classical Runge Kutta method
        """
        # calculate support points
        k1 = self.odefun(time, state)
        k2 = self.odefun(time+0.5*stepsize, state+0.5*stepsize*k1)
        k3 = self.odefun(time+0.5*stepsize, state+0.5*stepsize*k2)
        k4 = self.odefun(time+stepsize, state+stepsize*k3)
        # get increment
        delta = (k1+2*k2+2*k3+k4)/6.0
        return(delta)

    def rk45(self, time, state, stepsize):
        pass


def solve_ode(odefun, initial, num_steps, stepsize, method='euler'):
    """
    solve ode approximately by using iter 
    repetitions of a one-step method
    """
    # preparations
    dim = len(initial)
    sol = torch.zeros(num_steps+1, dim)
    sol[0] = initial.clone().detach()
    time = torch.zeros(num_steps+1)
    solver = OneStepMethod(odefun, method)
    # iterate
    for i in range(num_steps):
        time[i+1] = time[i]+stepsize
        sol[i+1] = solver.update(time[i], sol[i], stepsize)
    return(time, sol) 

def solve_ode_grid(odefun, initial, time, method='euler'):
    """
    solve ode approximately by using iter 
    repetitions of a one-step method
    """
    # preparations
    dim = len(initial)
    num_steps = len(time)
    sol = torch.zeros(num_steps+1, dim)
    sol[0] = initial.clone().detach()
    time = torch.zeros(num_steps+1)
    solver = OneStepMethod(odefun, method)
    # iterate
    for i in range(num_steps):
        time[i+1] = time[i]+stepsize
        sol[i+1] = solver.update(time[i], sol[i], stepsize)
    return(time, sol) 