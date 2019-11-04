import numpy as np
from pymbvi.models.model import Model
from pymbvi import util as ut
from scipy.linalg import solve
#from pymbvi.uitl import 


class PartitionControl(Model):
    """
    Class for MJPs where the control agrees with the partitioning and there is one parameter for each partition
    """

    def kl_prior(self, time, control, forward, rates):
        # transform control
        exp_control = np.exp(control)
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # compute kl div
        eff_control = 1.0-exp_control+exp_control*control
        kl = propensities * eff_control
        return(kl)

    def rcontrol_gradient(self, time, control, forward, backward, rates):
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # contribution from the constraint
        dim_forward = forward.shape[0:3]+(-1,)
        constraint_grad = self.constraint_gradient(control, forward, backward, rates)
        constraint_grad = ut.integrate_subsamples(time, constraint_grad)
        # contribution from prior kl
        control_grad = propensities*control*np.exp(control)-constraint_grad
        return(control_grad)

    def control_gradient(self, time, control, forward, backward, rates):
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # contribution from the constraint
        dim_forward = forward.shape[0:3]+(-1,)
        constraint_grad = self.constraint_gradient(control, forward, backward, rates)
        constraint_grad = ut.integrate_subsamples(time, constraint_grad)
        # contribution from the prior kl divergence
        control_grad = control-constraint_grad/(propensities*np.exp(control))
        return(control_grad)

    def rates_gradient(self, time, control, forward, backward, rates):
        """
        Joint natural gradient with respect to controls and rates
        This is a simplified version that only uses the diagonal elements of the fisher information tensor
        """
        # get propensities
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        eff_prop = propensities.reshape(forward.shape[:-1]+(-1,))*np.expand_dims(1-np.exp(control)+np.exp(control)*control, axis=2)
        eff_prop = ut.integrate(time, eff_prop)
        # contribution from the constraint
        dim_forward = forward.shape[0:3]+(-1,)
        constraint_grad = self.constraint_rates_gradient(control, forward, backward, rates)
        constraint_grad = ut.integrate(time, constraint_grad)
        # contribution from the prior kl divergence
        fisher_g = propensities.reshape(forward.shape[:-1]+(-1,))*np.expand_dims(np.exp(control), axis=2)
        fisher_g = ut.integrate(time, fisher_g)
        #print(eff_prop)
        rates_grad = (eff_prop - constraint_grad)/fisher_g
        return(rates_grad)

    def joint_gradient(self, time, control, forward, backward, rates):
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        dim_forward = forward.shape[0:3]+(-1,)
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        # control grad contribution from the constraint
        constraint_control_grad, constraint_rates_grad = self.constraint_full_gradient(control, forward, backward, rates)
        constraint_control_grad = ut.integrate_subsamples(time, constraint_control_grad)
        constraint_rates_grad = ut.integrate(time, constraint_rates_grad)
        # contribution from the prior kl divergence
        eff_prop = ut.integrate_subsamples(time, propensities, dim)
        control_g = eff_prop*np.exp(control)
        #control_grad = control-constraint_control_grad/control_g
        control_grad = eff_prop*control*np.exp(control)-constraint_control_grad
        # compute fisher information for the rates
        rates_g = propensities.reshape(forward.shape[:-1]+(-1,))*np.expand_dims(np.exp(control), axis=2)
        rates_g = ut.integrate(time, rates_g)
        # rates grad contribution from the constraint
        eff_prop = propensities.reshape(forward.shape[:-1]+(-1,))*np.expand_dims(1-np.exp(control)+np.exp(control)*control, axis=2)
        eff_prop = ut.integrate(time, eff_prop)        
        #rates_grad = (eff_prop - constraint_rates_grad)/rates_g
        rates_grad = eff_prop - constraint_rates_grad
        # global fisher transform of joint gradient
        #self.compute_fisher_g(control_g, rates_g)
        control_grad, rates_grad = self.fisher_transform(control_grad, rates_grad, control_g, rates_g)
        return(control_grad, rates_grad)

    def compute_fisher_g(self, g_control, g_rates):
        """
        Joint global fisher g for rates and control
        """
        dim = g_control.shape
        # control vs control contribution
        dim_0 = (dim[0], 1, 1, dim[0], 1, 1)
        dim_1 = (1, dim[1], 1, 1, dim[1], 1)
        dim_2 = (1, 1, dim[2], 1, 1, dim[2])
        g_u_u = np.eye(dim[0]).reshape(dim_0) * np.eye(dim[1]).reshape(dim_1) * np.eye(dim[2]).reshape(dim_2) * g_control.reshape(dim+(1, 1, 1))
        g_u_u = g_u_u.reshape((np.prod(dim), np.prod(dim)))
        # control vs rates contribution
        g_u_c = np.expand_dims(g_control, axis=-1)*np.eye(dim[2]).reshape((1, 1, dim[2], dim[2]))
        g_u_c = g_u_c.reshape((np.prod(dim), dim[2]))
        # rates vs rates contribution
        g_c_c = np.diag(g_rates)
        # construct full matrix
        fisher_g = np.block([[g_u_u, g_u_c], [g_u_c.T, g_c_c]])
        return(fisher_g)

    def fisher_transform(self, control_grad, rates_grad, control_g, rates_g):
        # compute fisher information
        fisher_g = self.compute_fisher_g(control_g, rates_g)
        # transform grad
        grad = np.concatenate([control_grad.flatten(), rates_grad])
        # apply inverse fisher information
        #natural_grad = solve(fisher_g, grad, sym_pos=True)
        natural_grad = ut.lstsq_reg(fisher_g, grad, k=1e-6)
        #natural_grad = np.linalg.lstsq(fisher_g, grad)[0]
        # split again and return
        control_grad = natural_grad[0:control_grad.size].reshape(control_grad.shape)
        rates_grad = natural_grad[control_grad.size:]
        print(np.linalg.norm(control_grad))
        print(np.linalg.norm(rates_grad))
        return(control_grad, rates_grad)


class PartitionControlOld(Model):
    """
    Class for MJPs where the control agrees with the partitioning and there is one parameter for each partition
    """

    def kl_prior(self, time, control, forward, rates):
        # transform control
        exp_control = np.exp(control)
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # compute kl div
        eff_control = 1.0-exp_control+exp_control*control
        kl = propensities * eff_control
        return(kl)

    def rcontrol_gradient(self, time, control, forward, backward, rates):
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # contribution from the constraint
        dim_forward = forward.shape[0:3]+(-1,)
        constraint_grad = self.constraint_gradient(control, forward, backward, rates)
        constraint_grad = ut.integrate_subsamples(time, constraint_grad)
        # contribution from prior kl
        control_grad = propensities*control*np.exp(control)-constraint_grad
        return(control_grad)

    def control_gradient(self, time, control, forward, backward, rates):
        # get propensities
        dim = forward.shape[:3]+control.shape[-1:]
        propensities = self.natural_moments(forward.reshape((-1, forward.shape[-1])), rates)
        propensities = ut.integrate_subsamples(time, propensities, dim)
        # contribution from the constraint
        dim_forward = forward.shape[0:3]+(-1,)
        constraint_grad = self.constraint_gradient(control, forward, backward, rates)
        constraint_grad = ut.integrate_subsamples(time, constraint_grad)
        # contribution from the prior kl divergence
        control_grad = control-constraint_grad/(propensities*np.exp(control))
        return(control_grad)

    def rates_gradient(self, time, control, forward, backward, rates):
        """
        Joint natural gradient with respect to controls and rates
        """
        raise NotImplementedError
