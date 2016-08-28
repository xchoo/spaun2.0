import numpy as np


class Forcefield(object):
    def __init__(self, arm):
        self.arm = arm
        self.max_force = 1000

    def generate(self, u):
        return np.zeros(self.arm.DOF)


class NoForcefield(Forcefield):
    def __init__(self, arm):
        super(NoForcefield, self).__init__(arm)

    def generate(self, u):
        return np.zeros(self.arm.DOF)


class ConstantForcefield(Forcefield):
    def __init__(self, arm):
        super(ConstantForcefield, self).__init__(arm)
        self.scale = 100

    def generate(self, u):
        return np.arange(self.arm.DOF) * self.scale


class QVelForcefield(Forcefield):
    def __init__(self, arm):
        super(QVelForcefield, self).__init__(arm)
        self.force_matrix = np.array([[-10.1, -11.1, -10.1],
                                      [-11.2, 11.1, 10.1],
                                      [-11.2, 11.1, -10.1]])

    def generate(self, u):
        """Generate the signal to add to the control signal.
        u np.array: the outgoing control signal
        arm Arm: the arm currently being controlled
        """
        scale = 20
        # calculate force to add
        force = np.dot(self.force_matrix * scale,
                       self.arm.dq)

        # translate to joint torques
        return np.maximum(np.minimum(force, self.max_force), -self.max_force)


class XYVelForcefield(Forcefield):
    def __init__(self, arm):
        super(XYVelForcefield, self).__init__(arm)
        scale = 20
        self.force_matrix = np.array([[-10.1, -11.2],
                                      [-11.2, 11.1]]) * scale

    def generate(self, u):
        """Generate the signal to add to the control signal.

        u np.array: the outgoing control signal
        arm Arm: the arm currently being controlled
        """

        JEE = self.arm.gen_jacEE()
        # calculate end-effector velocity
        dx = np.dot(JEE, self.arm.dq)

        # calculate force to add
        force = np.dot(JEE.T, np.dot(self.force_matrix, dx))

        # translate to joint torques
        return np.maximum(np.minimum(force, self.max_force), -self.max_force)
