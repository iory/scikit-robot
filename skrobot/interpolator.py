import numpy as np


class Interpolator(object):

    def __init__(self):
        self.time = 0.0
        self.segment_time = 0.0
        self.segment = 0
        self.segment_num = 0
        self.is_interpolating = False

    def reset(self, position_list=None,
              time_list=None):
        """"Initialize interpolator.

        Args:
            position-list:
                list of control point
            time-list:
                list of time from start for each control point,
                time in first control point is zero, so length
                of this list is length of control point minus 1
        """
        if position_list is None:
            position_list = self.position_list
        else:
            self.position_list = position_list
        if time_list is None:
            time_list = self.time_list
        else:
            self.time_list = time_list
        if len(position_list) != len(time_list) + 1:
            raise ValueError(
                'length of position_list must be length of time_list + 1')

        self.time = 0.0
        self.segment_time = 0.0
        self.segment = 0
        self.segment_num = len(position_list) - 1
        self.stop_interpolation()

    def start_interpolation(self):
        self.is_interpolating = True

    def stop_interpolation(self):
        self.is_interpolating = False

    def interpolation(self):
        raise NotImplementedError

    def pass_time(self, dt):
        """process interpolation for dt[sec]

        Args:
            dt (float):
                sec order
        """
        if self.is_interpolating:
            self.position = self.interpolation()
            self.time += dt
            self.segment_time += dt
            if self.time > self.time_list[self.segment]:
                self.segment_time = (self.time - self.time_list[self.segment])
                self.segment += 1
            if self.segment >= self.segment_num:
                self.reset()
            return self.position


class LinearInterpolator(Interpolator):

    def __init__(self):
        Interpolator.__init__(self)

    def interpolation(self):
        """Linear Interpolation."""
        v1 = self.position_list[self.segment]
        v2 = self.position_list[self.segment + 1]
        if self.segment > 0:
            total_time = self.time_list[self.segment] - \
                self.time_list[self.segment - 1]
        else:
            total_time = self.time_list[self.segment]
        t1 = self.segment_time
        t2 = total_time - t1
        v1 = v1 * (t2 / total_time)
        v2 = v2 * (t1 / total_time)
        return v1 + v2


class MinjerkInterpolator(Interpolator):

    def __init__(self):
        Interpolator.__init__(self)

    def reset(self,
              velocity_list=None,
              acceleration_list=None,
              **kwargs):
        """Initialize interpolator

        Args:
            position_list:
                list of control point
            velocity-list:
                list of velocity in each control point
            acceleration-list:
                list of acceleration in each control point

        """
        Interpolator.reset(self, **kwargs)
        if velocity_list is None:
            self.velocity_list = [
                np.zeros(len(self.position_list[0]))
                for _ in range(self.segment_num + 1)]
        else:
            self.velocity_list = velocity_list
        if acceleration_list is None:
            self.acceleration_list = [
                np.zeros(len(self.position_list[0]))
                for _ in range(self.segment_num + 1)]
        else:
            self.acceleration_list = acceleration_list

    def interpolation(self):
        """Minjerk interpolator, a.k.a Hoff & Arbib."""
        xi = self.position_list[self.segment]
        xf = self.position_list[self.segment + 1]
        vi = self.velocity_list[self.segment]
        vf = self.velocity_list[self.segment + 1]
        ai = self.acceleration_list[self.segment]
        af = self.acceleration_list[self.segment + 1]
        if self.segment > 0:
            total_time = self.time_list[self.segment] - \
                self.time_list[self.segment - 1]
        else:
            total_time = self.time_list[self.segment]

        # A=(gx-(x+v*t+(a/2.0)*t*t))/(t*t*t)
        # B=(gv-(v+a*t))/(t*t)
        # C=(ga-a)/toi

        A = (xf - (xi + total_time * vi + (total_time ** 2)
                   * 0.5 * ai)) / (total_time ** 3)
        B = (vf - (vi + total_time * ai)) / (total_time ** 2)
        C = (af - ai) / total_time

        # a0=x
        # a1=v
        # a2=a/2.0
        # a3=10*A-4*B+0.5*C
        # a4=(-15*A+7*B-C)/t
        # a5=(6*A-3*B+0.5*C)/(t*t)

        a0 = xi
        a1 = vi
        a2 = 0.5 * ai
        a3 = 10 * A - 4 * B + 0.5 * C
        a4 = (-15 * A + 7 * B - C) / total_time
        a5 = (6 * A - 3 * B + 0.5 * C) / (total_time * total_time)

        # x=a0+a1*t+a2*t*t+a3*t*t*t+a4*t*t*t*t+a5*t*t*t*t*t
        # v=a1+2*a2*t+3*a3*t*t+4*a4*t*t*t+5*a5*t*t*t*t
        # a=2*a2+6*a3*t+12*a4*t*t+20*a5*t*t*t

        self.position = a0 + \
            self.segment_time ** 1 * a1 + \
            self.segment_time ** 2 * a2 + \
            self.segment_time ** 3 * a3 + \
            self.segment_time ** 4 * a4 + \
            self.segment_time ** 5 * a5

        self.velocity = a1 + \
            self.segment_time ** 1 * a2 + \
            self.segment_time ** 2 * a3 + \
            self.segment_time ** 3 * a4 + \
            self.segment_time ** 4 * a5

        self.acceleration = a2 + \
            self.segment_time ** 1 * a3 + \
            self.segment_time ** 2 * a4 + \
            self.segment_time ** 3 * a5
        return self.position


def position_list_interpolation(
        position_list, time_list, dt,
        interpolator=MinjerkInterpolator(),
        initial_time=0.0,
        neglect_first=False,
        vel_vector_list=None,
        acc_vector_list=None):
    data_list = []
    tm_list = []
    vel_data_list = []
    acc_data_list = []

    if vel_vector_list is None:
        vel_vector_list = []
    if acc_vector_list is None:
        acc_vector_list = []

    r = []
    for n in time_list:
        if len(r):
            r.append(n + r[-1])
        else:
            r.append(n)
    kwargs = dict(position_list=position_list,
                  time_list=r)
    if hasattr(interpolator, 'velocity'):
        kwargs['velocity_list'] = vel_vector_list
    if hasattr(interpolator, 'acceleration'):
        kwargs['acceleration_list'] = acc_vector_list
    interpolator.reset(**kwargs)
    interpolator.start_interpolation()
    while interpolator.is_interpolating:
        if interpolator.is_interpolating:
            tm_list.append(initial_time + interpolator.time)
        else:
            tm_list.append(dt + tm_list[0])
        interpolator.pass_time(dt)
        data_list.append(interpolator.position)

        if hasattr(interpolator, 'velocity'):
            vel_data_list.append(interpolator.velocity)
        if hasattr(interpolator, 'acceleration'):
            acc_data_list.append(interpolator.acceleration)

    if neglect_first:
        data_list = data_list[1:]
        tm_list = tm_list[1:]

    result_dict = dict(position=data_list,
                       time=tm_list)
    if hasattr(interpolator, 'velocity'):
        if neglect_first:
            result_dict['velocity'] = vel_data_list[1:]
        else:
            result_dict['velocity'] = vel_data_list
    if hasattr(interpolator, 'acceleration'):
        if neglect_first:
            result_dict['acceleration'] = acc_data_list[1:]
        else:
            result_dict['acceleration'] = acc_data_list
    return result_dict
