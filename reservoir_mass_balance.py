import numpy as np
import matplotlib.pyplot as plt


class Reservoir:
    __capacity = -1
    __storage_area_curve = np.array([[], []])
    __evaporation_series = np.array([])
    __inflow_series = np.array([])
    __demand_series = np.array([])
    __stored_volume = np.array([])

    def __init__(self, storage_area_curve, evaporations, inflows, demands):
        """ Constructor for reservoir class

        Arguments:
            storage_area_curve {2D numpy matrix} -- Matrix with a
            row of storages and a row of areas
            evaporations {array/list} -- array of evaporations
            inflows {array/list} -- array of inflows
            demands {array/list} -- array of demands
        """

        self.__storage_area_curve = storage_area_curve
        assert(storage_area_curve.shape[0] == 2)
        assert(len(storage_area_curve[0]) == len(storage_area_curve[1]))

        self.__capacity = storage_area_curve[0, -1]
 
        n_weeks = len(demands)
        self.__stored_volume = np.ones(n_weeks, dtype=float) * self.__capacity
        self.__evaporation_series = evaporations
        self.__inflow_series = inflows
        self.__demand_series = demands
 
    def calculate_area(self, stored_volume):
        """ Calculates reservoir area based on its storage vs. area curve

        Arguments:
            stored_volume {float} -- current stored volume

        Returns:
            float -- reservoir area
        """

        storage_area_curve_T = self.__storage_area_curve.T .astype(float)

        if stored_volume > self.__capacity:
            print("Storage volume {} greater than capacity {}.".format(
                stored_volume, self.__capacity))
            raise ValueError

        for i in range(1, len(storage_area_curve_T)):
            s, a = storage_area_curve_T[i]
            # the &st; below needs to be replace with "smaller than" symbol. 
            # WordPress code highlighter has a bug that was distorting the 
            # code because of this "smaller than."
            if stored_volume < s:
                sm, am = storage_area_curve_T[i - 1]
                return am + (stored_volume - sm) / (s - sm) * (a - am)

        return a

    def mass_balance(self, upstream_flow, week):
        """ Perform mass balance on reservoir
        Stored volume is current stored volume - evaporation - demand

        Arguments:
            upstream_flow {float} -- release from upstream reservoir
            week {int} -- week

        Returns:
            double, double -- reservoir release and unfulfilled
                              demand (in case the reservoir gets empty)

        """
        if week < 1:
            print("Week must be >= 1, but was {}.".format(week))
            raise ValueError

        evaporation = self.__evaporation_series[week] *\
            self.calculate_area(self.__stored_volume[week - 1])
        new_stored_volume = self.__stored_volume[week - 1] + upstream_flow +\
            self.__inflow_series[week] - evaporation -\
            self.__demand_series[week]

        release = 0
        unfulfilled_demand = 0

        if (new_stored_volume > self.__capacity):
            release = new_stored_volume - self.__capacity
            new_stored_volume = self.__capacity
        elif (new_stored_volume < 0.):
            unfulfilled_demand = -new_stored_volume
            new_stored_volume = 0.

        self.__stored_volume[week] = new_stored_volume

        return release, unfulfilled_demand

    def get_stored_volume_series(self):
        """Return stored volume time series

        Returns:
            Numpy Array -- stored volumes over time.
        """
        return self.__stored_volume


def run_mass_balance(reservoirs, n_weeks):
    """Run mass balance.

    Arguments:
        reservoirs {List of Reservoir} -- list of reservoirs in the 
            order they are connected
        n_weeks {int} -- Number of weeks to simulate

    """
    for week in range(1, n_weeks):
        release, unfulfilled_demand, total_unfulfilled_demand = 0, 0, 0
        for reservoir in reservoirs:
            release, unfulfilled_demand = reservoir.mass_balance(release, week)
            total_unfulfilled_demand += unfulfilled_demand

    if unfulfilled_demand > 0:
        print("Total unfulfilled demand of {}".format(unfulfilled_demand))


def generate_streamflow(n_weeks, sin_amplitude, log_mu, log_sigma):
    """Log-normally distributed stream flow generator. Varies mean with sin(t).

    Arguments:
        n_weeks {int} -- number of weeks of stream flows
        sin_amplitude {double} -- amplitude of log-mean sinusoid fluctuation
        log_mu {double} -- mean log-mean
        log_sigma {double} -- log-sigma

    Returns:
        {Numpy array} -- stream flow series

    """
    streamflows = np.zeros(n_weeks)

    for i in range(n_weeks):
        # Transform standard normal into normal with specified sigma and mu.
        streamflow = np.random.randn() * log_sigma + log_mu *\
            (1. + sin_amplitude * np.sin(2. * np.pi / 52 * (i % 52)))
        streamflows[i] = np.exp(streamflow)

    return streamflows


if __name__ == "__main__":
    np.random.seed(0)
    n_weeks = 522
    sin_amplitude, log_mean, log_std = 1., 2.1, 1.8
    streamflows1 = generate_streamflow(n_weeks, sin_amplitude,
                                       log_mean, log_std)
    streamflows2 = generate_streamflow(n_weeks, sin_amplitude,
                                       log_mean / 2, log_std)
    storage_area_curve = np.array([[0, 1000, 3000, 4000], [0, 400, 600, 900]])

    reservoir1 = Reservoir(storage_area_curve, np.random.rand(n_weeks) / 8,
                           streamflows1, np.random.rand(n_weeks) * 25)
    reservoir2 = Reservoir(storage_area_curve, np.random.rand(n_weeks) / 8,
                           streamflows2, np.random.rand(n_weeks) * 25)

    reservoirs = [reservoir1, reservoir2]
    run_mass_balance(reservoirs, n_weeks)

    fig, axes = plt.subplots(len(reservoirs), sharex=True, sharey=True)
    for i in range(len(reservoirs)):
        axes[i].plot(reservoirs[i].get_stored_volume_series())
        axes[i].set_ylabel("Stored Volume [MG]")
        axes[i].set_title("Storage Over Time -- Reservoir {}".format(i + 1))
        axes[i].set_ylim(0, 4100)
    axes[-1].set_xlabel("Weeks [-]")
    plt.tight_layout()
    plt.show()
