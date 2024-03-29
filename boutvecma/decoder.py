import os

from boutdata.data import BoutOptionsFile
from xbout.load import open_boutdataset
import inspect
import re
import copy
import numpy as np


def flatten_dataframe_for_JSON(df):
    """Flatten a pandas dataframe to a plain list, suitable for JSON serialisation"""
    return df.values.flatten().tolist()


class BaseBOUTDecoder:
    def __init__(self, target_filename=None):
        """
        Parameters
        ==========
        target_filename: str or None
            Filename or glob to read in (default: "BOUT.dmp.*.nc")
        """
        self.target_filename = target_filename or "BOUT.dmp.*.nc"

    @staticmethod
    def _get_output_path(run_info=None, outfile=None):
        """
        Get the path the run directory, and optionally the file outfile
        """
        if run_info is None:
            raise RuntimeError("Passed 'None' to 'run_info'")

        run_path = run_info.get("run_dir", "data")

        if not os.path.isdir(run_path):
            raise RuntimeError(f"Run directory does not exist: {run_path}")

        return os.path.join(run_path, outfile)

    def sim_complete(self, run_info=None):
        """Return True if the simulation has finished"""
        settings_filename = self._get_output_path(run_info, "BOUT.settings")

        if not os.path.isfile(settings_filename):
            return False

        # Check for normal, clean finish
        settings_file = BoutOptionsFile(settings_filename)

        return "run:finished" in settings_file

    def get_outputs(self, run_info):
        """Read the BOUT++ outputs into an xarray dataframe"""
        data_files = self._get_output_path(run_info, self.target_filename)
        return open_boutdataset(data_files, info=False)

    def get_restart_dict(self):
        """Serialise the class to a dict of JSON serialisable variables"""
        # This is possibly too magic... get the variable names from
        # the __init__ method
        init_vars = inspect.signature(self.__init__).parameters.keys()
        # Assuming that the __init__ does `self.var = var`, we can
        # just return a dict of everything
        try:
            return {var: getattr(self, var) for var in init_vars}
        except AttributeError as e:
            # Let's try to give a helpful error message if this does go wrong
            match = re.search(r"no attribute '(.*)'", str(e))
            if match is None:
                # Oops, something else
                raise e
            varname = match.group(1)
            raise AttributeError(
                f"'{varname}' doesn't seem to be a member of this instance.\nDid you miss 'self.{varname} = {varname}' in '__init__'?"
            )

    @staticmethod
    def element_version():
        return "0.1.0"


class SimpleBOUTDecoder(BaseBOUTDecoder):
    """Just collects individual variables at the last timestep"""

    def __init__(self, target_filename=None, variables=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """
        super().__init__(target_filename=target_filename)

        # TODO: check it's an iterable or otherwise sensible type?
        self.variables = variables

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        return {
            variable: flatten_dataframe_for_JSON(df[variable][-1, ...])
            for variable in self.variables
        }

    @staticmethod
    def element_version():
        return "0.1.0"


class SampleLocationBOUTDecoder(BaseBOUTDecoder):
    """Samples variables at single points or areas"""

    def __init__(self, sample_locations):
        """
        sample_locations should be a list of dicts, each like:

            {
              "variable": "T",  # variable to sample
              "output_name": "T_centre",  # name in the output
              # Following are the indices to sample
              "x": 0,
              "y": 50,
              "z":0
            }
        """
        super().__init__()

        self.sample_locations = sample_locations

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        samples = copy.deepcopy(self.sample_locations)

        return {
            variable.pop("output_name"): flatten_dataframe_for_JSON(
                df[variable.pop("variable")][variable]
            )
            for variable in samples
        }

    @staticmethod
    def element_version():
        return "0.1.0"


class LogDataBOUTDecoder(BaseBOUTDecoder):
    """Returns log(variable)"""

    def __init__(self, target_filename=None, variables=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """
        super().__init__(target_filename=target_filename)

        self.variables = variables

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        return {
            variable: flatten_dataframe_for_JSON(np.log(df[variable][-1, ...]))
            for variable in self.variables
        }

    @staticmethod
    def element_version():
        return "0.1.0"


class AbsLogErrorBOUTDecoder(BaseBOUTDecoder):
    """Returns log(variable)"""

    def __init__(self, target_filename=None, variables=None, error_value=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """

        super().__init__(target_filename=target_filename)

        self.variables = variables
        self.error_value = error_value

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        return {
            variable: flatten_dataframe_for_JSON(
                np.log(np.abs(df[variable][-1, 0, 50, 0] - self.error_value))
            )
            for variable in self.variables
        }

    @staticmethod
    def element_version():
        return "0.1.0"


class AbsErrorBOUTDecoder(BaseBOUTDecoder):
    """Returns log(variable)"""

    def __init__(self, target_filename=None, variables=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """

        super().__init__(target_filename=target_filename)

        self.variables = variables

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        return {
            variable: flatten_dataframe_for_JSON(
                np.abs(df[variable][-1, 0, 50, 0] - 0.2810848142553153)
            )
            for variable in self.variables
        }

    @staticmethod
    def element_version():
        return "0.1.0"


class Blob2DDecoder(BaseBOUTDecoder):
    def __init__(self, use_peak=True):
        super().__init__()
        self.use_peak = use_peak

    def peak_index_position(self, n):
        return {k: v.compute() for k, v in n.argmax(dim=("x", "z")).items()}

    def position(self, df, indices):
        return df.x[indices["x"]], df.z[indices["z"]]

    def velocity(self, position):
        return position[0].differentiate("t"), position[1].differentiate("t")

    def com_index_position(self, n):
        size = n.shape

        indices = {"x": np.zeros(size[0], dtype=int), "z": np.zeros(size[0], dtype=int)}
        for i in range(len(n.t)):
            data = n[i, :, :] - n[0, 0, 0]  # use corner cell rather than nmin
            ntot = np.sum(data[:, :])

            indices["z"][i] = int(
                np.sum(np.sum(data[:, :], axis=0) * (np.arange(size[2]))) // ntot
            )
            indices["x"][i] = int(
                np.sum(np.sum(data[:, :], axis=1) * (np.arange(size[1]))) // ntot
            )

        return indices

    def parse_sim_output(self, run_info):
        df = self.get_outputs(run_info).squeeze()

        peak_indices = self.peak_index_position(df.n)
        peak_position = self.position(df, peak_indices)
        peak_velocity = self.velocity(peak_position)

        com_indices = self.com_index_position(df.n)
        com_position = self.position(df, com_indices)
        com_velocity = self.velocity(com_position)

        return {
            "peak_x": peak_position[0].flatten().tolist(),
            "peak_z": peak_position[1].flatten().tolist(),
            "peak_v_x": peak_velocity[0].flatten().tolist(),
            "peak_v_z": peak_velocity[1].flatten().tolist(),
            "com_x": com_position[0].flatten().tolist(),
            "com_z": com_position[1].flatten().tolist(),
            "com_v_x": com_velocity[0].flatten().tolist(),
            "com_v_z": com_velocity[1].flatten().tolist(),
        }
