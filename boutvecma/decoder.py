import os

from easyvvuq.decoders import BaseDecoder
from boutdata.data import BoutOptionsFile
from xbout.load import open_boutdataset


class BOUTDecoder(BaseDecoder, decoder_name="bout++"):
    def __init__(self, target_dir=None, target_filename=None, variables=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """
        self.target_dir = target_dir or "data"
        self.target_filename = target_filename or "BOUT.dmp.*.nc"

        # TODO: check it's an iterable or otherwise sensible type?
        self.variables = variables

    @staticmethod
    def _get_output_path(run_info=None, outfile=None):
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

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        data_files = self._get_output_path(run_info, self.target_filename)
        df = open_boutdataset(data_files)

        if self.variables is None:
            return df
        # For now, just return the variable itself
        # TODO: add analysis step
        return {variable: df[variable] for variable in self.variables}

    def get_restart_dict(self):
        return {
            "target_dir": self.target_dir,
            "target_filename": self.target_filename,
            "variables": self.variables,
        }

    @staticmethod
    def element_version():
        return "0.1.0"
