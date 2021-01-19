import os
from copy import deepcopy
from easyvvuq.encoders import BaseEncoder
from boutdata.data import BoutOptions, BoutOptionsFile


class BOUTEncoder(BaseEncoder, encoder_name="bout++"):
    def __init__(self, template_input=None):
        """Read an existing BOUT.inp file to use as a template.
        If no input is given, an empty set of options will be created

        Example
        -------

        from boutvecma.encoder import BOUTEncoder
        encoder = BOUTEncoder("data/BOUT.inp")
        """
        if template_input:
            self._options = BoutOptionsFile(template_input)
        else:
            self._options = BoutOptions()

        self.template_input = template_input

    def encode(self, params=None, target_dir=""):
        """Create a BOUT.inp file in target_dir with modified parameters.

        Sub-sections are specified using colons in the params keys

        Example
        -------

        encoder.encode({"section:key":42}, target_dir="data")

        modifies key so that the file "data/BOUT.inp" contains

        [section]
        key = 42

        """
        if params:
            options = deepcopy(self._options)

            for key, value in params.items():
                options[key] = value
        else:
            options = self._options

        # Note: Here options could be BoutOptions or BoutOptionsFile
        # so can't just use options.write
        with open(os.path.join(target_dir, "BOUT.inp"), "w") as f:
            f.write(str(options))

    def element_version(self):
        return "0.1"

    def get_restart_dict(self):
        return {'template_input':self.template_input}
