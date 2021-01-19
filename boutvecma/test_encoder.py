from . import encoder

import os
from easyvvuq.encoders import BaseEncoder
from boutdata.data import BoutOptionsFile


def test_create():
    bout_encoder = encoder.BOUTEncoder()

    assert isinstance(bout_encoder, BaseEncoder)


def test_encode_from_empty(tmpdir):
    bout_encoder = encoder.BOUTEncoder()

    newfile = tmpdir.join("BOUT.inp")
    bout_encoder.encode(
        params={"nx": 10, "conduction:chi": 12}, target_dir=tmpdir.strpath
    )

    # Check that the file exists
    assert os.path.exists(newfile.strpath)

    # Read and check contents
    options = BoutOptionsFile(newfile.strpath)
    assert options["nx"] == 10
    assert options["conduction"]["chi"] == 12


def test_encode_identity(tmpdir):
    # Create a template file
    template = tmpdir.join("template.inp")

    with open(template.strpath, "w") as f:
        f.write(
            """
        n = 10
        [section]
        key = 42
        """
        )

    bout_encoder = encoder.BOUTEncoder(template.strpath)

    newfile = tmpdir.join("BOUT.inp")
    bout_encoder.encode(target_dir=tmpdir.strpath)

    # Check that the file exists
    assert os.path.exists(newfile.strpath)

    # Read and check contents
    options = BoutOptionsFile(newfile.strpath)
    assert options["n"] == 10
    assert options["section"]["key"] == 42
