from . import decoder

from easyvvuq.decoders import BaseDecoder
from boutdata.data import BoutOptions


def test_decoder_create():
    bout_decoder = decoder.SimpleBOUTDecoder()

    assert isinstance(bout_decoder, BaseDecoder)


def test_decoder_sim_complete(tmpdir):

    bout_decoder = decoder.SimpleBOUTDecoder()
    assert not bout_decoder.sim_complete({"run_dir": tmpdir})

    settings_file = BoutOptions()
    settings_file.getSection("run")["finished"] = "Now"
    with open(tmpdir.join("BOUT.settings"), "w") as f:
        f.write(str(settings_file))

    assert bout_decoder.sim_complete({"run_dir": tmpdir})


def test_decoder_parse_sim_output_all_variables(tmpdir, monkeypatch):
    def mock_open_boutdataset(data_files):
        return {"T": [1, 2, 3], "n": [4, 5, 6]}

    monkeypatch.setattr(decoder, "open_boutdataset", mock_open_boutdataset)

    bout_decoder = decoder.SimpleBOUTDecoder()

    data = bout_decoder.parse_sim_output({"run_dir": tmpdir})

    assert len(data.keys()) == 2
    assert data["T"] == [1, 2, 3]
    assert data["n"] == [4, 5, 6]


def test_decoder_parse_sim_output_variable_list(tmpdir, monkeypatch):
    def mock_open_boutdataset(data_files):
        return {"T": [1, 2, 3], "n": [4, 5, 6]}

    monkeypatch.setattr(decoder, "open_boutdataset", mock_open_boutdataset)

    bout_decoder = decoder.SimpleBOUTDecoder(variables=["T"])

    data = bout_decoder.parse_sim_output({"run_dir": tmpdir})

    assert len(data.keys()) == 1
    assert data["T"] == [1, 2, 3]
