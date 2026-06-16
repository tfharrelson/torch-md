from torch_md.data.config import SourcesConfig
from assertpy import assert_that


class TestSourcesConfig:
    def test_get(self):
        config = SourcesConfig.get()
        assert_that(config.omol25).is_not_none()
