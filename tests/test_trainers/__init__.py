import pprint
import unittest


class TestMI(unittest.TestCase):

    def setUp(self):
        return None

    def test_config(self):
        self.read_config = get_config_from_file(experiment_name='mi',
                                                experiment_type='multivariate_gaussian',
                                                experiment_indentifier="mi_unittest2")
        pprint(asdict(self.read_config.dataloader))
        pprint(asdict(self.read_config.binary_classifier))
        pprint(asdict(self.read_config.trainer))

    def test_load(self):
        return None

