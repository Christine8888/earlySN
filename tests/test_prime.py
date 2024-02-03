import unittest
#from earlySN import is_prime
from earlySN import dataset

#def test_false():
#    assert is_prime(10) is False

#def test_true():
#    assert is_prime(11) is True

#def test_yao_load():
#    print("Testing import of default Yao et al. 2019 dataset")
#    yao = dataset.Dataset(default = 'yao', name = 'Yao et al. 2019')
#    assert yao.data.index.name == 'SN'

class test_prime(unittest.TestCase):

    def test_hello_world(self):
        assert "h" in "hello world"
