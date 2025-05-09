from src.enums import HurstMethodType, GarchMethodType
from src.launcher import Launcher

# config = {"hurst_method": HurstMethodType.SCALED_WINDOWED_VARIANCE,
#           "data": ["s&p500", "ssec", "ftse100", "ftsemib"],
#           "dates": {"Sub-Primes": ["01-07-2007", "01-09-2009"],
#                     "COVID-2019": ["31-12-2020", "31-12-2021"],
#                     "Ukraine War": ["24-02-2022", "24-02-2023"]},
#           "params": {"hurst_params": {"method": "SD", "exclusions": True},#{"max_scale": 20},#,
#                      "window": 1000}}
config = {"hurst_method": HurstMethodType.SCALED_WINDOWED_VARIANCE,
          "garch_method": GarchMethodType.DCC,
          "data": ["ftsemib", "ssec", "ftse100"],
          "dates": {"Sub-Primes": ["01-07-2007", "01-09-2009"]},
          "params": {"hurst_params": {"method": "SD", "exclusions": True},"window": 100}}

launcher = Launcher(**config)
test = launcher.run_process()