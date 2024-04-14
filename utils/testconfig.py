import sys
import os
from hydra.experimental import compose, initialize_config_dir


class TestConfig():
    """
    hydraによる設定値の取得 (conf)
    """
    @staticmethod
    def get_cnf():
        """
        設定値の辞書を取得
        @return
            cnf: OmegaDict
        """
        conf_dir = os.path.join(os.getcwd(), "conf")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir): # 1.2では、version_base=Noneという引数も必要になっていた
            cnf = compose(config_name="default.yaml")
            return cnf
