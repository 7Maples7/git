# -*- coding: utf-8 -*-
"""PyCharm-friendly launcher for the DClsEcho dataset viewer.

Right-click this file in PyCharm and choose Run. Change DEFAULT_DATA_PATH if you
want the viewer to open another dataset by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent if (HERE.parent / "radar_three_cls").is_dir() else HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_three_cls.echo_dataset_viewer import EchoDatasetViewer


DEFAULT_DATA_PATH = (
    "D:\\A3\u56de\u6ce2\u5f55\u53d6\u6570\u636e\\"
    "\u7b2c\u4e94\u7248\u534f\u8bae\u4e0b\u70b9\u8ff9\u5f55\u53d6\u6e05\u6d17\u540e\u7684\u6570\u636e\\"
    "\u6570\u636e\u96c626_06_06_15_35_14"
)


def main() -> None:
    data_path = DEFAULT_DATA_PATH if Path(DEFAULT_DATA_PATH).exists() else ""
    app = EchoDatasetViewer(initial_path=data_path)
    app.mainloop()


if __name__ == "__main__":
    main()
