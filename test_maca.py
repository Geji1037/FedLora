# test_maca.py
from triton.backends.metax.driver import maca_home_dirs
home = maca_home_dirs()
print("MACA_HOME detected:", home)

if home is None:
    print("❌ 失败：未找到 MACA_HOME，请检查环境变量或安装路径")
else:
    print("✅ 成功：已识别到沐曦 SDK 路径")