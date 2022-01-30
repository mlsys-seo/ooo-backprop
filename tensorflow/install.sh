#!/bin/bash

bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp

# If there is tensorflow already, remove the package and install the newly created package.
pip uninstall -y tensorflow
pip install ~/tmp/tensorflow-2.4.0-cp36-cp36m-linux_x86_64.whl

# There may be an issue about protocal buffer package version. Downgrade the package to the version 3.10.0
pip uninstall -y protobuf
pip install protobuf==3.10.0

