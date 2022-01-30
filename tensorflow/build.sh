#!/bin/bash

bazel build --config=opt --config=cuda --strip=never //tensorflow/tools/pip_package:build_pip_package
