# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pythainlp.tokenize import word_tokenize

for line in sys.stdin.readlines():
    line = line.rstrip('\n')
    print(' '.join(word_tokenize(line)))
