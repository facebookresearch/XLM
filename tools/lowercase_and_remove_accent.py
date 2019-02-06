# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import unicodedata
import six


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming UTF-8 input.
    """
    if isinstance(text, six.text_type):
        return text
    elif isinstance(text, six.binary_type):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def run_strip_accents(text):
    """
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


for line in sys.stdin:
    line = convert_to_unicode(line.rstrip().lower())
    line = run_strip_accents(line)
    print(u'%s' % line.lower())
