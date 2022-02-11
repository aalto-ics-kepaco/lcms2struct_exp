####
#
# The MIT License (MIT)
#
# Copyright 2021 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import os
import sqlite3


def get_backup_db(input_db_fn: str, postfix: str = "NEW", exists: str = "overwrite") \
        -> sqlite3.Connection:
    """
    Copy the input database (DB) into a new DB. That is useful when we want to make modifications to a DB and keep the
    old version.

    :param input_db_fn: string, filename of the input database.

    :param postfix: string, string appended to the original DB filename to create a output DB filename.

    :param exists: string, indicating what to do if the backup DB already exists:
        "overwrite": Remove existing and re-create backup DB
        "reuse": Simply open the existing backup DB
        "raise": Raise an error if the backup DB already exisits

    :return: sqlite3.Connection, to the new DB
    """
    ibasename = os.path.basename(input_db_fn)  # path/to/db__property.sqlite --> db__property.sqlite
    name, ext = ibasename.split(os.path.extsep)   # db__property.sqlite --> (db__property, sqlite)
    name = name.split("__")[0]  # db__property --> db

    ofn = os.path.join(
        os.path.dirname(input_db_fn),
        os.path.extsep.join(["__".join([name, postfix]), ext])
    )   # path/to/db_NEW.sqlite

    print("DB backup '%s' --> '%s'" % (ibasename, os.path.basename(ofn)))

    # Handle existing backup DB
    if os.path.exists(ofn):
        if exists == "overwrite":
            os.remove(ofn)
        elif exists == "reuse":
            oconn = sqlite3.connect(ofn)
            return oconn
        elif exists == "raise":
            raise RuntimeError("Output database already exists: '%s'" % ofn)
        else:
            raise ValueError("Invalid input for 'exists': %s" % exists)

    # Make backup
    iconn = sqlite3.connect("file:" + input_db_fn + "?mode=ro", uri=True)
    oconn = sqlite3.connect(ofn)
    iconn.backup(oconn)

    return oconn


